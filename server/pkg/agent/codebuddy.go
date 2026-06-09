package agent

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// codebuddyBackend implements Backend by spawning the CodeBuddy Code CLI
// with --output-format stream-json and parsing its NDJSON event stream.
type codebuddyBackend struct {
	cfg Config
}

func (b *codebuddyBackend) Execute(ctx context.Context, prompt string, opts ExecOptions) (*Session, error) {
	execPath := b.cfg.ExecutablePath
	if execPath == "" {
		execPath = "codebuddy"
	}
	if _, err := exec.LookPath(execPath); err != nil {
		return nil, fmt.Errorf("codebuddy executable not found at %q: %w", execPath, err)
	}

	timeout := opts.Timeout
	runCtx, cancel := runContext(ctx, timeout)

	args := buildCodeBuddyArgs(prompt, opts, b.cfg.Logger)

	cmd := exec.CommandContext(runCtx, execPath, args...)
	hideAgentWindow(cmd)
	b.cfg.Logger.Info("agent command", "exec", execPath, "args", args)
	cmd.WaitDelay = 10 * time.Second
	if opts.Cwd != "" {
		cmd.Dir = opts.Cwd
	}
	cmd.Env = buildCodeBuddyEnv(b.cfg.Env)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("codebuddy stdout pipe: %w", err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, fmt.Errorf("codebuddy stdin pipe: %w", err)
	}
	var closeStdinOnce sync.Once
	closeStdin := func() { closeStdinOnce.Do(func() { _ = stdin.Close() }) }
	// Capture stderr into both the daemon log and a bounded tail buffer so
	// we can include the last few KB in Result.Error when codebuddy exits
	// unexpectedly.
	stderrBuf := newStderrTail(newLogWriter(b.cfg.Logger, "[codebuddy:stderr] "), agentStderrTailBytes)
	cmd.Stderr = stderrBuf

	if err := cmd.Start(); err != nil {
		closeStdin()
		cancel()
		return nil, fmt.Errorf("start codebuddy: %w", err)
	}

	b.cfg.Logger.Info("codebuddy started", "pid", cmd.Process.Pid, "cwd", opts.Cwd, "model", opts.Model)

	msgCh := make(chan Message, 256)
	resCh := make(chan Result, 1)

	// Write the user prompt via stdin in a separate goroutine so it cannot
	// deadlock against the stdout reader. With --verbose --output-format
	// stream-json the CLI emits a startup banner before reading its first
	// stdin frame; if nothing is draining stdout while we write the prompt,
	// the child blocks writing stdout, never reads stdin, and our Write
	// blocks until runCtx fires.
	writeDone := make(chan error, 1)
	go func() {
		err := writeCodeBuddyInput(stdin, prompt)
		if err != nil {
			closeStdin()
		}
		writeDone <- err
	}()

	// Close stdout when the context is cancelled so scanner.Scan() unblocks.
	go func() {
		<-runCtx.Done()
		_ = stdout.Close()
	}()

	go func() {
		defer cancel()
		defer closeStdin()
		defer close(msgCh)
		defer close(resCh)

		// Emit a status message immediately so the daemon knows the
		// backend is alive and doesn't hit the first-turn no-progress
		// timeout. The real session ID arrives with the first system
		// event, but the upstream drain loop only needs any message to
		// reset the idle watchdog.
		trySend(msgCh, Message{Type: MessageStatus, Status: "running"})

		startTime := time.Now()
		var output strings.Builder
		var sessionID string
		finalStatus := "completed"
		var finalError string
		usage := make(map[string]TokenUsage)

		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}

			var evt codebuddyStreamEvent
			if err := json.Unmarshal([]byte(line), &evt); err != nil {
				continue
			}

			switch evt.Type {
			case "assistant":
				b.handleAssistant(evt, msgCh, &output, usage)
			case "user":
				b.handleUser(evt, msgCh)
			case "system":
				if evt.SessionID != "" {
					sessionID = evt.SessionID
				}
				trySend(msgCh, Message{Type: MessageStatus, Status: "running", SessionID: sessionID})
			case "result":
				if evt.ResultText != "" {
					output.Reset()
					output.WriteString(evt.ResultText)
				}
				if evt.IsError {
					finalStatus = "failed"
					finalError = evt.ResultText
				}
			}
		}

		// If stdin write failed, surface it so we don't silently hide a
		// deadlock that consumed the full timeout.
		if writeErr := <-writeDone; writeErr != nil && finalStatus == "completed" && finalError == "" {
			finalError = fmt.Sprintf("codebuddy stdin write failed: %v", writeErr)
			finalStatus = "failed"
		}

		waitErr := cmd.Wait()
		duration := time.Since(startTime)

		switch {
		case runCtx.Err() == context.DeadlineExceeded:
			finalStatus = "timeout"
			finalError = fmt.Sprintf("codebuddy timed out after %s", timeout)
		case runCtx.Err() == context.Canceled:
			finalStatus = "aborted"
			finalError = "execution cancelled"
		case waitErr != nil && finalStatus == "completed":
			finalStatus = "failed"
			finalError = fmt.Sprintf("codebuddy exited with error: %v", waitErr)
		}

		// Include stderr tail in the error message for diagnostics when the
		// child exits unexpectedly (crash, startup flag rejection, etc.).
		if finalStatus == "failed" || finalStatus == "aborted" {
			if tail := stderrBuf.Tail(); tail != "" {
				finalError += fmt.Sprintf("\n[stderr tail]\n%s", tail)
			}
		}

		b.cfg.Logger.Info("codebuddy finished", "pid", cmd.Process.Pid, "status", finalStatus, "duration", duration.Round(time.Millisecond).String())

		resCh <- Result{
			Status:     finalStatus,
			Output:     output.String(),
			Error:      finalError,
			DurationMs: duration.Milliseconds(),
			SessionID:  sessionID,
			Usage:      usage,
		}
	}()

	return &Session{Messages: msgCh, Result: resCh}, nil
}

func (b *codebuddyBackend) handleAssistant(evt codebuddyStreamEvent, ch chan<- Message, output *strings.Builder, usage map[string]TokenUsage) {
	var content codebuddyMessageContent
	if err := json.Unmarshal(evt.Message, &content); err != nil {
		return
	}

	if content.Usage != nil && content.Model != "" {
		u := usage[content.Model]
		u.InputTokens += content.Usage.InputTokens
		u.OutputTokens += content.Usage.OutputTokens
		u.CacheReadTokens += content.Usage.CacheReadInputTokens
		u.CacheWriteTokens += content.Usage.CacheCreationInputTokens
		usage[content.Model] = u
	}

	for _, block := range content.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				output.WriteString(block.Text)
				trySend(ch, Message{Type: MessageText, Content: block.Text})
			}
		case "thinking":
			if block.Text != "" {
				trySend(ch, Message{Type: MessageThinking, Content: block.Text})
			}
		case "tool_use":
			var input map[string]any
			if block.Input != nil {
				_ = json.Unmarshal(block.Input, &input)
			}
			trySend(ch, Message{
				Type:   MessageToolUse,
				Tool:   block.Name,
				CallID: block.ID,
				Input:  input,
			})
		}
	}
}

func (b *codebuddyBackend) handleUser(evt codebuddyStreamEvent, ch chan<- Message) {
	var content codebuddyMessageContent
	if err := json.Unmarshal(evt.Message, &content); err != nil {
		return
	}

	for _, block := range content.Content {
		if block.Type == "tool_result" {
			resultStr := ""
			if block.Content != nil {
				resultStr = string(block.Content)
			}
			trySend(ch, Message{
				Type:   MessageToolResult,
				CallID: block.ToolUseID,
				Output: resultStr,
			})
		}
	}
}

// ── CodeBuddy stream-json event types ──
// CodeBuddy Code uses a stream-json protocol similar to Claude Code.

type codebuddyStreamEvent struct {
	Type      string          `json:"type"`
	Message   json.RawMessage `json:"message,omitempty"`
	SessionID string          `json:"session_id,omitempty"`

	// result fields
	ResultText string `json:"result,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`
}

type codebuddyMessageContent struct {
	Role    string                  `json:"role"`
	Model   string                  `json:"model"`
	Content []codebuddyContentBlock `json:"content"`
	Usage   *codebuddyUsage         `json:"usage,omitempty"`
}

type codebuddyUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}

type codebuddyContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
}

// ── Arg builder ──

// codebuddyBlockedArgs are flags hardcoded by the daemon that must not be
// overridden by user-configured custom_args.
var codebuddyBlockedArgs = map[string]blockedArgMode{
	"-p":                blockedWithValue, // non-interactive prompt
	"--output-format":   blockedWithValue, // stream-json protocol
	"--input-format":    blockedWithValue, // stream-json protocol
	"--permission-mode": blockedWithValue, // bypassPermissions for autonomous operation
}

func buildCodeBuddyArgs(prompt string, opts ExecOptions, logger *slog.Logger) []string {
	args := []string{
		"-p",
		"--output-format", "stream-json",
		"--input-format", "stream-json",
		"--verbose",
		"--permission-mode", "bypassPermissions",
	}
	if opts.Model != "" {
		args = append(args, "--model", opts.Model)
	}
	if opts.MaxTurns > 0 {
		args = append(args, "--max-turns", fmt.Sprintf("%d", opts.MaxTurns))
	}
	if opts.SystemPrompt != "" {
		args = append(args, "--append-system-prompt", opts.SystemPrompt)
	}
	if opts.ResumeSessionID != "" {
		args = append(args, "--resume", opts.ResumeSessionID)
	}
	args = append(args, prompt)
	args = append(args, filterCustomArgs(opts.ExtraArgs, codebuddyBlockedArgs, logger)...)
	args = append(args, filterCustomArgs(opts.CustomArgs, codebuddyBlockedArgs, logger)...)
	return args
}

// buildCodeBuddyEnv wraps buildEnv and adds CodeBuddy-specific environment
// defaults. Filters out internal CodeBuddy runtime markers that would
// interfere with child process execution.
func buildCodeBuddyEnv(extra map[string]string) []string {
	return buildEnv(extra)
}

// writeCodeBuddyInput sends the user prompt to codebuddy via stdin using the
// stream-json protocol. The format is identical to Claude Code:
// {"type":"user","message":{"role":"user","content":[{"type":"text","text":"..."}]}}
func writeCodeBuddyInput(w io.Writer, prompt string) error {
	data, err := buildCodeBuddyInput(prompt)
	if err != nil {
		return err
	}
	if _, err := w.Write(data); err != nil {
		return err
	}
	return nil
}

func buildCodeBuddyInput(prompt string) ([]byte, error) {
	payload := map[string]any{
		"type": "user",
		"message": map[string]any{
			"role": "user",
			"content": []map[string]string{
				{
					"type": "text",
					"text": prompt,
				},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("build codebuddy input: %w", err)
	}
	return append(data, '\n'), nil
}
