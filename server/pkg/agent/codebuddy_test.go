package agent

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ── Event parsing tests ──

func TestCodeBuddyHandleAssistantText(t *testing.T) {
	t.Parallel()

	raw := `{"role":"assistant","model":"deepseek-v4-pro-ioa","content":[{"type":"text","text":"hello world"}],"usage":{"input_tokens":100,"output_tokens":20,"cache_read_input_tokens":10,"cache_creation_input_tokens":5}}`
	var content codebuddyMessageContent
	if err := json.Unmarshal([]byte(raw), &content); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if content.Role != "assistant" {
		t.Errorf("role = %q, want assistant", content.Role)
	}
	if content.Model != "deepseek-v4-pro-ioa" {
		t.Errorf("model = %q, want deepseek-v4-pro-ioa", content.Model)
	}
	if len(content.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(content.Content))
	}
	if content.Content[0].Type != "text" || content.Content[0].Text != "hello world" {
		t.Errorf("content[0] = %+v, want text block 'hello world'", content.Content[0])
	}
	if content.Usage == nil {
		t.Fatal("usage is nil")
	}
	if content.Usage.InputTokens != 100 {
		t.Errorf("input_tokens = %d, want 100", content.Usage.InputTokens)
	}
	if content.Usage.OutputTokens != 20 {
		t.Errorf("output_tokens = %d, want 20", content.Usage.OutputTokens)
	}
	if content.Usage.CacheReadInputTokens != 10 {
		t.Errorf("cache_read = %d, want 10", content.Usage.CacheReadInputTokens)
	}
	if content.Usage.CacheCreationInputTokens != 5 {
		t.Errorf("cache_write = %d, want 5", content.Usage.CacheCreationInputTokens)
	}
}

func TestCodeBuddyHandleAssistantToolUse(t *testing.T) {
	t.Parallel()

	raw := `{"role":"assistant","model":"claude-sonnet-4.6","content":[{"type":"tool_use","id":"toolu_01","name":"Bash","input":{"command":"ls"}}]}`
	var content codebuddyMessageContent
	if err := json.Unmarshal([]byte(raw), &content); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(content.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(content.Content))
	}
	block := content.Content[0]
	if block.Type != "tool_use" {
		t.Errorf("type = %q, want tool_use", block.Type)
	}
	if block.ID != "toolu_01" {
		t.Errorf("id = %q, want toolu_01", block.ID)
	}
	if block.Name != "Bash" {
		t.Errorf("name = %q, want Bash", block.Name)
	}
}

func TestCodeBuddyHandleUserToolResult(t *testing.T) {
	t.Parallel()

	raw := `{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01","content":"file1.txt\nfile2.txt"}]}`
	var content codebuddyMessageContent
	if err := json.Unmarshal([]byte(raw), &content); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(content.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(content.Content))
	}
	block := content.Content[0]
	if block.Type != "tool_result" {
		t.Errorf("type = %q, want tool_result", block.Type)
	}
	if block.ToolUseID != "toolu_01" {
		t.Errorf("tool_use_id = %q, want toolu_01", block.ToolUseID)
	}
}

func TestCodeBuddyHandleAssistantThinking(t *testing.T) {
	t.Parallel()

	raw := `{"role":"assistant","model":"deepseek-v4-pro-ioa","content":[{"type":"thinking","text":"Let me think about this..."}]}`
	var content codebuddyMessageContent
	if err := json.Unmarshal([]byte(raw), &content); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(content.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(content.Content))
	}
	block := content.Content[0]
	if block.Type != "thinking" {
		t.Errorf("type = %q, want thinking", block.Type)
	}
	if block.Text != "Let me think about this..." {
		t.Errorf("text = %q, want 'Let me think about this...'", block.Text)
	}
}

func TestCodeBuddyStreamEventParsing(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		line     string
		wantType string
	}{
		{
			name:     "system init",
			line:     `{"type":"system","subtype":"init","session_id":"abc-123","model":"deepseek-v4-pro-ioa"}`,
			wantType: "system",
		},
		{
			name:     "assistant message",
			line:     `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}`,
			wantType: "assistant",
		},
		{
			name:     "user tool result",
			line:     `{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"id1","content":"output"}]}}`,
			wantType: "user",
		},
		{
			name:     "result success",
			line:     `{"type":"result","subtype":"success","is_error":false,"result":"done","total_cost_usd":0}`,
			wantType: "result",
		},
		{
			name:     "result error",
			line:     `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"something went wrong"}`,
			wantType: "result",
		},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			var evt codebuddyStreamEvent
			if err := json.Unmarshal([]byte(c.line), &evt); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if evt.Type != c.wantType {
				t.Errorf("type = %q, want %q", evt.Type, c.wantType)
			}
		})
	}
}

func TestCodeBuddyResultEventFields(t *testing.T) {
	t.Parallel()

	raw := `{"type":"result","subtype":"success","is_error":false,"result":"output text","uuid":"uuid-1","session_id":"sess-1","duration_ms":9500,"duration_api_ms":8700,"num_turns":2,"total_cost_usd":0.05,"usage":{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":10,"cache_read_input_tokens":90}}`
	var evt codebuddyStreamEvent
	if err := json.Unmarshal([]byte(raw), &evt); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if evt.Type != "result" {
		t.Errorf("type = %q, want result", evt.Type)
	}
	if evt.IsError {
		t.Error("is_error should be false")
	}
	if evt.ResultText != "output text" {
		t.Errorf("result = %q, want 'output text'", evt.ResultText)
	}
	if evt.SessionID != "sess-1" {
		t.Errorf("session_id = %q, want sess-1", evt.SessionID)
	}
}

// ── Arg builder tests ──

func TestBuildCodeBuddyArgsDefaults(t *testing.T) {
	t.Parallel()

	args := buildCodeBuddyArgs("hello", ExecOptions{}, slog.Default())
	if len(args) < 5 {
		t.Fatalf("expected at least 5 args, got %d: %v", len(args), args)
	}
	// Check core protocol flags
	if args[0] != "-p" {
		t.Errorf("arg[0] = %q, want -p", args[0])
	}
	if !containsFlagPair(args, "--output-format", "stream-json") {
		t.Error("missing --output-format stream-json")
	}
	if !containsFlagPair(args, "--input-format", "stream-json") {
		t.Error("missing --input-format stream-json")
	}
	if !containsFlag(args, "--verbose") {
		t.Error("missing --verbose")
	}
	if !containsFlagPair(args, "--permission-mode", "bypassPermissions") {
		t.Error("missing --permission-mode bypassPermissions")
	}
	// Prompt should be the last non-custom-arg argument
	foundPrompt := false
	for _, a := range args {
		if a == "hello" {
			foundPrompt = true
			break
		}
	}
	if !foundPrompt {
		t.Error("prompt not found in args")
	}
}

func TestBuildCodeBuddyArgsWithModel(t *testing.T) {
	t.Parallel()

	args := buildCodeBuddyArgs("test", ExecOptions{Model: "gpt-5.5"}, slog.Default())
	if !containsFlagPair(args, "--model", "gpt-5.5") {
		t.Error("missing --model gpt-5.5")
	}
}

func TestBuildCodeBuddyArgsWithMaxTurns(t *testing.T) {
	t.Parallel()

	args := buildCodeBuddyArgs("test", ExecOptions{MaxTurns: 10}, slog.Default())
	if !containsFlagPair(args, "--max-turns", "10") {
		t.Error("missing --max-turns 10")
	}
}

func TestBuildCodeBuddyArgsWithSystemPrompt(t *testing.T) {
	t.Parallel()

	args := buildCodeBuddyArgs("test", ExecOptions{SystemPrompt: "You are helpful"}, slog.Default())
	if !containsFlagPair(args, "--append-system-prompt", "You are helpful") {
		t.Error("missing --append-system-prompt")
	}
}

func TestBuildCodeBuddyArgsWithResume(t *testing.T) {
	t.Parallel()

	args := buildCodeBuddyArgs("test", ExecOptions{ResumeSessionID: "sess-abc"}, slog.Default())
	if !containsFlagPair(args, "--resume", "sess-abc") {
		t.Error("missing --resume sess-abc")
	}
}

func TestBuildCodeBuddyArgsWithCustomArgs(t *testing.T) {
	t.Parallel()

	args := buildCodeBuddyArgs("test", ExecOptions{
		CustomArgs: []string{"--add-dir", "/tmp/extra"},
	}, slog.Default())
	if !containsFlagPair(args, "--add-dir", "/tmp/extra") {
		t.Error("custom_args --add-dir /tmp/extra not found")
	}
}

func TestBuildCodeBuddyArgsFiltersBlockedArgs(t *testing.T) {
	t.Parallel()

	// User should not be able to override protocol-critical flags via custom_args.
	args := buildCodeBuddyArgs("test", ExecOptions{
		CustomArgs: []string{
			"--output-format", "json",
			"--permission-mode", "default",
		},
	}, slog.Default())

	// Blocked flags must not appear with user-supplied values.
	if containsFlagPair(args, "--output-format", "json") {
		t.Error("blocked --output-format json leaked through")
	}
	if containsFlagPair(args, "--permission-mode", "default") {
		t.Error("blocked --permission-mode default leaked through")
	}
	// The daemon-set values must still be present.
	if !containsFlagPair(args, "--output-format", "stream-json") {
		t.Error("missing daemon --output-format stream-json")
	}
	if !containsFlagPair(args, "--permission-mode", "bypassPermissions") {
		t.Error("missing daemon --permission-mode bypassPermissions")
	}
}

func TestBuildCodeBuddyArgsMaxTurnsNotAddedWhenZero(t *testing.T) {
	t.Parallel()

	args := buildCodeBuddyArgs("test", ExecOptions{MaxTurns: 0}, slog.Default())
	if containsFlag(args, "--max-turns") {
		t.Error("--max-turns should not be present when MaxTurns is 0")
	}
}

// ── Input builder tests ──

func TestBuildCodeBuddyInput(t *testing.T) {
	t.Parallel()

	data, err := buildCodeBuddyInput("say hello")
	if err != nil {
		t.Fatalf("buildCodeBuddyInput: %v", err)
	}
	if !strings.HasSuffix(string(data), "\n") {
		t.Error("input should end with newline")
	}

	// Parse back to verify structure.
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	var typ string
	if err := json.Unmarshal(payload["type"], &typ); err != nil {
		t.Fatalf("unmarshal type: %v", err)
	}
	if typ != "user" {
		t.Errorf("type = %q, want user", typ)
	}

	var msg map[string]json.RawMessage
	if err := json.Unmarshal(payload["message"], &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	var role string
	if err := json.Unmarshal(msg["role"], &role); err != nil {
		t.Fatalf("unmarshal role: %v", err)
	}
	if role != "user" {
		t.Errorf("role = %q, want user", role)
	}
}

// ── Model discovery tests ──

func TestParseCodeBuddyModelsFromHelp(t *testing.T) {
	t.Parallel()

	helpOutput := `Usage: codebuddy [options] [prompt]
  --model <model>  Model for the current session. Please provide the model ID. Currently supported: (claude-sonnet-4.6, claude-sonnet-4.6-1m, claude-opus-4.8, gpt-5.5, gpt-5.4, deepseek-v4-pro-ioa, deepseek-v4-flash-ioa, glm-5.1-ioa, kimi-k2.6-ioa, custom-local:deepseek-v4-pro)`

	models := parseCodeBuddyModelsFromHelp(helpOutput)
	if len(models) != 10 {
		t.Fatalf("got %d models, want 10", len(models))
	}

	// First model should be default.
	if !models[0].Default {
		t.Error("first model should be default")
	}

	// Check vendor prefixes are assigned correctly.
	wantProviders := map[string]string{
		"claude-sonnet-4.6":    "anthropic",
		"claude-sonnet-4.6-1m": "anthropic",
		"claude-opus-4.8":      "anthropic",
		"gpt-5.5":              "openai",
		"gpt-5.4":              "openai",
		"deepseek-v4-pro-ioa":  "deepseek",
		"deepseek-v4-flash-ioa": "deepseek",
		"glm-5.1-ioa":          "zhipu",
		"kimi-k2.6-ioa":        "moonshot",
		"custom-local:deepseek-v4-pro": "custom",
	}
	for _, m := range models {
		if want, ok := wantProviders[m.ID]; ok {
			if m.Provider != want {
				t.Errorf("model %q: provider = %q, want %q", m.ID, m.Provider, want)
			}
		}
	}

	// Check deduplication.
	dupOutput := "Currently supported: (gpt-5.5, gpt-5.5, gpt-5.4)"
	dupModels := parseCodeBuddyModelsFromHelp(dupOutput)
	if len(dupModels) != 2 {
		t.Errorf("deduplication: got %d models, want 2", len(dupModels))
	}
}

func TestParseCodeBuddyModelsFromHelpEmpty(t *testing.T) {
	t.Parallel()

	models := parseCodeBuddyModelsFromHelp("no models here")
	if len(models) != 0 {
		t.Errorf("expected 0 models from non-matching input, got %d", len(models))
	}
}

func TestParseCodeBuddyModelsFromHelpWhitespace(t *testing.T) {
	t.Parallel()

	// Extra spaces and newlines inside the parentheses.
	help := "Currently supported: (  claude-sonnet-4.6  ,  gpt-5.5  )"
	models := parseCodeBuddyModelsFromHelp(help)
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].ID != "claude-sonnet-4.6" {
		t.Errorf("model[0].ID = %q, want claude-sonnet-4.6", models[0].ID)
	}
	if models[1].ID != "gpt-5.5" {
		t.Errorf("model[1].ID = %q, want gpt-5.5", models[1].ID)
	}
}

func TestModelIDToLabel(t *testing.T) {
	t.Parallel()

	cases := []struct {
		id    string
		label string
	}{
		{"claude-sonnet-4.6", "Claude-sonnet-4.6"},
		{"gpt-5.5", "GPT-5.5"},
		{"gemini-3.1-pro", "Gemini-3.1-pro"},
		{"deepseek-v4-pro-ioa", "DeepSeek-v4-pro"},
		{"glm-5.1-ioa", "GLM-5.1"},
		{"kimi-k2.6-ioa", "Kimi-k2.6"},
		{"minimax-m3-ioa", "MiniMax-m3"},
	}
	for _, c := range cases {
		t.Run(c.id, func(t *testing.T) {
			got := modelIDToLabel(c.id)
			if got != c.label {
				t.Errorf("modelIDToLabel(%q) = %q, want %q", c.id, got, c.label)
			}
		})
	}
}

func TestCodeBuddyStaticModelsFallback(t *testing.T) {
	t.Parallel()

	models := codebuddyStaticModels()
	if len(models) == 0 {
		t.Fatal("static models should not be empty")
	}
	// First model should be default.
	if !models[0].Default {
		t.Error("first static model should be default")
	}
	// All models should have non-empty ID and Label.
	for i, m := range models {
		if m.ID == "" {
			t.Errorf("model[%d].ID is empty", i)
		}
		if m.Label == "" {
			t.Errorf("model[%d].Label is empty", i)
		}
	}
}

// ── Backend integration tests ──

func TestCodeBuddyExecuteStreamsResult(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	script := "#!/bin/sh\n" +
		"IFS= read -r _\n" +
		"printf '%s\\n' '{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"sess-1\"}'\n" +
		"printf '%s\\n' '{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"done\"}]}}'\n" +
		"printf '%s\\n' '{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"done\",\"session_id\":\"sess-1\",\"total_cost_usd\":0}'\n"
	writeTestExecutable(t, fakePath, []byte(script))

	backend, err := New("codebuddy", Config{ExecutablePath: fakePath, Logger: slog.Default()})
	if err != nil {
		t.Fatalf("new codebuddy backend: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	session, err := backend.Execute(ctx, "hello", ExecOptions{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}

	// Drain messages.
	go func() {
		for range session.Messages {
		}
	}()

	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without value")
		}
		if result.Status != "completed" {
			t.Errorf("status = %q, want completed (error=%q)", result.Status, result.Error)
		}
		if result.Output != "done" {
			t.Errorf("output = %q, want done", result.Output)
		}
		if result.SessionID != "sess-1" {
			t.Errorf("session_id = %q, want sess-1", result.SessionID)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}
}

func TestCodeBuddyExecuteRecordsModelUsage(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	script := "#!/bin/sh\n" +
		"IFS= read -r _\n" +
		"printf '%s\\n' '{\"type\":\"system\",\"session_id\":\"sess-usage\"}'\n" +
		"printf '%s\\n' '{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"model\":\"deepseek-v4-pro-ioa\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}],\"usage\":{\"input_tokens\":123,\"output_tokens\":45,\"cache_read_input_tokens\":10,\"cache_creation_input_tokens\":5}}}'\n" +
		"printf '%s\\n' '{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"hi\",\"session_id\":\"sess-usage\"}'\n"
	writeTestExecutable(t, fakePath, []byte(script))

	backend, err := New("codebuddy", Config{ExecutablePath: fakePath, Logger: slog.Default()})
	if err != nil {
		t.Fatalf("new codebuddy backend: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	session, err := backend.Execute(ctx, "hello", ExecOptions{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	go func() {
		for range session.Messages {
		}
	}()

	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without value")
		}
		if result.Status != "completed" {
			t.Fatalf("unexpected status: %s", result.Status)
		}
		u, ok := result.Usage["deepseek-v4-pro-ioa"]
		if !ok {
			t.Fatalf("usage missing for deepseek-v4-pro-ioa, got keys: %v", modelKeys(result.Usage))
		}
		if u.InputTokens != 123 {
			t.Errorf("input_tokens = %d, want 123", u.InputTokens)
		}
		if u.OutputTokens != 45 {
			t.Errorf("output_tokens = %d, want 45", u.OutputTokens)
		}
		if u.CacheReadTokens != 10 {
			t.Errorf("cache_read = %d, want 10", u.CacheReadTokens)
		}
		if u.CacheWriteTokens != 5 {
			t.Errorf("cache_write = %d, want 5", u.CacheWriteTokens)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}
}

func TestCodeBuddyExecuteSurfacesStderrWhenChildExitsEarly(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	script := "#!/bin/sh\n" +
		"IFS= read -r _\n" +
		"echo \"FATAL: CLI initialization failed\" >&2\n" +
		"exit 3\n"
	writeTestExecutable(t, fakePath, []byte(script))

	backend, err := New("codebuddy", Config{ExecutablePath: fakePath, Logger: slog.Default()})
	if err != nil {
		t.Fatalf("new codebuddy backend: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	session, err := backend.Execute(ctx, "hello", ExecOptions{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	go func() {
		for range session.Messages {
		}
	}()

	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without value")
		}
		if result.Status != "failed" {
			t.Errorf("status = %q, want failed (error=%q)", result.Status, result.Error)
		}
		if !strings.Contains(result.Error, "codebuddy exited with error") {
			t.Errorf("error should mention exit, got %q", result.Error)
		}
		if !strings.Contains(result.Error, "FATAL") {
			t.Errorf("error should include stderr hint, got %q", result.Error)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}
}

func TestCodeBuddyExecuteEmitsInitialStatus(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	// Emit a system event after a delay to simulate cold startup, then
	// result immediately. The test verifies the initial MessageStatus
	// arrives before the system event.
	script := "#!/bin/sh\n" +
		"IFS= read -r _\n" +
		"printf '%s\\n' '{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"sess-status\"}'\n" +
		"printf '%s\\n' '{\"type\":\"result\",\"subtype\":\"success\",\"is_error\":false,\"result\":\"ok\",\"session_id\":\"sess-status\"}'\n"
	writeTestExecutable(t, fakePath, []byte(script))

	backend, err := New("codebuddy", Config{ExecutablePath: fakePath, Logger: slog.Default()})
	if err != nil {
		t.Fatalf("new codebuddy backend: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	session, err := backend.Execute(ctx, "hello", ExecOptions{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}

	// Collect messages.
	var msgs []Message
	done := make(chan struct{})
	go func() {
		for m := range session.Messages {
			msgs = append(msgs, m)
		}
		close(done)
	}()

	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without value")
		}
		if result.Status != "completed" {
			t.Errorf("status = %q, want completed", result.Status)
		}
		<-done // wait for message drain
		// First message should be a status.
		if len(msgs) == 0 {
			t.Fatal("no messages received")
		}
		if msgs[0].Type != MessageStatus {
			t.Errorf("first message type = %q, want status", msgs[0].Type)
		}
		if msgs[0].Status != "running" {
			t.Errorf("first message status = %q, want running", msgs[0].Status)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}
}

func TestCodeBuddyExecuteTimeout(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fixture is POSIX-only")
	}

	fakePath := filepath.Join(t.TempDir(), "codebuddy")
	// Script reads stdin then sleeps forever — will be killed by timeout.
	script := "#!/bin/sh\n" +
		"IFS= read -r _\n" +
		"sleep 30\n"
	writeTestExecutable(t, fakePath, []byte(script))

	backend, err := New("codebuddy", Config{ExecutablePath: fakePath, Logger: slog.Default()})
	if err != nil {
		t.Fatalf("new codebuddy backend: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	session, err := backend.Execute(ctx, "hello", ExecOptions{Timeout: 1 * time.Second})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	go func() {
		for range session.Messages {
		}
	}()

	select {
	case result, ok := <-session.Result:
		if !ok {
			t.Fatal("result channel closed without value")
		}
		if result.Status != "timeout" {
			t.Errorf("status = %q, want timeout (error=%q)", result.Status, result.Error)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for result")
	}
}

func TestCodeBuddyExecuteBinaryNotFound(t *testing.T) {
	t.Parallel()

	backend := &codebuddyBackend{cfg: Config{ExecutablePath: "/nonexistent/codebuddy", Logger: slog.Default()}}
	_, err := backend.Execute(context.Background(), "test", ExecOptions{})
	if err == nil {
		t.Fatal("expected error for nonexistent binary")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("error = %q, want 'not found'", err)
	}
}

// ── Helpers ──

func containsFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func containsFlagPair(args []string, flag, value string) bool {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

func modelKeys(m map[string]TokenUsage) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// Ensure the file doesn't end up in a production build.
var _ = os.Stdout
