package agent

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

// TestBuildStreamJSONExternalArgsBaseFlagsKept verifies the manifest's
// command.args (passed via cfg.ACPArgs) appear first in the resulting
// argv. Stream-json runtimes typically declare the protocol flags
// (`-p --output-format stream-json --input-format stream-json --verbose`)
// in their manifest; the daemon must preserve that prefix so the CLI
// enters non-interactive mode.
func TestBuildStreamJSONExternalArgsBaseFlagsKept(t *testing.T) {
	t.Parallel()
	cfg := Config{
		ACPArgs: []string{"-p", "--output-format", "stream-json"},
		Logger:  slog.Default(),
	}
	args := buildStreamJSONExternalArgs(cfg, ExecOptions{}, nil)
	if len(args) < 3 || args[0] != "-p" || args[1] != "--output-format" || args[2] != "stream-json" {
		t.Errorf("base flags not preserved at head: %v", args)
	}
}

// TestBuildStreamJSONExternalArgsCapabilityGated documents the same
// gating contract the ACP backend uses: model/effort/max-turns/resume
// only appear when the matching capability is on.
func TestBuildStreamJSONExternalArgsCapabilityGated(t *testing.T) {
	t.Parallel()
	cfg := Config{ACPArgs: []string{"-p"}, Logger: slog.Default()}
	t.Run("disabled", func(t *testing.T) {
		args := buildStreamJSONExternalArgs(cfg, ExecOptions{
			Model:           "claude-sonnet-4",
			ThinkingLevel:   "high",
			MaxTurns:        25,
			ResumeSessionID: "sess-1",
		}, nil)
		joined := strings.Join(args, " ")
		for _, want := range []string{"--model", "--effort", "--max-turns", "--resume"} {
			if strings.Contains(joined, want) {
				t.Errorf("flag %s leaked through with caps disabled: %s", want, joined)
			}
		}
	})
	t.Run("enabled", func(t *testing.T) {
		cfg.Capabilities = ConfigCapabilities{
			Thinking:       true,
			SessionResume:  true,
			MaxTurns:       true,
			ModelSelection: true,
		}
		args := buildStreamJSONExternalArgs(cfg, ExecOptions{
			Model:           "claude-sonnet-4",
			ThinkingLevel:   "high",
			MaxTurns:        25,
			ResumeSessionID: "sess-1",
		}, nil)
		joined := strings.Join(args, " ")
		for _, want := range []string{"--model claude-sonnet-4", "--effort high", "--max-turns 25", "--resume sess-1"} {
			if !strings.Contains(joined, want) {
				t.Errorf("expected %q in argv, got %s", want, joined)
			}
		}
	})
}

// TestBuildStreamJSONExternalArgsCustomArgsAppended verifies user-supplied
// extra/custom args land at the tail (after the daemon-managed flags) and
// are run through the blocked-args filter so a manifest's protected flags
// can't be overridden by per-agent custom_args.
func TestBuildStreamJSONExternalArgsCustomArgsAppended(t *testing.T) {
	t.Parallel()
	cfg := Config{ACPArgs: []string{"-p"}, Logger: slog.Default()}
	blocked := manifestBlockedArgs(map[string]string{"--output-format": "value"})
	args := buildStreamJSONExternalArgs(cfg, ExecOptions{
		ExtraArgs:  []string{"--extra"},
		CustomArgs: []string{"--output-format", "json", "--debug"},
	}, blocked)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--extra") {
		t.Errorf("extra args missing: %s", joined)
	}
	if strings.Contains(joined, "--output-format json") {
		t.Errorf("blocked flag leaked through: %s", joined)
	}
	if !strings.Contains(joined, "--debug") {
		t.Errorf("safe custom arg dropped: %s", joined)
	}
}

// TestWriteStreamJSONUserFrameProducesNDJSON pins the wire frame the
// daemon writes to stdin. The format must match Claude/CodeBuddy
// stream-json or the receiving CLI will reject the first message and
// exit immediately.
func TestWriteStreamJSONUserFrameProducesNDJSON(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	if err := writeStreamJSONUserFrame(&buf, "hello world"); err != nil {
		t.Fatalf("writeStreamJSONUserFrame: %v", err)
	}
	out := buf.Bytes()
	if !bytes.HasSuffix(out, []byte("\n")) {
		t.Errorf("frame must end in newline, got %q", out)
	}
	var frame map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(out), &frame); err != nil {
		t.Fatalf("frame is not valid JSON: %v", err)
	}
	if frame["type"] != "user" {
		t.Errorf("frame.type = %v, want user", frame["type"])
	}
	msg, _ := frame["message"].(map[string]any)
	if msg == nil {
		t.Fatalf("frame.message missing")
	}
	if msg["role"] != "user" {
		t.Errorf("message.role = %v, want user", msg["role"])
	}
	content, _ := msg["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("message.content len = %d, want 1", len(content))
	}
	block, _ := content[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "hello world" {
		t.Errorf("content[0] = %v, want {type:text, text:hello world}", block)
	}
}

// TestStreamJSONExternalRouting_AgentNew sanity checks that the factory
// hands back a streamJSONExternalBackend when transport == "stream-json".
// This is the gate that lets an external runtime use the same parser
// the codebuddy and claude built-ins do, without growing a new switch
// case in `agent.New`.
func TestStreamJSONExternalRouting_AgentNew(t *testing.T) {
	t.Parallel()
	b, err := New("custom-runtime", Config{
		ExecutablePath: "echo",
		Transport:      "stream-json",
		IsExternal:     true,
		Logger:         slog.Default(),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, ok := b.(*streamJSONExternalBackend); !ok {
		t.Errorf("backend = %T, want *streamJSONExternalBackend", b)
	}
}

// TestACPExternalRouting_AgentNew is the matching gate for transport ==
// "acp-stdio". External runtimes that don't pick a transport default to
// ACP for backwards-compat with schema v1.
func TestACPExternalRouting_AgentNew(t *testing.T) {
	t.Parallel()
	t.Run("explicit acp-stdio", func(t *testing.T) {
		b, err := New("custom-acp-runtime", Config{
			ExecutablePath: "echo",
			Transport:      "acp-stdio",
			IsExternal:     true,
			Logger:         slog.Default(),
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, ok := b.(*acpExternalBackend); !ok {
			t.Errorf("backend = %T, want *acpExternalBackend", b)
		}
	})
	t.Run("backcompat default", func(t *testing.T) {
		b, err := New("legacy", Config{
			ExecutablePath: "echo",
			IsExternal:     true,
			Logger:         slog.Default(),
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if _, ok := b.(*acpExternalBackend); !ok {
			t.Errorf("backend = %T, want *acpExternalBackend", b)
		}
	})
}
