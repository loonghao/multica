package daemon

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadRuntimeManifests(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()

	// Create a valid manifest.
	subDir := filepath.Join(dir, "codebuddy")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	valid := RuntimeManifest{
		ID:       "codebuddy",
		Name:     "CodeBuddy Code",
		Version:  "1.0.0",
		Provider: "codebuddy",
		Transport: "acp-stdio",
		Command: RuntimeManifestCommand{
			Executable: "codebuddy",
			Args:       []string{"--acp"},
		},
	}
	data, _ := json.MarshalIndent(valid, "", "  ")
	if err := os.WriteFile(filepath.Join(subDir, "runtime.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	// Create an invalid manifest (missing required fields).
	invalidDir := filepath.Join(dir, "broken")
	if err := os.MkdirAll(invalidDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(invalidDir, "runtime.json"), []byte(`{"id": "broken"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	// Create a directory without runtime.json — should be skipped silently.
	emptyDir := filepath.Join(dir, "empty")
	if err := os.MkdirAll(emptyDir, 0o755); err != nil {
		t.Fatal(err)
	}

	manifests, err := LoadRuntimeManifests(dir)
	if err != nil {
		t.Fatalf("LoadRuntimeManifests: %v", err)
	}
	if len(manifests) != 1 {
		t.Fatalf("expected 1 manifest, got %d", len(manifests))
	}
	m := manifests[0]
	if m.ID != "codebuddy" {
		t.Errorf("id = %q, want codebuddy", m.ID)
	}
	if m.Provider != "codebuddy" {
		t.Errorf("provider = %q, want codebuddy", m.Provider)
	}
	if m.Transport != "acp-stdio" {
		t.Errorf("transport = %q, want acp-stdio", m.Transport)
	}
	if m.Command.Executable != "codebuddy" {
		t.Errorf("command.executable = %q, want codebuddy", m.Command.Executable)
	}
	if len(m.Command.Args) != 1 || m.Command.Args[0] != "--acp" {
		t.Errorf("command.args = %v, want [--acp]", m.Command.Args)
	}
}

func TestLoadRuntimeManifestsMissingDir(t *testing.T) {
	t.Parallel()

	manifests, err := LoadRuntimeManifests("/nonexistent/path/12345")
	if err != nil {
		t.Fatalf("LoadRuntimeManifests should not error on missing dir: %v", err)
	}
	if len(manifests) != 0 {
		t.Errorf("expected 0 manifests from missing dir, got %d", len(manifests))
	}
}

func TestLoadRuntimeManifestsEmptyDir(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	manifests, err := LoadRuntimeManifests(dir)
	if err != nil {
		t.Fatalf("LoadRuntimeManifests: %v", err)
	}
	if len(manifests) != 0 {
		t.Errorf("expected 0 manifests from empty dir, got %d", len(manifests))
	}
}

func TestDefaultRuntimesDir(t *testing.T) {
	t.Parallel()
	dir := DefaultRuntimesDir()
	if dir == "" {
		t.Fatal("DefaultRuntimesDir returned empty string")
	}
	// Should contain "runtimes" somewhere.
	if !stringsContains(dir, "runtimes") {
		t.Errorf("DefaultRuntimesDir = %q, want to contain 'runtimes'", dir)
	}
}

func stringsContains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
