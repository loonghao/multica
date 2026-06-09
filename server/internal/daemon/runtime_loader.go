package daemon

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// RuntimeManifest describes a user-installed agent runtime extension.
// It mirrors the AionUi extension manifest pattern: a JSON file that
// declares an ACP-compatible CLI and its capabilities, discovered by
// scanning ~/.multica/runtimes/*/runtime.json at daemon startup.
//
// Schema version: v1
type RuntimeManifest struct {
	ID            string                    `json:"id"`
	Name          string                    `json:"name"`
	Version       string                    `json:"version"`
	Description   string                    `json:"description,omitempty"`
	Provider      string                    `json:"provider"`
	Transport     string                    `json:"transport"`     // "acp-stdio" or "stream-json"
	LaunchHeader  string                    `json:"launch_header,omitempty"`
	Command       RuntimeManifestCommand    `json:"command"`
	Capabilities  *RuntimeManifestCaps      `json:"capabilities,omitempty"`
	Models        []RuntimeManifestModel    `json:"models,omitempty"`
	ConfigFile    string                    `json:"config_file,omitempty"`    // "AGENTS.md", "CLAUDE.md", "" to skip
	SkillsRoot    string                    `json:"skills_root,omitempty"`    // relative to home or absolute
	IconURL       string                    `json:"icon_url,omitempty"`       // remote URL for the provider icon
	Pricing       map[string]RuntimePricing `json:"pricing,omitempty"`
	MinCLIVersion string                    `json:"min_cli_version,omitempty"` // e.g. "0.2.0"
	Env           map[string]string         `json:"env,omitempty"`
}

// RuntimeManifestCommand describes how to launch the ACP-compatible CLI.
type RuntimeManifestCommand struct {
	Executable         string              `json:"executable"`
	Args               []string            `json:"args,omitempty"`
	BlockedArgs        map[string]string   `json:"blocked_args,omitempty"`         // flag → "flag" or "value"
	ExtraArgsAllowlist []string            `json:"extra_args_allowlist,omitempty"` // daemon-level args allowed through
}

// RuntimeManifestCaps declares which optional features the runtime supports.
type RuntimeManifestCaps struct {
	Thinking           bool `json:"thinking,omitempty"`
	McpConfig          bool `json:"mcp_config,omitempty"`
	InlineSystemPrompt bool `json:"inline_system_prompt,omitempty"`
	SessionResume      bool `json:"session_resume,omitempty"`
	MaxTurns           bool `json:"max_turns,omitempty"`
	ModelSelection     bool `json:"model_selection,omitempty"`
	LocalSkills        bool `json:"local_skills,omitempty"`
}

// RuntimeManifestModel describes a model exposed by the runtime.
type RuntimeManifestModel struct {
	ID       string   `json:"id"`
	Label    string   `json:"label,omitempty"`
	Default  bool     `json:"default,omitempty"`
	Thinking []string `json:"thinking,omitempty"` // e.g. ["none","low","medium","high"]
}

// RuntimePricing is per-model pricing in USD per million tokens.
type RuntimePricing struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead,omitempty"`
	CacheWrite float64 `json:"cacheWrite,omitempty"`
}

// ToAgentEntry converts a runtime manifest into a daemon AgentEntry for
// registration. The transport field determines how the daemon spawns the
// CLI at task time.
func (m RuntimeManifest) ToAgentEntry() AgentEntry {
	transport := m.Transport
	if transport == "" {
		transport = "acp-stdio"
	}
	return AgentEntry{
		Path:         m.Command.Executable,
		Transport:    transport,
		ACPArgs:      m.Command.Args,
		IsExternal:   true,
		LaunchHeader: m.LaunchHeader,
		ConfigFile:   m.ConfigFile,
		SkillsRoot:   m.SkillsRoot,
		IconURL:      m.IconURL,
		Models:       manifestModelsToAgentModels(m.Models),
		Pricing:      m.Pricing,
		ManifestName: m.Name,
		rawCaps:      m.Capabilities,
	}
}

func manifestModelsToAgentModels(models []RuntimeManifestModel) []AgentModel {
	if len(models) == 0 {
		return nil
	}
	out := make([]AgentModel, len(models))
	for i, m := range models {
		out[i] = AgentModel{
			ID:       m.ID,
			Label:    m.Label,
			Default:  m.Default,
			Thinking: m.Thinking,
		}
	}
	return out
}

// LoadRuntimeManifests scans rootDir (typically ~/.multica/runtimes/)
// for subdirectories containing a runtime.json file and returns the
// parsed manifests. Invalid or unparseable files are skipped with a
// logged warning; the caller decides how to handle an empty result.
func LoadRuntimeManifests(rootDir string) ([]RuntimeManifest, error) {
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read runtime dir %s: %w", rootDir, err)
	}

	var manifests []RuntimeManifest
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		manifestPath := filepath.Join(rootDir, entry.Name(), "runtime.json")
		data, err := os.ReadFile(manifestPath)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			fmt.Fprintf(os.Stderr, "warning: failed to read runtime manifest %s: %v\n", manifestPath, err)
			continue
		}

		var m RuntimeManifest
		if err := json.Unmarshal(data, &m); err != nil {
			fmt.Fprintf(os.Stderr, "warning: invalid runtime manifest %s: %v\n", manifestPath, err)
			continue
		}

		// Validate required fields.
		var missing []string
		if m.ID == "" {
			missing = append(missing, "id")
		}
		if m.Name == "" {
			missing = append(missing, "name")
		}
		if m.Provider == "" {
			missing = append(missing, "provider")
		}
		if m.Transport == "" {
			missing = append(missing, "transport")
		}
		if m.Command.Executable == "" {
			missing = append(missing, "command.executable")
		}
		if len(missing) > 0 {
			fmt.Fprintf(os.Stderr, "warning: runtime manifest %s missing required fields: %s\n", manifestPath, strings.Join(missing, ", "))
			continue
		}

		manifests = append(manifests, m)
	}

	return manifests, nil
}

// DefaultRuntimesDir returns the default path for user-installed runtime
// extensions (~/.multica/runtimes). The directory is NOT created here;
// callers should create it if needed before scanning.
func DefaultRuntimesDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".multica", "runtimes")
	}
	return filepath.Join(home, ".multica", "runtimes")
}
