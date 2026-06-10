// The set of runtime providers whose backend reads `agent.mcp_config` and
// forwards MCP servers to the underlying CLI. The MCP config tab is hidden
// for every other provider so a user can't save a value the runtime will
// silently ignore. Keep this list in sync with the backends in
// `server/pkg/agent/` that read `ExecOptions.McpConfig`, plus the OpenClaw
// per-task wrapper preparer in `server/internal/daemon/execenv/` which
// materialises `mcp.servers` into the synthesised config rather than going
// through ExecOptions.
//
// NOTE: this set covers built-in providers only. External runtime
// extensions control MCP visibility through their manifest's
// `capabilities.mcp_config` flag — see `isTabVisibleForRuntime` in
// tab-visibility.ts. Adding a new built-in provider here is the only
// hard-coded path; everything else is data-driven.
const MCP_SUPPORTED_PROVIDERS = new Set([
  "claude",
  "codex",
  "hermes",
  "kimi",
  "kiro",
  "opencode",
  "openclaw",
]);

export function providerSupportsMcpConfig(provider: string | undefined | null): boolean {
  if (!provider) return false;
  return MCP_SUPPORTED_PROVIDERS.has(provider);
}
