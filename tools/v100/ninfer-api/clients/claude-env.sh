# Claude Code against the local NInfer server. Source it, or put the same pairs in the "env" object
# of ~/.claude/settings.json (CLAUDE_CONFIG_DIR=... to keep it separate).
export ANTHROPIC_BASE_URL=http://192.168.1.211:8080
export ANTHROPIC_AUTH_TOKEN=local        # must match --api-key if the server sets one
export ANTHROPIC_MODEL=qwen3.8-27b-nvfp4
export ANTHROPIC_SMALL_FAST_MODEL=qwen3.8-27b-nvfp4
# Claude Code does not know this alias and would assume a 200k window; this is the real ceiling.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=200000
