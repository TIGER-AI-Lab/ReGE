"""Model registry and API keys.

Keys are read from the environment or from a .env file at the repository root (NAME=value lines).
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "gpt56luna": (os.environ.get("GPT_BASE_URL", "https://api.openai.com/v1"), "gpt-5.6-luna"),
    # Both Claude keys serve the same model; CLAUDE_SONNET5_BASE and CLAUDE_SONNET5B_BASE select the endpoint.
    "claude-sonnet5": (os.environ.get("CLAUDE_SONNET5_BASE", os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1")), "claude-sonnet-5"),
    "claude-sonnet5b": (os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1"), "claude-sonnet-5"),
    "dsv4flash": ("https://api.deepseek.com/v1", "deepseek-v4-flash"),
}


def _env_key(name):
    try:
        for line in open(ROOT / ".env"):
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return os.environ.get(name, "")


def key_for(base_url: str) -> str:
    """Provider API key for an endpoint (environment or ROOT/.env); other endpoints take LOCAL_API_KEY."""
    if "deepseek" in base_url:
        return _env_key("DEEPSEEK_API_KEY")
    if "anthropic.com" in base_url:
        return _env_key("ANTHROPIC_API_KEY")
    if "openai.com" in base_url:
        return _env_key("OPENAI_API_KEY")
    return os.environ.get("LOCAL_API_KEY", "oauth")


BASH_TOOL = [{"type": "function", "function": {
    "name": "bash", "description": "Run a bash command in the working directory.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
