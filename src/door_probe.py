"""Request settings for Claude: every Claude request disables extended thinking."""
from __future__ import annotations


def thinking_off_kw(model: str) -> dict:
    """Request kwargs that turn extended thinking off for claude models."""
    return {"extra_body": {"thinking": {"type": "disabled"}}} if str(model).startswith("claude") else {}
