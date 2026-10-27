"""Shared helpers: source-pool and task-cache paths, the bash tool, and the container helpers that the
Terminal-Bench 2.0 collector (tb2_collect.py) uses to run commands in a task container and snapshot /app.

TB_RUNS selects the source-pool directory (relative paths resolve under data/), TASKCACHE the task-cache
directory with one task.yaml per task.
"""
from __future__ import annotations
import os, subprocess
from pathlib import Path
import stitch_validate as SV

TASKCACHE = Path(os.environ.get("TASKCACHE",
    Path(__file__).resolve().parent.parent / "data" / "taskcache_tb"))
# A relative TB_RUNS is resolved against the repository's data/ directory, independent of the working directory.
_HERE = Path(__file__).resolve().parent
_TB = os.environ.get("TB_RUNS")
OUT = Path(_TB) if (_TB and Path(_TB).is_absolute()) else ((_HERE.parent / "data") / (_TB or "runs_tb"))
BASH_TOOL = SV.BASH_TOOL
MODELS = SV.MODELS


class FatalInfraError(RuntimeError):
    """Quota exhausted or authentication failed: retrying cannot help, and recording the rollout would book an
    infrastructure failure as a task failure, so the batch aborts instead."""


# Error messages that mean the credential or quota is unusable (not a transient 429/502 that a retry can fix).
_FATAL_MARKERS = ("quota_exhausted", "insufficient_quota", "exceeded your current quota",
                  "valid api key", "invalid_argument", "api key not valid", "invalid_api_key",
                  "permission denied", "permission_denied", "error code: 401", "error code: 403",
                  "error code: 402", "insufficient balance", "account_deactivated")


def fatal_infra(s: str) -> bool:
    s = (s or "").lower()
    return any(m in s for m in _FATAL_MARKERS)


# Asks the agent to state its reasoning before each command, so the trajectory records it.
REASON_SYS = ("You are a careful shell agent operating in /app. Complete the task exactly. "
              "The grader only sees the final filesystem state under /app, so every final answer, code change, "
              "script, generated data file, or report must be written to the exact /app path requested by the "
              "task. Do not stop after printing an answer; write it to the required file. Temporary work under "
              "/tmp, /root, or another directory is allowed only if the final deliverable is copied back into "
              "/app. Create a first working deliverable early, then test and refine it. Before your final "
              "response, verify that the expected /app file(s) exist and contain the intended content. "
              "Think step by step: BEFORE every bash command, first write 1-3 sentences of REASONING "
              "(as plain message text) — what you will do next and WHY, including what you concluded from "
              "the previous command's output — THEN call the bash tool.")


def tcs_echo(tcs):
    """Echo tool calls back to the API, keeping any provider-specific extra_content."""
    out = []
    for t in tcs:
        d = {"id": t.id, "type": "function",
             "function": {"name": t.function.name, "arguments": t.function.arguments}}
        ec = t.model_dump().get("extra_content")
        if ec:
            d["extra_content"] = ec
        out.append(d)
    return out


def sh(*args, inp=None, timeout=1800):
    # Commands can print binary output; decode leniently so a rollout is never lost to a decoding error.
    return subprocess.run(args, input=inp, capture_output=True, text=True, errors="replace", timeout=timeout)


def exec_in(name: str, cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(["docker", "exec", "-i", name, "bash"], input=cmd,
                           capture_output=True, text=True, errors="replace", timeout=timeout)
        return (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"


def capture_state(name: str) -> dict:
    """path -> 16-hex content sha for regular files under /app (skip hidden, >=10MB).
    `-size -Nc` counts bytes; `-size -1M` would round up to 1M blocks and match only empty files."""
    out = exec_in(name, "cd /app && find . -type f -not -path './.*' -size -10485760c -exec sha1sum {} + 2>/dev/null")
    m = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) >= 16:
            sha, path = parts[0], parts[1]
            if path.startswith("./"):
                path = path[2:]
            m[path] = sha[:16]
    return m


def save_artifacts(name: str, hashmap: dict, baseline: dict, store: Path):
    """copy content of agent-created/changed files (keyed by sha), once each."""
    for path, sha in hashmap.items():
        if baseline.get(path) == sha:
            continue  # unchanged input file
        dest = store / sha
        if dest.exists():
            continue
        sh("docker", "cp", f"{name}:/app/{path}", str(dest), timeout=60)
