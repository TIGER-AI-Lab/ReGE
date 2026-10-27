"""Source pools and the coach call.

A task's source pool is the set of recorded attempts under <TB_RUNS>/<task>/: one JSON record per attempt with
its steps, its final files (path -> content sha; contents stored under artifacts/<sha>) and its grade. load_pool()
reads the attempts, serialize_pool() renders them for the coach, and agg_call() sends one coach or judge request.
"""
from __future__ import annotations
import json, glob, threading, time, os
from pathlib import Path
import httpx
from openai import OpenAI
import tb_collect as TC
import door_probe as DP

OUT = TC.OUT
# Coach and judge model; the pipelines set both to the base model of the run.
AGG_URL = os.environ.get("AGG_URL", os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1"))
AGG_MODEL = os.environ.get("AGG_MODEL", "claude-sonnet-5")

# Captured files that never count as an attempt's output (caches, test-harness files).
NOISE_SUBSTR = ("__pycache__", ".tbench-testing", ".pytest_cache", ".git/", "/.local/", "node_modules")
NOISE_SUFFIX = (".pyc", ".pyo")
NOISE_EXACT = ("run-tests.sh", "test_output.json", ".auto_gen_key.dat")


def is_noise(path: str) -> bool:
    return (any(s in path for s in NOISE_SUBSTR) or path.endswith(NOISE_SUFFIX) or path in NOISE_EXACT)


def clean_deliv(deliv: dict) -> dict:
    return {p: s for p, s in deliv.items() if not is_noise(p)}


def load_pool(task, store, mcap=None):
    """Recorded attempts of the task. Each = {id, model, idx, deliv, d}. An attempt that left no output file
    but has a real trajectory stays in the pool with an empty deliv (POOL_KEEP_EMPTY=0 drops it).
    mcap (or env POOL_MCAP, default 4): keep the first m attempts per model in idx order."""
    m = mcap if mcap is not None else int(os.environ.get("POOL_MCAP", "4"))
    rolls = []
    for f in sorted(glob.glob(str(OUT / task / "*.json"))):
        b = Path(f).name
        d = json.load(open(f))
        if "final_files" not in d or "baseline" not in d:
            continue
        deliv = clean_deliv({p: s for p, s in d["final_files"].items() if d["baseline"].get(p) != s})
        if not deliv:
            if os.environ.get("POOL_KEEP_EMPTY", "1") in ("1", "true", "yes") and len(d.get("steps") or []) > 1:
                rolls.append({"id": b[:-5], "model": d["model"], "idx": d.get("idx"), "deliv": {}, "d": d, "empty": True})
            continue
        rolls.append({"id": b[:-5], "model": d["model"], "idx": d.get("idx"), "deliv": deliv, "d": d})
    if m is not None:
        bym = {}
        for r in sorted(rolls, key=lambda r: (r["model"], r["idx"] if r["idx"] is not None else 1 << 30)):
            bym.setdefault(r["model"], []).append(r)
        keep = {id(r) for v in bym.values() for r in v[:m]}
        rolls = [r for r in rolls if id(r) in keep]
    return rolls


def deliv_text(deliv, store, cap=6000):
    out = {}
    for p, sha in sorted(deliv.items()):
        fp = store / sha
        try:
            out[p] = fp.read_text(errors="replace")[:cap] if fp.exists() else ""
        except Exception:
            out[p] = ""
    return out


def serialize_files(deliv, store, cap=6000, redact_deliv=False):
    if redact_deliv:                                     # file contents withheld, paths kept
        if not deliv:
            return "(no files)"
        return ("[FINAL FILES redacted to avoid answer leakage — contents withheld] produced paths: "
                + ", ".join(sorted(deliv.keys())))
    t = deliv_text(deliv, store, cap)
    return "\n".join(f"--- FILE: {p} ---\n{c}" for p, c in t.items()) or "(no files)"


# At most this many rendered steps per attempt; the character budgets bound the total size.
PSYN_STEP_CAP = int(os.environ.get("PSYN_STEP_CAP", "120"))


def serialize_pool(rolls, store, cap=6000, step_cap=None, redact_deliv=False):
    """Numbered candidate blocks: per attempt the trajectory first (per step: reasoning <= 200 and command <= 160
    characters, no observations), then the final files. redact_deliv=True hides final-file contents and keeps
    their paths."""
    if step_cap is None:
        step_cap = PSYN_STEP_CAP
    blocks = []
    for i, r in enumerate(rolls, 1):
        head = f"===== CANDIDATE {i} (model={r['model']}) ====="
        sl = []
        for s in r["d"].get("steps", []):
            think = (s.get("think") or "").strip().replace("\n", " ")
            cmd = (s.get("cmd") or "").strip().replace("\n", " ")
            if not think and not cmd:
                continue
            sl.append(f"  · {think[:200]}" + (f"  $ {cmd[:160]}" if cmd else ""))
        body = ("TRAJECTORY:\n" + "\n".join(sl[:step_cap]) + "\n\nFINAL FILES:\n"
                + serialize_files(r["deliv"], store, cap, redact_deliv))
        blocks.append(head + "\n" + body)
    return "\n\n".join(blocks)


# One client per base URL, shared by all threads of the process.
_CLIENTS: dict = {}
_CLIENTS_LOCK = threading.Lock()


def _client_for(base_url):
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(base_url)
        if c is None:
            c = _CLIENTS[base_url] = OpenAI(base_url=base_url, api_key=TC.SV.key_for(base_url),
                                            timeout=240, max_retries=0)
        return c


# Wall-clock cap per attempt. The client timeout only fires after a period of silence, so each attempt runs on a
# side thread and is abandoned once the cap passes; the cap grows with the output budget, and AGG_CALL_CAP=<sec>
# sets a flat one.
def _call_cap(max_tokens):
    env = os.environ.get("AGG_CALL_CAP")
    return int(env) if env else min(900, max(120, 60 + int(max_tokens) // 40))


# Token accounting: every response agg_call receives appends its API usage to a thread-local list. A builder calls
# usage_reset() before its coach call(s) and usage_take() after; nothing sent or returned changes.
_USAGE = threading.local()


def usage_reset():
    _USAGE.acc = []


def usage_take():
    acc = getattr(_USAGE, "acc", None) or []
    _USAGE.acc = []
    return acc


def _usage_note(resp, attempts, secs, model=None):
    try:
        u = getattr(resp, "usage", None)
        entry = {"in": int(getattr(u, "prompt_tokens", None) or 0) if u is not None else None,
                 "out": int(getattr(u, "completion_tokens", None) or 0) if u is not None else None}
        # Claude endpoints report prompt_tokens without cached input; cache reads and writes are recorded when the
        # response carries them, so the full input can be reconstructed.
        for _k, _n in (("cache_read_input_tokens", "cache_read"), ("cache_creation_input_tokens", "cache_write")):
            _v = (getattr(u, "model_extra", None) or {}).get(_k) if u is not None else None
            if isinstance(_v, (int, float)):
                entry[_n] = int(_v)
    except Exception:
        entry = {"in": None, "out": None}
    entry.update({"attempts": attempts, "secs": round(secs, 1), "model": model})
    if getattr(_USAGE, "acc", None) is None:
        _USAGE.acc = []
    _USAGE.acc.append(entry)


def agg_call(sys_prompt, user, model=AGG_MODEL, base_url=AGG_URL, max_tokens=8000, temperature=None, role=None):
    """One coach or judge request: up to six attempts; returns the reply text, or "" if every attempt failed."""
    client = _client_for(base_url)
    kw = {"temperature": temperature} if temperature is not None else {}
    kw.update(DP.thinking_off_kw(model))
    budget = max_tokens
    cap = _call_cap(budget)
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
    last = ""
    for attempt in range(6):
        t0, box = time.time(), {}

        def _run(box=box, budget=budget):
            try:
                box["r"] = client.chat.completions.create(
                    model=model, max_tokens=budget, messages=msgs,
                    timeout=httpx.Timeout(cap, connect=20), **kw)
            except BaseException as e:            # noqa: BLE001 -- surfaced to the caller below
                box["e"] = e

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        th.join(cap + 5)
        if "r" in box:
            ch = box["r"].choices[0]
            txt = ch.message.content or ""
            _usage_note(box["r"], attempt + 1, time.time() - t0, model)
            return txt
        if th.is_alive():
            last = f"call-cap {cap}s exceeded (call abandoned to side thread)"
        else:
            e = box.get("e")
            last = f"{type(e).__name__}: {str(e)[:140]}"
        print(f"    [agg_call retry {attempt + 1}/6] {model} {time.time() - t0:.0f}s {last}", flush=True)
        time.sleep(2 * (attempt + 1))
    print(f"    [agg_call FAILED] {model} {last}", flush=True)
    return ""
