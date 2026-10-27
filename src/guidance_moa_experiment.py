"""Guidance construction: each builder turns a task's source pool into the text handed to a fresh student.

  psyn      ReGE-Syn: one coach call over the rendered pool, which returns an executable step-by-step plan
  best1     ReGE-Sel: the trajectory picked by the label-free judge (judge_sweep.py), rendered verbatim
  psyna     ReGE-A: the ReGE-Syn call with an explanation-and-answer request instead of the plan request
  psynnp    Closed-book: the ReGE-Syn call with no source attempts
  allraw    Direct history: every source trajectory rendered verbatim, without a coach
  sum4m     Independent summaries: one structured summary per source attempt, concatenated
  aggagent  AggAgent (aggagent_arm.py), registered when EXTRA_ARMS includes it

extra_arm_guides() builds the requested arms for one task. Correctness labels of the sources are never shown.
"""
from __future__ import annotations
import hashlib, json, os, re, sys, time
from pathlib import Path
import aggregate_deliverable as A


def _r_ok(r):
    d = r["d"]
    if isinstance(d.get("ok"), bool):
        return d["ok"]
    return bool((d.get("ok_grade") or {}).get("ok"))


# ---- single-trajectory rendering (ReGE-Sel judge and guide, Direct history) ----
# At most V250_STEPS steps per trajectory; the reasoning and command of each step are cut to think_ch / cmd_ch
# characters (ReGE-Sel: 200 / 160; Direct history: 250 / 600).
V250_STEPS = int(os.environ.get("V250_STEPS", "120"))


def _render_traj(r, think_ch=250, cmd_ch=600, max_steps=None):
    if max_steps is None:
        max_steps = V250_STEPS
    steps = r["d"].get("steps", [])
    out = [f"=== rollout {r['id']} (model {r['model']}, {len(steps)} steps) ==="]
    for i, s in enumerate(steps[:max_steps], 1):
        t = (s.get("think") or "").strip().replace("\n", " ")[:think_ch]
        if not t and s.get("cmd") == "[ui-step]":
            # AndroidWorld records keep the agent's per-step action summary in `obs`, so it stands in for the reasoning.
            t = (s.get("obs") or "").strip().replace("\n", " ")[:think_ch]
        c = (s.get("cmd") or "")[:cmd_ch]
        out.append(f"step {i}: {t}" + (f"\n  $ {c}" if c else ""))
    if len(steps) > max_steps:
        out.append(f"... ({len(steps) - max_steps} more steps truncated)")
    return "\n".join(out)


# ---- ReGE-Sel ----
BEST1_HDR = ("## A PREVIOUS AGENT'S COMPLETE ATTEMPT (judged most likely correct among several independent "
             "attempts; follow it but verify as you go):")
# AndroidWorld delivery: the GPT line uses AW_RESET_HDR in place of BEST1_HDR; the Claude line adds AW_RESET_NOTE
# after BEST1_HDR.
AW_RESET_HDR = ("## RESET-AND-VERIFICATION GUIDANCE\n"
                "The following material comes from a previous attempt at this task. "
                "Your environment has been reset to its initial state. "
                "Use current observations to determine which actions are needed, "
                "and verify the current task state before declaring completion.")
AW_RESET_NOTE = ("The following record comes from a separate, earlier execution.\n"
                 "The environment for your current execution has been reset to the task's initial state.\n"
                 "Statements about actions taken or completion in the record describe that earlier execution only; "
                 "they do not establish the state or completion of your current task.")

# JUDGE_PICK_FILE: json {task: rollout_id}, the picks of judge_sweep.py (merged by scripts/merge_parts.py).
JUDGE_PICK_FILE = os.environ.get("JUDGE_PICK_FILE")


def g_best1(task, instr, rolls, store, ctx):
    if not JUDGE_PICK_FILE:
        raise RuntimeError("best1 requires JUDGE_PICK_FILE (judge picks from judge_sweep.py)")
    rid = json.load(open(JUDGE_PICK_FILE))[task]             # a missing pick fails loudly, never falls back
    match = next(r for r in rolls if r["id"] == rid)
    body = f"\n(chosen: {match['id']}, label-free judge)\n\n" + _render_traj(match, 200, 160, V250_STEPS)
    if match["d"].get("collector") == "droid_guides.transcribe":
        if str(match["model"]).startswith("claude"):
            return BEST1_HDR + "\n\n" + AW_RESET_NOTE + body
        return AW_RESET_HDR + body
    return BEST1_HDR + body


# ---- ReGE-Syn, ReGE-A and Closed-book ----
# One coach call over the serialized pool (per step: reasoning <= 200 and command <= 160 characters, no
# observations; final files <= 6,000 characters each), a 10,000-token output request, delivered under NAPLAN_HDR.
NAPLAN_HDR = ("## GUIDANCE (synthesized by an aggregation agent that inspected several previous "
              "independent attempts at this task; follow it but verify as you go):")

PSYN_SYS = ("You are an aggregation agent: several independent agents attempted a task (correctness "
            "unknown) and you synthesize the single best guidance for a fresh agent. You do NOT have "
            "access to the ground truth solution.")
PSYN_ASK = ("Write an EXECUTABLE STEP-BY-STEP PLAN for a fresh agent that will solve the task from "
            "scratch: (1) numbered steps in execution order, each concrete enough to act on; (2) the "
            "critical facts the attempts uncovered -- exact paths, values, commands, formats, edge "
            "cases -- placed at the steps where they matter; (3) pitfalls the attempts exposed and "
            "how to avoid them. If the evidence already supports a final answer or artifact, state it "
            "in the plan together with how to verify it. The plan must be self-contained: the fresh "
            "agent sees only this plan, never the attempts.")
# ReGE-A: the explanation-and-answer request, the same contract as AggAgent's finish tool.
NA_ASK = ("Produce your solution as a single string with exactly two XML sections: "
          "<explanation>...</explanation><answer>...</answer>. The explanation must be "
          "self-contained -- a reader who never saw this prompt understands how the answer was "
          "derived. The answer is your best final solution to the task.")

# Overflow policy: if the serialization exceeds PSYN_SER_CAP characters, the file contents are removed (paths
# kept); if it is still too long, it is truncated with a marker.
PSYN_SER_CAP = int(os.environ.get("PSYN_SER_CAP", "350000"))     # chars ~= 87k tokens


def _ser_capped(rolls, store, redact):
    ser = A.serialize_pool(rolls, store, redact_deliv=redact)
    if len(ser) > PSYN_SER_CAP and not redact:
        ser = A.serialize_pool(rolls, store, redact_deliv=True)
    if len(ser) > PSYN_SER_CAP:
        ser = ser[:PSYN_SER_CAP] + "\n\n[POOL SERIALIZATION TRUNCATED AT BUDGET]"
    return ser


def _psyn_build(instr, rolls, store, redact):
    user = (f"TASK:\n{instr}\n\n{len(rolls)} independent previous attempts (full trajectories + final "
            f"files{', file contents redacted' if redact else ''}):\n\n"
            + _ser_capped(rolls, store, redact)
            + "\n\n" + PSYN_ASK)
    return NAPLAN_HDR + "\n\n" + A.agg_call(PSYN_SYS, user, max_tokens=10000)


def g_psyn(task, instr, rolls, store, ctx):
    return _psyn_build(instr, rolls, store, False)


def g_psynnp(task, instr, rolls, store, ctx):
    """Closed-book: the ReGE-Syn call (same system prompt, request, budget and header) with zero attempts."""
    return _psyn_build(instr, [], store, False)


def g_psyna(task, instr, rolls, store, ctx):
    user = (f"TASK:\n{instr}\n\n{len(rolls)} independent previous attempts (full trajectories + final "
            f"files):\n\n" + _ser_capped(rolls, store, False) + "\n\n" + NA_ASK)
    return NAPLAN_HDR + "\n\n" + A.agg_call(PSYN_SYS, user, max_tokens=10000)


# ---- Direct history ----
ALLRAW_HDR = ("## ALL PREVIOUS ATTEMPTS AT THIS TASK, VERBATIM (no selection, no summary; "
              "several independent agents, outcomes unknown):")


def g_allraw(task, instr, rolls, store, ctx):
    """Every source trajectory rendered verbatim and concatenated; no judge, no coach, no labels."""
    parts = [_render_traj(r, 250, 600, V250_STEPS) for r in rolls]
    return ALLRAW_HDR + "\n\n" + "\n\n----------\n\n".join(parts)


# ---- Independent summaries ----
# Each source attempt is summarized on its own, from exactly the block the ReGE-Syn coach sees for it (the
# pool-level serialization split per candidate, so its caps and redaction carry over), with a 2,500-token output
# request; the summaries are concatenated under NAPLAN_HDR. The summary schema follows the structured summaries of
# agentic PDR (Kim et al., 2026). The first output that holds a complete JSON object with the required keys is
# kept; up to three attempts, the retries asking for a shorter summary. If none qualifies, the last non-empty
# output is kept and flagged, and an all-empty result becomes a placeholder. Summaries are cached on disk, keyed
# by task, source, prompt text, output request and model, so a rerun does not repeat a finished summary.
SUM1_SYS = ("You convert an agent rollout into a compact, structured summary which preserves its salient "
            "hypotheses, progress, and failure modes while discarding low-signal trace details. "
            "Output STRICT JSON only, following exactly this schema:\n"
            '{"task_requirements": {"primary_objective": str, "expected_behavior": str, '
            '"current_behavior": str, "reproduction_steps": str, "explicit_constraints": [str], '
            '"success_criteria": [str]},\n'
            ' "agent_actions": {"exploration": {"files_examined": [str], "navigation": [str], '
            '"commands_used": [str], "key_discoveries": [str]},\n'
            '  "solution_approach": {"strategy": str, "stated_reasoning": str, "approach_changes": [str]},\n'
            '  "implementation": {"modification_scope": str, "files_modified": [str], "edit_methods": [str], '
            '"command_dependencies": [str]}},\n'
            ' "artifact_changes": [{"description": str, "file_path": str, "before": str, "after": str}],\n'
            ' "command_execution": {"commands_with_response": [{"step": str, "command": str, '
            '"return_code": int, "stdout_summary": str, "stderr_summary": str}]}}')

PDR_PROMPT_VERSION = "pdr-v3-20260909"   # version of the summary prompts, part of the cache key
SUM4_M_MAX_TOKENS = int(os.environ.get("SUM4_M_MAX_TOKENS", "2500"))
SUM4_FORMAT_RETRIES = int(os.environ.get("SUM4_FORMAT_RETRIES", "3"))
PDR_CACHE_DIR = Path(os.environ.get("PDR_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "result" / "pdr_summ_cache")))


# Length limits are stated in the prompt; a retry repeats the same schema with halved limits and the same output
# request.
def _sum4_limits(words, cmds, items, chars, code):
    return ("\nLENGTH LIMITS (hard): the whole JSON must stay under about {w} words so that it closes properly. "
            "commands_with_response: at most {c} entries -- keep only the commands that changed state, produced key "
            "evidence, or failed; every list: at most {i} items; every string value: at most {s} characters; "
            "before_code/after_code: at most {k} characters each (cut with '...'). Do not repeat a finding in several "
            "fields; do not narrate step by step. Keep concrete commands, paths, values and failure causes."
            ).format(w=words, c=cmds, i=items, s=chars, k=code)


SUM4_M_SYS = SUM1_SYS + _sum4_limits(1000, 6, 6, 250, 300)
SUM4_M_SHRINK = SUM1_SYS + _sum4_limits(500, 3, 4, 120, 150)
SHRINK_NOTE = ("\nYOUR PREVIOUS OUTPUT EXCEEDED THE LENGTH LIMIT AND WAS CUT OFF. Produce the same summary again, "
               "shorter, respecting the halved limits above; the JSON must be complete.")
SUMMARY_PLACEHOLDER = "[summary unavailable: the summarizer returned no usable output after the fixed retries]"
OUTSIDE_BUDGET_TEXT = "[this attempt fell outside the shared pool serialization budget and was not summarized]"


def _pdr_sha(txt):
    return hashlib.sha1(txt.encode("utf-8", "replace")).hexdigest()


def _pdr_cache_path(key_obj):
    d = PDR_CACHE_DIR / (os.environ.get("PDR_DOMAIN") or "misc")
    d.mkdir(parents=True, exist_ok=True)
    return d / (_pdr_sha(json.dumps(key_obj, sort_keys=True)) + ".json")


def _pdr_atomic_write(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _summary_json_ok(txt):
    """Acceptance: the output holds one complete JSON object with the two top-level schema keys. Text around the
    object and a code fence are stripped; the first balanced object is decoded. Content is never altered."""
    s = (txt or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    i = s.find("{")
    if i < 0:
        return False, None
    try:
        obj, end = json.JSONDecoder().raw_decode(s[i:])
    except Exception:
        return False, None
    ok = isinstance(obj, dict) and "task_requirements" in obj and "agent_actions" in obj
    return ok, (s[i:i + end] if ok else None)


def _psyn_visible_blocks(rolls, store):
    """The per-candidate blocks exactly as the ReGE-Syn coach sees them (pool-level cap and redaction inherited)."""
    ser = _ser_capped(rolls, store, False)
    parts = [p for p in re.split(r"(?m)^(?=={5} CANDIDATE \d+ \(model=)", ser) if p]
    return parts, ser, ser.endswith("[POOL SERIALIZATION TRUNCATED AT BUDGET]")


def _summarize_source(task, rid, view, sys_prompt, user, max_tokens, meta_extra):
    key = {"task": task, "source": rid, "view": view, "sys_sha": _pdr_sha(sys_prompt), "user_sha": _pdr_sha(user),
           "max_tokens": max_tokens, "model": A.AGG_MODEL}
    path = _pdr_cache_path(key)
    if path.exists():
        try:
            c = json.load(open(path))
            if c.get("text"):
                return c["text"], c, True
        except Exception:
            pass
    calls, text, valid, hit_cap, nonempty_last, shrink_used = [], "", False, False, "", 0
    for att in range(SUM4_FORMAT_RETRIES):
        before = len(getattr(A._USAGE, "acc", None) or [])
        sp = sys_prompt
        if att > 0:                                   # format retry: same schema, halved limits, same output request
            sp = SUM4_M_SHRINK + SHRINK_NOTE
            shrink_used += 1
        out = (A.agg_call(sp, user, max_tokens=max_tokens, role="coach") or "").strip()
        new = (getattr(A._USAGE, "acc", None) or [])[before:]
        calls.extend(new)
        last_out = new[-1].get("out") if new else None
        if last_out and last_out >= max_tokens - 8:
            hit_cap = True
        if out:
            nonempty_last = out
        ok, cleaned = _summary_json_ok(out)
        if ok:
            text, valid = cleaned, True
            break
        time.sleep(2 * (att + 1))
    placeholder = False
    if not valid:
        if nonempty_last:
            text = nonempty_last          # the model's own summary in an invalid format, flagged
        else:
            text, placeholder = SUMMARY_PLACEHOLDER, True
    rec = {"task": task, "source": rid, "view": view, "prompt_version": PDR_PROMPT_VERSION, "max_tokens": max_tokens,
           "model": A.AGG_MODEL, "valid_json": valid, "hit_cap": hit_cap, "placeholder": placeholder,
           "attempts": len(calls), "shrink_retries": shrink_used, "calls": calls, "in_chars": len(user), "out_chars": len(text),
           "text": text, "built": time.strftime("%F %T"), **meta_extra}
    _pdr_atomic_write(path, rec)
    return text, rec, False


def g_sum4m(task, instr, rolls, store, ctx):
    if not rolls:
        raise RuntimeError("sum4m needs a non-empty pool")
    blocks, _ser, pool_trunc = _psyn_visible_blocks(rolls, store)
    outs, status = [], []
    for i, r in enumerate(rolls):
        meta = {"pool_truncated": pool_trunc}
        src = blocks[i] if i < len(blocks) else None
        if src is None:
            outs.append((r, OUTSIDE_BUDGET_TEXT))
            status.append({"source": r["id"], "outside_budget": True})
            continue
        user = f"TASK:\n{instr}\n\nONE PREVIOUS ATTEMPT TO SUMMARIZE (recorded as-is; correctness unknown):\n\n{src}"
        text, rec, cached = _summarize_source(task, r["id"], "M", SUM4_M_SYS, user, SUM4_M_MAX_TOKENS, meta)
        outs.append((r, text))
        status.append({"source": r["id"], "cached": cached, "valid_json": rec.get("valid_json"),
                       "placeholder": rec.get("placeholder"), "hit_cap": rec.get("hit_cap"),
                       "attempts": rec.get("attempts"), "in_chars": rec.get("in_chars"), "out_chars": rec.get("out_chars")})
    _pdr_atomic_write(_pdr_cache_path({"meta": True, "task": task, "view": "M", "prompt": PDR_PROMPT_VERSION}),
                      {"task": task, "view": "M", "sources": [r["id"] for r in rolls], "status": status,
                       "built": time.strftime("%F %T")})
    body = "\n\n".join(f"### Summary {i + 1} of attempt {r['id']}:\n{t}" for i, (r, t) in enumerate(outs))
    return NAPLAN_HDR + "\n\n" + body


# ---- registry ----
EXTRA_BUILDERS = {"best1": g_best1, "psyn": g_psyn, "psyna": g_psyna, "psynnp": g_psynnp,
                  "allraw": g_allraw, "sum4m": g_sum4m}

# AggAgent loads its own package (aggagent_arm.py), so it is registered only when EXTRA_ARMS asks for it.
if "aggagent" in os.environ.get("EXTRA_ARMS", ""):
    from aggagent_arm import g_aggagent
    EXTRA_BUILDERS["aggagent"] = g_aggagent


def _empty_guide(txt):
    """True for empty text or a lone '## ...' header line (every builder = HDR + '\n\n' + LLM text)."""
    s = (txt or "").strip()
    return (not s) or (s.startswith("## ") and "\n" not in s)


def _usage_begin():
    """Start token accounting for one build: the coach calls (agg_call) and, if loaded, the AggAgent loop."""
    A.usage_reset()
    _ag = sys.modules.get("aggagent_arm")
    if _ag is not None:
        _ag.usage_reset()


def _usage_end():
    calls = A.usage_take()
    _ag = sys.modules.get("aggagent_arm")
    loop = _ag.usage_take() if _ag is not None else None
    rec = {"in": sum((c.get("in") or 0) for c in calls) + ((loop or {}).get("in") or 0),
           "out": sum((c.get("out") or 0) for c in calls) + ((loop or {}).get("out") or 0),
           "attempts": 1}
    if calls:
        rec["coach_calls"] = calls                     # per agg_call usage
    if loop:
        rec["loop"] = loop                             # the AggAgent loop
    return rec


def extra_arm_guides(task, instr, rolls, store, cache, arms=None, usage_out=None):
    """Build guidance for the requested arms (`arms`, or the EXTRA_ARMS env as a comma list). Two attempts per arm;
    an arm that fails twice is skipped and logged, never stored as an empty guidance.
    usage_out: optional dict, filled per arm with the API token usage of the build (summed over retries)."""
    want = [a.strip() for a in (arms if arms is not None else os.environ.get("EXTRA_ARMS", "").split(","))
            if a.strip() in EXTRA_BUILDERS]
    out = {}
    for a in want:
        last = None
        for _ in range(2):
            _usage_begin()
            try:
                g = EXTRA_BUILDERS[a](task, instr, rolls, store, {})
                # A header without text means the model call came back empty; storing it would hand the student a
                # bare header, i.e. an unguided run filed under this arm.
                if _empty_guide(g):
                    raise RuntimeError("empty guidance (LLM call returned nothing; header only)")
                out[a] = g
                last = None
            except Exception as e:
                last = e
            finally:
                if usage_out is not None:
                    u, p = _usage_end(), usage_out.get(a)
                    if p:                              # a retry: keep both attempts' spend
                        u = {**p, **u, "in": (p.get("in") or 0) + (u["in"] or 0),
                             "out": (p.get("out") or 0) + (u["out"] or 0),
                             "attempts": (p.get("attempts") or 1) + 1}
                    usage_out[a] = u
            if last is None:
                break
        if last is not None:
            print(f"    [extra_arm SKIP] {task}/{a} failed twice: {str(last)[:120]}", flush=True)
    return out
