"""ReGE-Sel judge: for every task of one benchmark, show the judge model the source pool (A.load_pool order) at the
ReGE-Sel view (200 reasoning / 160 command characters per step, at most 120 steps) without success labels, and
record the ONE rollout it picks. The picks are stored here and handed to the guide builders via JUDGE_PICK_FILE
(scripts/merge_parts.py picks); the original grade of the picked source is the Best-of-N result.

env:   TB_RUNS (source pool), TASKCACHE (task instructions), AGG_URL (judge endpoint)
usage: python3 judge_sweep.py <bench_tag> <@file|task1,task2,...> [judges_csv] [workers]
       task ids as in the task lists; judges_csv defaults to AGG_MODEL
out:   ../result/judge_sweep_<bench_tag>.json
"""
from __future__ import annotations
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_argv = sys.argv[1:]
import yaml
import tb_collect as TC
import aggregate_deliverable as A
import guidance_moa_experiment as G

BENCH = _argv[0]
_c = _argv[1]
CASES = [l.strip() for l in open(_c[1:])] if _c.startswith("@") else [t for t in _c.split(",") if t.strip()]
JUDGES = [j.strip() for j in (_argv[2] if len(_argv) > 2 else A.AGG_MODEL).split(",")]
WORKERS = int(_argv[3]) if len(_argv) > 3 else 6
OUT_F = Path(__file__).resolve().parent.parent / "result" / f"judge_sweep_{BENCH}.json"
TRIES = 8                       # judge attempts per task, each with agg_call's own retries


def judge_once(model, instr, rolls):
    listing = "\n\n".join(G._render_traj(r, 200, 160, G.V250_STEPS) for r in rolls)      # ReGE-Sel view
    user = (f"TASK:\n{instr}\n\n{len(rolls)} independent agents attempted this task (success labels hidden):\n\n"
            + listing + "\n\nWhich ONE rollout is most likely FULLY correct end-to-end? "
            "Reply with EXACTLY one rollout id (e.g. claude-sonnet_1) and nothing else.")
    # The output request grows over the attempts (2,000 / 8,000 / 20,000 tokens). The id is parsed from the reply:
    # an exact match first, else the last pool id mentioned.
    for att in range(1, TRIES + 1):
        mt = 2000 if att <= 2 else (8000 if att <= 4 else 20000)
        txt = (A.agg_call("You judge agent trajectories.", user, model=model, max_tokens=mt, role="judge") or "").strip()
        match = next((r for r in rolls if r["id"] == txt), None)
        if match is None and txt:
            seen = []                            # (last-mention pos, id); (?!\d) so _1 can't hit inside _10
            for r in rolls:
                ms = [m.start() for m in re.finditer(re.escape(r["id"]) + r"(?!\d)", txt)]
                if ms:
                    seen.append((ms[-1], r["id"]))
            if seen:
                match = next(r for r in rolls if r["id"] == max(seen)[1])
        if match is not None:
            return match["id"], att, len(user)
        time.sleep(3 * att)
    raise RuntimeError(f"judge {model} gave no parsable pick after {TRIES} tries (last: {txt[:80]!r})")


def work(task):
    instr = yaml.safe_load(open(TC.TASKCACHE / task / "task.yaml"))["instruction"]
    store = A.OUT / task / "artifacts"
    rolls = A.load_pool(task, store)
    if not rolls:
        return task, {"error": "empty pool"}
    rec = {"pool_n": len(rolls), "pool_ok": {r["id"]: bool(G._r_ok(r)) for r in rolls},
           "picks": {}, "picked_ok": {}, "attempts": {}, "prompt_chars": None}
    for jm in JUDGES:
        try:
            rid, att, nch = judge_once(jm, instr, rolls)
        except Exception as e:
            # No parsable pick after TRIES attempts: fall back to the first source without consulting
            # labels; attempts=-1 marks the fallback.
            rid, att, nch = rolls[0]["id"], -1, 0
            print(f"  [{task}] {jm} FALLBACK(label-free, pool[0]) -> {rid} ({str(e)[:50]})", flush=True)
        rec["picks"][jm] = rid
        rec["picked_ok"][jm] = rec["pool_ok"][rid]
        rec["attempts"][jm] = att
        rec["prompt_chars"] = nch
        print(f"  [{task}] {jm} -> {rid} (ok={rec['pool_ok'][rid]}, attempt {att})", flush=True)
    return task, rec


def main():
    print(f"[*] judge_sweep {BENCH}: {len(CASES)} tasks x judges={JUDGES}; pool={TC.OUT}", flush=True)
    out = {"_meta": {"bench": BENCH, "judges": JUDGES, "tb_runs": str(TC.OUT), "n_tasks": len(CASES),
                     "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "protocol": "one call per judge per task, g_best1 prompt at the ReGE-Sel view (200/160/120), "
                                 "parse-retry x8 w/ max_tokens escalation"}}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for task, rec in ex.map(work, CASES):
            out[task] = rec
    json.dump(out, open(OUT_F, "w"), indent=1)
    solved = [t for t in out if t != "_meta" and "error" not in out[t]]
    if len(JUDGES) == 2:
        j1, j2 = JUDGES
        agree = sum(1 for t in solved if out[t]["picks"][j1] == out[t]["picks"][j2])
        print(f"[*] agreement {j1} vs {j2}: {agree}/{len(solved)}")
    for jm in JUDGES:
        hit = sum(1 for t in solved if out[t]["picked_ok"][jm])
        print(f"[*] {jm}: picked-a-true-solver on {hit}/{len(solved)} tasks")
    print(f"[*] wrote {OUT_F}")


if __name__ == "__main__":
    main()
