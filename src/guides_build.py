#!/usr/bin/env python3
"""Guide builder for pools whose records are already in the common format (steps + final_files + artifacts/), used
for BrowseComp-Plus: builds the requested arms (psyn, best1, aggagent, ...) for every task of a list.
usage: TB_RUNS=<runs dir, relative to data/> TASKCACHE=<taskcache dir> JUDGE_PICK_FILE=<picks.json> \
       [EXTRA_ARMS=aggagent AGG_URL=... AGG_MODEL=... AGGAGENT_MODEL_KEY=... GUIDE_POOL_K=<k>] \
       python guides_build.py <ids_file> <out_prefix> <arms,comma> <part> <nparts>
Writes result/<out_prefix>_part<part>.json ({task: {guides, pool_n, pool_solve, extra_chars, agg_max_iter, coach_usage}}).
Guides already in result/<out_prefix>_all.json or in the part file are not built again. best1 needs no model call;
psyn and aggagent call the coach (agg_call) or the AggAgent loop."""
import json, os, sys, time
_ids_file, _prefix, _arms, _part, _nparts = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import yaml
import aggregate_deliverable as A
import guidance_moa_experiment as G
import tb_collect as TC

ARMS = [a for a in _arms.split(",") if a]


def _fmt_usage(cu):
    """coach_usage -> 'psyn 2.1k/1.4k aggagent 187.3k/6.2k 23r' (API input/output tokens; r = AggAgent loop rounds)."""
    bits = []
    for a, u in cu.items():
        lp = u.get("loop") or {}
        bits.append(f"{a} {(u.get('in') or 0) / 1000:.1f}k/{(u.get('out') or 0) / 1000:.1f}k"
                    + (f" {lp.get('iterations')}r" if lp else ""))
    return " ".join(bits) or "-"
ids = [l.strip() for l in open(_ids_file) if l.strip()]
mine = [t for i, t in enumerate(ids) if i % _nparts == _part]
outp = os.path.join(ROOT, "result", f"{_prefix}_part{_part}.json")
try:
    out = json.load(open(outp))
except Exception:
    out = {}
try:
    allg = json.load(open(os.path.join(ROOT, "result", f"{_prefix}_all.json")))
except Exception:
    allg = {}
missing_arms = [a for a in ARMS if a not in G.EXTRA_BUILDERS]
if missing_arms:
    raise SystemExit(f"arms not registered in EXTRA_BUILDERS: {missing_arms} (aggagent needs EXTRA_ARMS=aggagent in env)")
for task in mine:
    have = {**(((allg.get(task) or {}).get("guides")) or {}), **(((out.get(task) or {}).get("guides")) or {})}
    have = {k: v for k, v in have.items() if v}
    missing = [a for a in ARMS if not have.get(a)]
    if not missing:
        if task not in out and task in allg:
            out[task] = allg[task]
        continue
    t0 = time.time()
    try:
        instr = yaml.safe_load(open(TC.TASKCACHE / task / "task.yaml"))["instruction"]
        store = A.OUT / task / "artifacts"
        _pk = int(os.environ.get("GUIDE_POOL_K", "0"))       # source-count ablation: only the first K attempts
        rolls = A.load_pool(task, store, mcap=_pk if _pk > 0 else None)
        if _pk > 0:
            rolls = rolls[:_pk]
        cu = {}                                     # per-arm API token usage of this build
        extra = G.extra_arm_guides(task, instr, rolls, store, {}, arms=missing, usage_out=cu)
        ps = sum(1 for r in rolls if (r["d"].get("ok_grade") or {}).get("ok"))
        guides = {**have, "control": "", **extra}
        prev = out.get(task) if isinstance(out.get(task), dict) else {}
        prev_cu = prev.get("coach_usage") if isinstance(prev.get("coach_usage"), dict) else {}
        out[task] = {**{k: v for k, v in prev.items() if k not in ("guides", "extra_chars", "error", "coach_usage")},
                     "guides": guides, "pool_n": len(rolls), "pool_solve": ps,
                     "extra_chars": {k: len(v) for k, v in guides.items() if k != "control"},
                     **({"agg_max_iter": int(os.environ.get("AGG_MAX_ITER", "100"))} if "aggagent" in missing else {}),
                     "coach_usage": {**prev_cu, **cu},
                     "built": time.strftime("%F %T")}
        print(f"{task}: {time.time()-t0:.0f}s built={missing} chars={out[task]['extra_chars']} pool_n={len(rolls)} solve={ps} usage={_fmt_usage(cu)}", flush=True)
    except Exception as e:
        print(f"{task}: FAILED {str(e)[:160]}", flush=True)
        out[task] = {**(out.get(task) if isinstance(out.get(task), dict) else {}), "error": str(e)[:200]}
    json.dump(out, open(outp, "w"), indent=1)
print(f"part{_part}: {len(out)} tasks in {outp}", flush=True)
