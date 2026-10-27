"""Terminal-Bench 2.0 guide builder (the pool records are already in the common format).
usage: tb2_guides.py build <part> <nparts>
env:   TB_RUNS (source pool), TB2_GUIDE_IDS (task list), TB2_GUIDE_PREFIX (output files), TB2_NEED (arms),
       JUDGE_PICK_FILE (best1), TB2_GUIDE_ALL (optional existing all-file whose guides count as built)."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(part: int, nparts: int):
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    import aggregate_deliverable as A
    import guidance_moa_experiment as G
    ids = sorted(l.strip() for l in open(os.environ.get("TB2_GUIDE_IDS") or (ROOT / "result" / "tb2_ids.txt")) if l.strip())
    mine = [c for i, c in enumerate(ids) if i % nparts == part]
    prefix = os.environ.get("TB2_GUIDE_PREFIX", "tb2_guides")
    outp = ROOT / "result" / f"{prefix}_part{part}.json"
    try:
        out = json.load(open(outp))
    except Exception:
        out = {}
    try:
        allg = json.load(open(os.environ["TB2_GUIDE_ALL"])) if os.environ.get("TB2_GUIDE_ALL") else {}
    except Exception:
        allg = {}
    NEED = tuple(a for a in os.environ.get("TB2_NEED", "psyn,aggagent").split(",") if a)
    for task in mine:
        have = {**(((allg.get(task) or {}).get("guides")) or {}), **(((out.get(task) or {}).get("guides")) or {})}
        have = {kk: v for kk, v in have.items() if v}
        missing = [a for a in NEED if not have.get(a)]
        if not missing:
            continue
        try:
            instr = yaml.safe_load(open(ROOT / "data" / "taskcache_tb2" / task / "task.yaml"))["instruction"]
            store = A.OUT / task / "artifacts"
            rolls = A.load_pool(task, store)
            t0 = time.time()
            cu = {}                                      # per-arm coach token usage
            extra = G.extra_arm_guides(task, instr, rolls, store, {}, arms=missing, usage_out=cu)
            ps = sum(1 for r in rolls if (r["d"].get("ok_grade") or {}).get("ok"))
            guides = {**have, "control": "", **extra}
            out[task] = {"guides": guides, "pool_n": len(rolls), "pool_solve": ps,
                         "extra_chars": {kk: len(v) for kk, v in guides.items() if kk != "control"},
                         **({"agg_max_iter": int(os.environ.get("AGG_MAX_ITER", "100"))} if "aggagent" in missing else {}),
                         "coach_usage": {**((((out.get(task) or {}).get("coach_usage")) or {})), **cu},
                         "built": time.strftime("%F %T")}
            print(f"{task}: {time.time()-t0:.0f}s built={missing} "
                  f"chars={out[task]['extra_chars']}", flush=True)
        except Exception as e:
            print(f"{task}: FAILED {str(e)[:120]}", flush=True)
            out[task] = {"error": str(e)[:200]}
        json.dump(out, open(outp, "w"), indent=1)
    print(f"part{part}: {len(out)} tasks done", flush=True)


if __name__ == "__main__":
    build(int(sys.argv[2]), int(sys.argv[3])) if sys.argv[1] == "build" else exit(1)
