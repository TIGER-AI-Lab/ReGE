"""Spider2-Lite source pool -> common-format records, and guide building.

transcribe: <SPIDER_POOL_DIR>/<iid>/<key>_<k>.json -> <SPIDER_HOUSE_DIR>/<iid>/<key>_<k>.json
  steps       = the agent's SQL calls ({'think','cmd','obs'})
  final_files = {'__sql__': sha, '__answer__': sha} with artifacts/<sha> holding the submitted SQL
                and the first rows of its executed result (what the coach sees; no labels)
  ok_grade    = the official result comparison from the pool's grade file (SPIDER_POOL_GRADES)

build <part> <nparts>: builds the arms in SPIDER_NEED for the pool's tasks into
  result/<SPIDER_GUIDE_PREFIX>_part<p>.json (env: TB_RUNS=<house dir>, TASKCACHE=data/taskcache_spider,
  JUDGE_PICK_FILE for best1, AGG_URL/AGG_MODEL for the coach). scripts/run_spider.sh sets all of them.
"""
from __future__ import annotations
import csv, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Pool directory, common-format directory, model key and grade file of the run (scripts/run_spider.sh sets them).
POOL_KEY = os.environ.get("SPIDER_POOL_KEY", "gpt56luna")                            # file family <key>_<k>.json/.csv
SRC_RUNS = ROOT / "data" / os.environ.get("SPIDER_POOL_DIR", "runs_spider")
HOUSE = ROOT / "data" / os.environ.get("SPIDER_HOUSE_DIR", "runs_spider_house")
GRADES = ROOT / "result" / os.environ.get("SPIDER_POOL_GRADES", f"spider_grade_pool_{POOL_KEY}.json")
POOL_K = int(os.environ.get("SPIDER_POOL_K", "4"))                                    # attempts per task in the pool
SERVED = {"gpt56luna": "gpt-5.6-luna", "claude-sonnet5b": "claude-sonnet-5", "claude-sonnet5": "claude-sonnet-5"}.get(POOL_KEY, POOL_KEY)


def sha16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]


def transcribe_all():
    g = json.load(open(GRADES))
    n = 0
    for iid_dir in sorted(SRC_RUNS.iterdir()):
        iid = iid_dir.name
        for f in sorted(iid_dir.glob(f"{POOL_KEY}_*.json")):
            k = f.stem[len(POOL_KEY) + 1:]
            if not k.isdigit() or int(k) >= POOL_K:
                continue
            rec = json.load(open(f))
            tdir = HOUSE / iid
            (tdir / "artifacts").mkdir(parents=True, exist_ok=True)
            steps = []
            for c in rec.get("calls", []):
                if "sql" in c:
                    steps.append({"think": "", "cmd": c["sql"][:600],
                                  "obs": "ok" if c.get("ok") else "SQL ERROR"})
                elif "submit_err" in c:
                    steps.append({"think": "", "cmd": "[submit rejected]",
                                  "obs": c["submit_err"][:300]})
            final_sql = rec.get("final_sql") or "(no submission)"
            ans_rows = ""
            csvp = iid_dir / f"{POOL_KEY}_{k}.csv"
            if csvp.exists():
                try:
                    rows = list(csv.reader(open(csvp)))[:31]
                    ans_rows = "\n".join(" | ".join(x[:80] for x in r)[:400] for r in rows)
                except Exception:
                    ans_rows = "(csv unreadable)"
            payloads = {"__sql__": final_sql[:8000],
                        "__answer__": (ans_rows or "(no result)")[:8000]}
            ff = {}
            for key, p in payloads.items():
                h = sha16(p)
                ff[key] = h
                fp = tdir / "artifacts" / h
                if not fp.exists():
                    fp.write_text(p)
            ok = g.get(k, {}).get(iid, 0) == 1
            out = {"task": iid, "model": POOL_KEY, "idx": int(k), "steps": steps,
                   "final_files": ff, "baseline": {},
                   "ok_grade": {"ok": bool(ok), "msg": "official exec_result compare"},
                   "collector": "spider_guides.transcribe", "served_model": SERVED,
                   "src_file": str(f)}
            json.dump(out, open(tdir / f"{POOL_KEY}_{k}.json", "w"), indent=1)
            n += 1
    print(f"transcribed {n} rollouts -> {HOUSE} (pool {SRC_RUNS}, key {POOL_KEY}, grades {GRADES.name})")


def build(part: int, nparts: int):
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    import aggregate_deliverable as A
    import guidance_moa_experiment as G
    ids = sorted(d.name for d in HOUSE.iterdir() if d.is_dir())
    _gf = os.environ.get("SPIDER_GUIDE_IDS")            # optional task list
    if _gf:
        _keep = {l.strip() for l in open(_gf) if l.strip()}
        ids = [c for c in ids if c in _keep]
    mine = [c for i, c in enumerate(ids) if i % nparts == part]
    prefix = os.environ.get("SPIDER_GUIDE_PREFIX", "spider_guides")
    outp = ROOT / "result" / f"{prefix}_part{part}.json"
    try:
        out = json.load(open(outp))
    except Exception:
        out = {}
    try:                                                 # guides already built in the merged all-file
        allg = json.load(open(ROOT / "result" / os.environ.get("SPIDER_GUIDE_ALL", "spider_guides_all.json")))
    except Exception:
        allg = {}
    NEED = tuple(a for a in os.environ.get("SPIDER_NEED", "psyn,aggagent").split(",") if a)
    for task in mine:
        have = {**(((allg.get(task) or {}).get("guides")) or {}), **(((out.get(task) or {}).get("guides")) or {})}
        have = {kk: v for kk, v in have.items() if v}
        missing = [a for a in NEED if not have.get(a)]
        if not missing:
            continue
        try:
            instr = yaml.safe_load(open(ROOT / "data" / "taskcache_spider" / task / "task.yaml"))["instruction"]
            store = A.OUT / task / "artifacts"
            rolls = A.load_pool(task, store)
            t0 = time.time()
            extra = G.extra_arm_guides(task, instr, rolls, store, {}, arms=missing)
            ps = sum(1 for r in rolls if (r["d"].get("ok_grade") or {}).get("ok"))
            guides = {**have, "control": "", **extra}
            out[task] = {"guides": guides, "pool_n": len(rolls), "pool_solve": ps,
                         "extra_chars": {kk: len(v) for kk, v in guides.items() if kk != "control"},
                         **({"agg_max_iter": int(os.environ.get("AGG_MAX_ITER", "100"))} if "aggagent" in missing else {})}
            print(f"{task}: {time.time()-t0:.0f}s built={missing} "
                  f"chars={out[task]['extra_chars']}", flush=True)
        except Exception as e:
            print(f"{task}: FAILED {str(e)[:120]}", flush=True)
            out[task] = {"error": str(e)[:200]}
        json.dump(out, open(outp, "w"), indent=1)
    print(f"part{part}: {len(out)} tasks done", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "transcribe":
        transcribe_all()
    elif sys.argv[1] == "build":
        build(int(sys.argv[2]), int(sys.argv[3]))
