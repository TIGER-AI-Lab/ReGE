"""AndroidWorld source pool -> common-format records, and guide building.

transcribe: <DROID_POOL_DIR>/<Tpl>/<key>_c<c>_<k>.json -> <DROID_HOUSE_DIR>/<Tpl>__c<c>/<key>_<k>.json
  steps       = the agent's per-step action summaries ({'cmd':'[ui-step]','obs': summary}), T3A's own summaries
  final_files = {'__actions__': sha, '__answer__': sha} with artifacts/<sha> holding the joined
                action narrative and the final step summary (what the coach sees; no labels)
  ok_grade    = the official is_successful device-state verdict recorded at rollout time
  taskcache_<house>/<iid>/task.yaml carries the instance goal (parameters of the combination).

build <part> <nparts>: builds the arms in DROID_NEED for the pool's instances into
  result/<DROID_GUIDE_PREFIX>_part<p>.json (env: TB_RUNS=<house dir>, TASKCACHE=data/taskcache_<house>,
  JUDGE_PICK_FILE for best1, AGG_URL/AGG_MODEL for the coach). scripts/run_androidworld.sh sets all of them.
Runs under the main Python environment.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Pool directory, common-format directory, model key and output files of the run (scripts/run_androidworld.sh sets them).
POOL_KEY = os.environ.get("DROID_POOL_KEY", "gpt56luna")
SRC = ROOT / "data" / os.environ.get("DROID_POOL_DIR", "runs_droid")
HOUSE = ROOT / "data" / os.environ.get("DROID_HOUSE_DIR", "runs_droid_house")
TC = ROOT / "data" / ("taskcache_" + HOUSE.name[len("runs_"):])
PREFIX = os.environ.get("DROID_GUIDE_PREFIX", "droid_guides")               # part/all file family
POOL_K = int(os.environ.get("DROID_POOL_K", "4"))
SERVED = {"gpt56luna": "gpt-5.6-luna", "claude-sonnet5b": "claude-sonnet-5", "claude-sonnet5": "claude-sonnet-5"}.get(POOL_KEY, POOL_KEY)


def sha16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16]


def transcribe_all():
    import yaml
    n = 0
    for tdir in sorted(SRC.iterdir()):
        if not tdir.is_dir() or tdir.name.startswith("Vlc"):
            continue
        for f in sorted(tdir.glob(f"{POOL_KEY}_c*_*.json")):
            rec = json.load(open(f))
            c = rec.get("combo", 0)
            k = rec.get("k", rec.get("idx", 0))
            if k >= POOL_K:
                continue
            iid = f"{tdir.name}__c{c}"
            d = HOUSE / iid
            (d / "artifacts").mkdir(parents=True, exist_ok=True)
            sums = rec.get("summaries") or []
            steps = [{"cmd": None, "files": {}}] + [
                {"cmd": "[ui-step]", "obs": s[:400]} for s in sums]
            narrative = "\n".join(f"{i+1}. {s[:300]}" for i, s in enumerate(sums)) or "(no steps)"
            payloads = {"__actions__": narrative[:9000],
                        "__answer__": (sums[-1][:800] if sums else "(no steps)")}
            ff = {}
            for key, p in payloads.items():
                h = sha16(p)
                ff[key] = h
                fp = d / "artifacts" / h
                if not fp.exists():
                    fp.write_text(p)
            out = {"task": iid, "model": POOL_KEY, "idx": int(k), "steps": steps,
                   "final_files": ff, "baseline": {},
                   "ok_grade": {"ok": bool(rec.get("success")), "msg": "official is_successful"},
                   "collector": "droid_guides.transcribe", "served_model": SERVED,
                   "src_file": str(f), "error": rec.get("error")}
            json.dump(out, open(d / f"{POOL_KEY}_{k}.json", "w"), indent=1)
            tcd = TC / iid
            tcd.mkdir(parents=True, exist_ok=True)
            if not (tcd / "task.yaml").exists():
                yaml.safe_dump({"instruction": rec.get("goal", ""), "template": tdir.name,
                                "combo": c}, open(tcd / "task.yaml", "w"),
                               allow_unicode=True, width=10000)
            n += 1
    print(f"transcribed {n} rollouts -> {HOUSE} (pool {SRC}, key {POOL_KEY})")


def build(part: int, nparts: int):
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    import aggregate_deliverable as A
    import guidance_moa_experiment as G
    ids = sorted(d.name for d in HOUSE.iterdir() if d.is_dir())
    _gf = os.environ.get("GUIDE_IDS_FILE")           # optional instance or template list
    if _gf:
        _keep = {l.strip() for l in open(_gf) if l.strip()}
        ids = [c for c in ids if c in _keep or c.split("__")[0] in _keep]
    mine = [c for i, c in enumerate(ids) if i % nparts == part]
    outp = ROOT / "result" / f"{PREFIX}_part{part}.json"
    try:
        out = json.load(open(outp))
    except Exception:
        out = {}
    # Guides already in the merged all-file are not built again.
    try:
        allg = json.load(open(ROOT / "result" / f"{PREFIX}_all.json"))
    except Exception:
        allg = {}
    NEED = tuple(a for a in os.environ.get("DROID_NEED", "psyn,aggagent").split(",") if a)
    for task in mine:
        have = {**(((allg.get(task) or {}).get("guides")) or {}), **(((out.get(task) or {}).get("guides")) or {})}
        have = {kk: v for kk, v in have.items() if v}
        missing = [a for a in NEED if not have.get(a)]
        if not missing:
            if task not in out and task in allg:       # keep the entry in this part file
                out[task] = allg[task]
            continue
        try:
            instr = yaml.safe_load(open(TC / task / "task.yaml"))["instruction"]
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
    if sys.argv[1] == "transcribe":
        transcribe_all()
    elif sys.argv[1] == "build":
        build(int(sys.argv[2]), int(sys.argv[3]))
