#!/usr/bin/env python3
"""Task success per benchmark and model line from the pipeline outputs (the accuracy columns of the main table).

For each arm (Control, ReGE-Syn, ReGE-Sel, AggAgent) the rate is taken over the tasks with a valid Control record
that also have a valid record of that arm; a record is valid when it carries a grade and no infrastructure error.
Best-of-N is the original grade of the judge-selected source on the same tasks, ReGE is the higher of ReGE-Syn and
ReGE-Sel, and coverage is the share of tasks with at least one successful source attempt (first four attempts).

usage: python scripts/summarize.py [--line gpt claude] [--bench tb2 spider androidworld bcp] [--out result/summary.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("control", "psyn", "best1", "aggagent")
NAMES = {"control": "Control", "psyn": "ReGE-Syn", "best1": "ReGE-Sel", "aggagent": "AggAgent"}
LAYOUT = {
    ("tb2", "gpt"): dict(pool="runs_tb2", arms="runs_tb2_arms", key="gpt56luna", pool_key="gpt56luna", picks="judge_picks_tb2_luna.json"),
    ("tb2", "claude"): dict(pool="runs_tb2_s5pool", arms="runs_tb2_s5darms", key="claude-sonnet5b", pool_key="claude-sonnet5", picks="judge_picks_tb2_s5d.json"),
    ("spider", "gpt"): dict(pool="runs_spider_house", arms="runs_spider_arms", key="gpt56luna", pool_key="gpt56luna", picks="judge_picks_spider_luna.json", grade="{arm}"),
    ("spider", "claude"): dict(pool="runs_spider_house_s5e", arms="runs_spider_s5earms", key="claude-sonnet5b", pool_key="claude-sonnet5b", picks="judge_picks_spider_s5e.json", grade="{arm}_s5e"),
    ("androidworld", "gpt"): dict(pool="runs_droid_house", arms="runs_droid_arms_ilv0825", key="gpt56luna", pool_key="gpt56luna", picks="judge_picks_droid_luna.json"),
    ("androidworld", "claude"): dict(pool="runs_droid_house_s5e", arms="runs_droid_s5xarms", key="claude-sonnet5b", pool_key="claude-sonnet5b", picks="judge_picks_droid_s5e.json"),
    ("bcp", "gpt"): dict(pool="runs_bcp_lunapool", arms="runs_bcp_lunaarms", key="gpt56luna", pool_key="gpt56luna", picks="judge_picks_bcp_luna.json"),
    ("bcp", "claude"): dict(pool="runs_bcp_s5b200pool", arms="runs_bcp_s5b200arms", key="claude-sonnet5b", pool_key="claude-sonnet5b", picks="judge_picks_bcp_s5b200.json"),
}


def read(path):
    return json.loads(Path(path).read_text())


def valid(d, droid=False):
    errors = [d.get(k) for k in ("error", "grade_error", "infra_error") if d.get(k)]
    errors += [s["error"] for s in d.get("steps", []) if isinstance(s, dict) and s.get("error")]
    errors = [e for e in errors if not any(w in str(e).lower() for w in ("wall-timeout", "wall timeout", "wall cap", "agent_s"))]
    return not errors and (d.get("success") if droid else (d.get("ok_grade") or {}).get("ok")) is not None


def outcomes(bench, cfg, arm):
    if bench == "spider":
        g = ROOT / "result" / f"spider_grade_{cfg['grade'].format(arm=arm)}_{cfg['key']}.json"
        if not g.exists():
            return {}
        return {t: v == 1 for t, v in read(g).get("0", {}).items() if v in (0, 1)}
    droid = bench == "androidworld"
    pattern = f"*/{cfg['key']}_c*_0.json" if droid else f"*/{cfg['key']}_0.json"
    out = {}
    for p in sorted((ROOT / "data" / cfg["arms"] / arm).glob(pattern)):
        d = read(p)
        if valid(d, droid):
            task = p.parent.name + (f"__c{d.get('combo', 0)}" if droid else "")
            out[task] = bool(d["success"] if droid else d["ok_grade"]["ok"])
    return out


def summarize(bench, line):
    cfg = LAYOUT[(bench, line)]
    res = {a: outcomes(bench, cfg, a) for a in ARMS}
    ids = set(res["control"])
    if not ids:
        return None
    row = {}
    for a in ARMS:
        common = ids & set(res[a])
        row[NAMES[a]] = {"success": sum(res[a][t] for t in common), "n": len(common)}
    picks_f = ROOT / "result" / cfg["picks"]
    picks = read(picks_f) if picks_f.exists() else {}
    selected, covered = {}, []
    for t in sorted(ids):
        pool = {}
        for p in (ROOT / "data" / cfg["pool"] / t).glob(f"{cfg['pool_key']}_[0-3].json"):
            grade = (read(p).get("ok_grade") or {}).get("ok")
            if grade is not None:
                pool[p.stem] = bool(grade)
        if pool:
            covered.append(any(pool.values()))
        if picks.get(t) in pool:
            selected[t] = pool[picks[t]]
    row["Best-of-N"] = {"success": sum(selected.values()), "n": len(selected)}
    row["coverage"] = {"success": sum(covered), "n": len(covered)}
    rate = {k: (100 * v["success"] / v["n"] if v["n"] else None) for k, v in row.items()}
    syn, sel = rate["ReGE-Syn"], rate["ReGE-Sel"]
    rate["ReGE"] = max(x for x in (syn, sel) if x is not None) if (syn is not None or sel is not None) else None
    return {"counts": row, "percent": rate}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--line", nargs="+", default=["gpt", "claude"], choices=["gpt", "claude"])
    ap.add_argument("--bench", nargs="+", default=["tb2", "spider", "androidworld", "bcp"],
                    choices=["tb2", "spider", "androidworld", "bcp"])
    ap.add_argument("--out", type=Path, default=ROOT / "result" / "summary.json")
    args = ap.parse_args()
    report = {}
    cols = ["Control", "Best-of-N", "AggAgent", "ReGE-Syn", "ReGE-Sel", "ReGE", "coverage"]
    print(f"{'line':7s} {'benchmark':13s} " + " ".join(f"{c:>9s}" for c in cols) + "   n")
    for line in args.line:
        for bench in args.bench:
            r = summarize(bench, line)
            if r is None:
                continue
            report[f"{line}/{bench}"] = r
            cells = " ".join(f"{r['percent'][c]:9.1f}" if r["percent"].get(c) is not None else f"{'-':>9s}" for c in cols)
            print(f"{line:7s} {bench:13s} {cells}   {r['counts']['Control']['n']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
