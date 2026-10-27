#!/usr/bin/env python3
"""Merge outputs that are built in parallel parts.

  guides <prefix>                         result/<prefix>_part*.json (+ an existing result/<prefix>_all.json)
                                          -> result/<prefix>_all.json; per task and arm the longest non-empty guide wins
  picks <judge_sweep.json> <judge_model> <out.json>
                                          judge_sweep output -> {task: picked rollout id} for JUDGE_PICK_FILE
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def merge_guides(prefix):
    out_path = ROOT / "result" / f"{prefix}_all.json"
    files = ([out_path] if out_path.exists() else []) + [Path(p) for p in sorted(glob.glob(str(ROOT / "result" / f"{prefix}_part*.json")))]
    merged = {}
    for f in files:
        for task, rec in json.load(open(f)).items():
            if not isinstance(rec, dict):
                continue
            m = merged.setdefault(task, {"guides": {}})
            for arm, text in (rec.get("guides") or {}).items():
                if text and len(text) >= len(m["guides"].get(arm) or ""):
                    m["guides"][arm] = text
            for k, v in rec.items():
                if k not in ("guides", "extra_chars", "error"):
                    m[k] = v
            if rec.get("error") and not m["guides"]:
                m["error"] = rec["error"]
    for m in merged.values():
        m["guides"].setdefault("control", "")
        m["extra_chars"] = {a: len(t) for a, t in m["guides"].items() if a != "control"}
    json.dump(merged, open(out_path, "w"), indent=1)
    arms = sorted({a for m in merged.values() for a, t in m["guides"].items() if t})
    print(f"{out_path.name}: {len(merged)} tasks; arms with guidance: "
          + ", ".join(f"{a} {sum(1 for m in merged.values() if m['guides'].get(a))}" for a in arms))


def picks(sweep, judge, out):
    d = json.load(open(sweep))
    p = {t: r["picks"][judge] for t, r in d.items()
         if t != "_meta" and isinstance(r, dict) and (r.get("picks") or {}).get(judge)}
    json.dump(p, open(out, "w"), indent=1)
    print(f"{out}: {len(p)} picks from {sweep}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "guides":
        merge_guides(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == "picks":
        picks(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
