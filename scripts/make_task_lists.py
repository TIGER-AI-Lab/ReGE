#!/usr/bin/env python3
"""Write the task lists the pipelines read from result/.

  result/tb2_ids.txt               the 89 Terminal-Bench 2.0 tasks (sorted task directories)
  result/spider_ids.txt            the 135 gradeable local-SQLite Spider2-Lite instances
  result/bcp_luna_ids.txt          the 830 BrowseComp-Plus queries in run order (configs/bcp_query_order.txt)

AndroidWorld instance lists are written by run_androidworld.sh from the transcribed pools.
usage: python scripts/make_task_lists.py [tb2] [spider] [bcp]      (default: all three)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "result"


def write(name, ids):
    RESULT.mkdir(exist_ok=True)
    (RESULT / name).write_text("".join(f"{i}\n" for i in ids))
    print(f"{name}: {len(ids)}")


def tb2():
    ds = ROOT / "data" / "tb2_dataset" / "terminal-bench"
    ids = sorted(p.parent.name for p in ds.glob("*/task.toml"))
    if len(ids) != 89:
        raise SystemExit(f"expected 89 Terminal-Bench 2.0 tasks under {ds}, found {len(ids)}")
    write("tb2_ids.txt", ids)


def spider():
    sys.path.insert(0, str(ROOT / "src"))
    import spider_collect
    ids = sorted(spider_collect.gradeable_ids())
    if len(ids) != 135:
        raise SystemExit(f"expected 135 gradeable Spider2-Lite local instances, found {len(ids)}")
    write("spider_ids.txt", ids)


def bcp():
    order = [l.strip() for l in open(ROOT / "configs" / "bcp_query_order.txt") if l.strip()]
    if len(order) != 830 or len(set(order)) != 830:
        raise SystemExit("configs/bcp_query_order.txt must list 830 distinct queries")
    write("bcp_luna_ids.txt", order)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["tb2", "spider", "bcp"]
    for t in targets:
        {"tb2": tb2, "spider": spider, "bcp": bcp}[t]()
