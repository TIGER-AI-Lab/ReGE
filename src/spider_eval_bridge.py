"""Bridge to the official Spider2-Lite comparison functions.

evaluate.py redirects sys.stdout to a TeeOutput("log.txt") when it is imported, so it is imported from a scratch
working directory and stdout/stderr are restored right after. The comparison functions
(compare_multi_pandas_table / compare_pandas_table) take DataFrames and do not touch the disk.
"""
import os
import sys
from pathlib import Path

EVALS = Path(os.environ.get("SPIDER2_LITE_DIR", Path(__file__).resolve().parent.parent / "third_party" / "Spider2" / "spider2-lite")) / "evaluation_suite"
WORK = Path(__file__).resolve().parent.parent / "data" / "spider_evalwork"
WORK.mkdir(parents=True, exist_ok=True)

_out, _err, _cwd = sys.stdout, sys.stderr, os.getcwd()
os.chdir(WORK)
sys.path.insert(0, str(EVALS))
try:
    import evaluate as _ev
finally:
    sys.stdout, sys.stderr = _out, _err
    os.chdir(_cwd)


def compare_multi(pred, gold_dfs, condition_cols, ignore_order) -> int:
    return _ev.compare_multi_pandas_table(pred, gold_dfs, condition_cols, ignore_order)
