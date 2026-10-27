"""Spider2-Lite collector and grader (local SQLite instances, official execution-result comparison).

- Instances: the `local*` instances of spider2-lite.jsonl (no cloud credentials needed), sorted by id.
- Databases are opened read-only. The agent explores with execute_sql (truncated previews), then submits one final
  SELECT with submit_sql; its full result is written to <iid>.csv, the official exec_result submission format.
  Budget: SPIDER_MAX_CALLS model calls (18).
- Grading calls the official comparison functions of evaluation_suite/evaluate.py, with the gold result variants,
  condition_cols and ignore_order of gold/spider2lite_eval.jsonl.
- Guidance (an arm's guide) is placed before the task prompt.

Layout: data/<SPIDER_OUT>/<iid>/<model>_<idx>.json + .csv ; data/taskcache_spider/<iid>/task.yaml

Subcommands:
  spider_collect.py cache                      # task cache with the full task prompt of every instance
  spider_collect.py batch <ids_csv|@file|all> <K> <model> [workers]
  spider_collect.py grade <model> <idx_csv>    # grade every instance with a csv; writes the result json
"""
from __future__ import annotations
import csv, json, os, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import door_probe as DP

ROOT = Path(__file__).resolve().parent.parent
SP = Path(os.environ.get("SPIDER2_LITE_DIR", ROOT / "third_party" / "Spider2" / "spider2-lite"))
DBDIR = SP / "resource" / "databases" / "spider2-localdb"
DOCS = SP / "resource" / "documents"
EVALS = SP / "evaluation_suite"
RUNS = Path(os.environ.get("SPIDER_OUT") or (ROOT / "data" / "runs_spider"))
CACHE = ROOT / "data" / "taskcache_spider"
MAX_CALLS = int(os.environ.get("SPIDER_MAX_CALLS", "18"))
SCHEMA_CAP = 9000
DOC_CAP = 5000
_LOCK = threading.Lock()

def _api_key(base):
    envmap = {"deepseek": "DEEPSEEK_API_KEY", "anthropic.com": "ANTHROPIC_API_KEY", "openai.com": "OPENAI_API_KEY"}
    for host, keyname in envmap.items():
        if host in base:
            if os.environ.get(keyname):
                return os.environ[keyname]
            if (ROOT / ".env").exists():
                for l in open(ROOT / ".env"):
                    if l.startswith(keyname + "="):
                        return l.split("=",1)[1].strip().strip('"').strip("'")
            return ""
    return os.environ.get("LOCAL_API_KEY", "oauth")


MODELS = {  # same keys and endpoints as stitch_validate.MODELS
    "gpt56luna": (os.environ.get("GPT_BASE_URL", "https://api.openai.com/v1"), "gpt-5.6-luna"),
    "claude-sonnet5": (os.environ.get("CLAUDE_SONNET5_BASE", os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1")), "claude-sonnet-5"),
    "claude-sonnet5b": (os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1"), "claude-sonnet-5"),
    "dsv4flash": ("https://api.deepseek.com/v1", "deepseek-v4-flash"),
}

SYS = ("You are an expert SQL analyst working against a SQLite database. "
       "Explore with the execute_sql tool (results are truncated), verify your logic, then call "
       "submit_sql exactly once with the FINAL complete SELECT that answers the question. "
       "The full result of submit_sql is your answer -- column choice and row set must match the "
       "question precisely. SQLite dialect only.")

TOOLS = [
    {"type": "function", "function": {
        "name": "execute_sql",
        "description": "Run a SQL query against the database; returns truncated preview of results.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}},
                       "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "submit_sql",
        "description": "Submit the final SELECT. Its full execution result is graded.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}},
                       "required": ["sql"]}}},
]


def universe():
    rows = [json.loads(l) for l in open(SP / "spider2-lite.jsonl")]
    loc = [r for r in rows if r["instance_id"].startswith("local")]
    return sorted(loc, key=lambda r: r["instance_id"])


def gradeable_ids() -> list[str]:
    """Instances with at least one official gold exec_result file (the only gradeable ones)."""
    have = {p.name.split("_")[0].split(".")[0] for p in (EVALS / "gold" / "exec_result").glob("local*.csv")}
    return [r["instance_id"] for r in universe() if r["instance_id"] in have]


def _conn(db: str) -> sqlite3.Connection:
    """Read-only connection with a wall-clock limit (SQL_CAP seconds), also for schema sampling, since some views
    take very long to read."""
    p = DBDIR / f"{db}.sqlite"
    if not p.exists():
        raise FileNotFoundError(p)
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=20)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    deadline = time.time() + SQL_CAP
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
    return con


_SCHEMA_MEMO: dict[str, str] = {}


def schema_text(db: str) -> str:
    with _LOCK:
        if db in _SCHEMA_MEMO:
            return _SCHEMA_MEMO[db]
    con = _conn(db)
    parts = []
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    for t in tables:
        try:
            cols = con.execute(f'PRAGMA table_info("{t}")').fetchall()
            head = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            parts.append(f"TABLE {t} ({head})")
            rows = con.execute(f'SELECT * FROM "{t}" LIMIT 2').fetchall()
            for r in rows:
                cells = [str(x)[:40] for x in r]
                parts.append("  e.g. " + " | ".join(cells)[:300])
        except Exception as e:
            parts.append(f"TABLE {t} (schema error: {str(e)[:60]})")
    con.close()
    out = "\n".join(parts)
    if len(out) > SCHEMA_CAP:
        out = out[:SCHEMA_CAP] + f"\n...[schema truncated; {len(tables)} tables total]"
    with _LOCK:
        _SCHEMA_MEMO[db] = out
    return out


SQL_CAP = int(os.environ.get("SPIDER_SQL_CAP", "120"))


def run_sql(db: str, sql: str, preview: bool):
    """Runs a query with a wall-clock limit (SQL_CAP seconds); a query that runs longer is interrupted and the error
    is returned to the agent, the same for every arm."""
    con = _conn(db)
    deadline = time.time() + SQL_CAP
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(30) if preview else cur.fetchmany(200000)
        if preview:
            more = cur.fetchone() is not None
            body = "\n".join(" | ".join(str(x)[:200] for x in r)[:400] for r in rows)
            txt = f"columns: {cols}\n{body}"
            if more:
                txt += "\n...[more rows]"
            return True, txt[:2500], cols, rows
        return True, "", cols, rows
    except Exception as e:
        return False, f"SQL ERROR: {str(e)[:300]}", [], []
    finally:
        con.close()


def task_prompt(row: dict, guide: str = "") -> str:
    doc = ""
    if row.get("external_knowledge"):
        try:
            doc = (DOCS / row["external_knowledge"]).read_text()[:DOC_CAP]
        except Exception:
            doc = ""
    p = f"Database: {row['db']} (SQLite)\n\nSCHEMA:\n{schema_text(row['db'])}\n"
    if doc:
        p += f"\nEXTERNAL KNOWLEDGE ({row['external_knowledge']}):\n{doc}\n"
    p += f"\nQUESTION:\n{row['question']}\n"
    if guide:
        p = f"{guide}\n\n{p}"
    return p


_SIDE = ThreadPoolExecutor(max_workers=32)  # model calls run here under a wall-clock cap
CALL_CAP = int(os.environ.get("SPIDER_CALL_CAP", "900"))

import re as _re
import threading as _threading


class InfraError(RuntimeError):
    """The model could not be reached for a whole rollout (7 attempts). The rollout is not written (a later batch
    retries it) instead of being counted as a failure."""


class FatalInfraError(InfraError):
    """Account-level failure (402 Insufficient Balance, 401/403 auth): abort the whole batch at once."""


_FATAL_RE = _re.compile(r"Error code: 40[123]\b|Insufficient Balance|insufficient_quota|quota_exhausted|invalid_api_key|authentication_error|account_deactivated", _re.I)


def _call_llm(client, model, msgs):
    """Wall-clock cap per attempt (CALL_CAP) and up to 7 attempts with backoff; a call over the cap is abandoned
    and retried."""
    from concurrent.futures import TimeoutError as FutTimeout
    last = None
    for i in range(7):
        fut = _SIDE.submit(client.chat.completions.create, model=model, messages=msgs, tools=TOOLS,
                           **DP.thinking_off_kw(model))
        try:
            return fut.result(timeout=CALL_CAP)
        except FutTimeout:
            fut.cancel()
            last = f"call-cap {CALL_CAP}s exceeded"
        except Exception as e:
            last = str(e)[:150]
            if _FATAL_RE.search(str(e)):
                print(f"LLM FATAL: {str(e)[:160]}", flush=True)
                raise FatalInfraError(f"{model}: {str(e)[:160]}")
        time.sleep(min(2 ** i, 45))
    raise InfraError(f"{model}: LLM unreachable after 7 attempts: {last}")


def single(iid: str, model_key: str, idx: int, out_root: Path, guide: str = "") -> dict:
    from openai import OpenAI
    row = next(r for r in universe() if r["instance_id"] == iid)
    base, model = MODELS[model_key]
    client = OpenAI(base_url=base, api_key=_api_key(base), timeout=600)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": task_prompt(row, guide)}]
    rec = {"task": iid, "model": model_key, "idx": idx, "calls": [], "final_sql": None,
           "submitted": False, "usage": []}
    final_rows, final_cols = [], []
    for step in range(MAX_CALLS):
        try:
            resp = _call_llm(client, model, msgs)
        except InfraError:
            raise                                   # never written: the rollout did not happen
        except Exception as e:
            rec["error"] = str(e)[:200]
            break
        try:
            u = resp.usage
            rec["usage"].append({"in": u.prompt_tokens, "out": u.completion_tokens})
        except Exception:
            pass
        m = resp.choices[0].message
        tcs = m.tool_calls or []
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": t.id, "type": "function",
                                     "function": {"name": t.function.name,
                                                  "arguments": t.function.arguments}} for t in tcs]})
        if not tcs:
            break
        stop = False
        for t in tcs:
            try:
                sql = json.loads(t.function.arguments).get("sql", "")
            except Exception:
                sql = ""
            if t.function.name == "submit_sql":
                ok, err, cols, rows = run_sql(row["db"], sql, preview=False)
                if ok:
                    rec["final_sql"], rec["submitted"] = sql, True
                    final_cols, final_rows = cols, rows
                    msgs.append({"role": "tool", "tool_call_id": t.id,
                                 "content": f"submitted ({len(rows)} rows)"})
                    stop = True
                else:
                    rec["calls"].append({"submit_err": err[:200]})
                    msgs.append({"role": "tool", "tool_call_id": t.id, "content": err})
            else:
                ok, txt, _, _ = run_sql(row["db"], sql, preview=True)
                rec["calls"].append({"sql": sql[:400], "ok": ok})
                msgs.append({"role": "tool", "tool_call_id": t.id, "content": txt})
        if stop:
            break
    out_dir = out_root / iid
    out_dir.mkdir(parents=True, exist_ok=True)
    if rec["submitted"]:
        with open(out_dir / f"{model_key}_{idx}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(final_cols)
            w.writerows(final_rows)
    rec["n_steps"] = len(rec["calls"])
    json.dump(rec, open(out_dir / f"{model_key}_{idx}.json", "w"), indent=1)
    return rec


# ---------------- grading (official comparison functions) ----------------

def grade_inline(iids: list[str], model_key: str, idx: int, out_root: Path) -> dict:
    """Per-id verdicts via the official compare functions (evaluate.py exec_result semantics)."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    import spider_eval_bridge as B
    cfg = {}
    for l in open(EVALS / "gold" / "spider2lite_eval.jsonl"):
        d = json.loads(l)
        cfg[d["instance_id"]] = d
    res = {}
    for iid in iids:
        pred_p = out_root / iid / f"{model_key}_{idx}.csv"
        if not pred_p.exists():
            res[iid] = 0
            continue
        golds = sorted((EVALS / "gold" / "exec_result").glob(f"{iid}*.csv"))
        golds = [g for g in golds if g.stem == iid or g.stem.startswith(iid + "_")]
        if not golds:
            res[iid] = -1  # no gold -> cannot grade
            continue
        try:
            pred = pd.read_csv(pred_p)
            gold_dfs = [pd.read_csv(g) for g in golds]
            c = cfg.get(iid, {})
            res[iid] = B.compare_multi(pred, gold_dfs, c.get("condition_cols", []),
                                       bool(c.get("ignore_order", False)))
        except Exception as e:
            res[iid] = 0
            print(f"{iid}: grade error {str(e)[:100]}", flush=True)
    return res


# ---------------- subcommands ----------------

def cmd_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    import yaml
    man = []
    for r in universe():
        d = CACHE / r["instance_id"]
        d.mkdir(exist_ok=True)
        yaml.safe_dump({"instruction": task_prompt(r), "db": r["db"],
                        "question": r["question"]}, open(d / "task.yaml", "w"),
                       allow_unicode=True, width=10000)
        man.append({"task": r["instance_id"], "db": r["db"]})
    json.dump(man, open(CACHE / "manifest.json", "w"), indent=1)
    print(f"cache: {len(man)} tasks")


def cmd_batch(ids_arg: str, K: int, model_key: str, workers: int):
    RUNS.mkdir(parents=True, exist_ok=True)
    # One batch per output root: a second batch on the same SPIDER_OUT would rerun unfinished tasks and could pair
    # one run's .csv with the other's .json, so it waits up to 30 s for the lock and then exits.
    import fcntl
    _lock = open(RUNS / ".batch.lock", "w")
    for _i in range(30):
        try:
            fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            time.sleep(1)
    else:
        print(f"BATCH-REFUSED: another spider_collect batch already holds {RUNS}/.batch.lock", flush=True)
        return
    if ids_arg == "all":
        ids = [r["instance_id"] for r in universe()]
    elif ids_arg.startswith("@"):
        ids = [l.strip() for l in open(ids_arg[1:]) if l.strip()]
    else:
        ids = ids_arg.split(",")
    guides = {}
    gf = os.environ.get("SPIDER_GUIDE_FILE")
    if gf:
        guides = json.load(open(gf))
    jobs = []
    for iid in ids:
        for k in range(K):
            if (RUNS / iid / f"{model_key}_{k}.json").exists():
                continue
            jobs.append((iid, k))
    print(f"batch: {len(jobs)} rollouts ({len(ids)} ids x K={K}, skip-existing), workers={workers}",
          flush=True)

    stop = _threading.Event()
    infra_streak = [0]
    streak_lock = _threading.Lock()

    def work(job):
        if stop.is_set():
            return
        iid, k = job
        t0 = time.time()
        try:
            g = guides.get(iid) or ""
            if isinstance(g, dict):
                g = g.get("guides", {}).get(os.environ.get("SPIDER_GUIDE_ARM", "psyn"), "")
            rec = single(iid, model_key, k, RUNS, guide=g)
            with streak_lock:
                infra_streak[0] = 0
            print(f"{iid}#{k}: sub={rec['submitted']} steps={rec['n_steps']} {time.time()-t0:.0f}s",
                  flush=True)
        except FatalInfraError as e:
            stop.set()
            print(f"BATCH-ABORT-FATAL: {str(e)[:200]} -- nothing recorded; relaunch after fixing the account", flush=True)
        except InfraError as e:
            with streak_lock:
                infra_streak[0] += 1
                n = infra_streak[0]
            print(f"{iid}#{k}: SKIPPED (infra, not recorded): {str(e)[:150]}", flush=True)
            if n >= 3:
                stop.set()
                print("BATCH-ABORT-INFRA: 3 consecutive LLM outages -- stopping; relaunch to retry", flush=True)
        except Exception as e:
            print(f"{iid}#{k}: FAILED {str(e)[:120]}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))
    print("BATCH-DONE" if not stop.is_set() else "BATCH-ABORTED", flush=True)


def cmd_grade(model_key: str, idxs: str):
    ids = [r["instance_id"] for r in universe()]
    out = {}
    for idx in [int(x) for x in idxs.split(",")]:
        v = grade_inline(ids, model_key, idx, RUNS)
        got = {k: x for k, x in v.items() if (RUNS / k / f"{model_key}_{idx}.json").exists()}
        out[idx] = got
        n = len(got)
        s = sum(1 for x in got.values() if x == 1)
        print(f"idx={idx}: {s}/{n} solved" + (f" ({s/n:.0%})" if n else ""), flush=True)
    tag = os.environ.get("SPIDER_GRADE_TAG") or (RUNS.name if RUNS.name != "runs_spider" else "pool")
    p = ROOT / "result" / f"spider_grade_{tag}_{model_key}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        cmd_cache()
    elif cmd == "batch":
        cmd_batch(sys.argv[2], int(sys.argv[3]), sys.argv[4],
                  int(sys.argv[5]) if len(sys.argv) > 5 else 8)
    elif cmd == "grade":
        cmd_grade(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("unknown subcommand")
