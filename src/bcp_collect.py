"""BrowseComp-Plus collector and grader.

Environment: the fixed corpus of 100,195 documents (Tevatron/browsecomp-plus-corpus) in data/bcp/corpus.db; no live
web. The agent has three tools: search(query) -> top-10 (docid, title, snippet) by dense retrieval
(Qwen3-Embedding-8B query embedding against data/bcp/dense); open_doc(docid, offset) -> a 6,000-character window;
submit_answer(answer) -> records the final short answer and ends the rollout. Budget: BCP_MAX_STEPS steps (40).
Guidance (an arm's guide) is appended to the question.

Output: the submitted answer; final_files = {"__answer__": sha16(normalized answer)}, and artifacts/<sha> holds the
answer with the opened docids.

Grading: normalized exact match against the gold answer; on a mismatch, the LLM equivalence judge (BCP_JUDGE,
DeepSeek-V4-Flash with thinking disabled). ok_grade.msg records which path decided ('exact' / 'judge=...').

Output layout: data/<BCP_OUT>/bcp-<qid>/<model>_<idx>.json + artifacts/.

Subcommands:
  python3 bcp_collect.py cache                 build data/taskcache_bcp/<task>/task.yaml (830)
  python3 bcp_collect.py regrade <runs_dir>    re-score records whose judge call failed
  python3 bcp_collect.py <qids_csv|@file> <K> <models_csv> [workers]
"""
from __future__ import annotations
import hashlib, json, os, re, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
import stitch_validate as SV
import door_probe as DP

ROOT = Path(__file__).resolve().parent.parent
BCP = ROOT / "data" / "bcp"
QUERIES = {r["query_id"]: r for r in (json.loads(l) for l in open(BCP / "queries.jsonl"))}
RUNS = Path(os.environ.get("BCP_OUT") or (ROOT / "data" / "runs_bcp"))
MAX_STEPS = int(os.environ.get("BCP_MAX_STEPS", "40"))
_SUBMIT_NUDGE = os.environ.get("BCP_SUBMIT_NUDGE", "1") == "1"
JUDGE_KEY = os.environ.get("BCP_JUDGE", "dsv4flash")


class InfraError(Exception):
    """The student model never answered (API unavailable after retries): the rollout is not recorded, so a
    relaunch redoes it."""


_FATAL_RE = re.compile(r"Error code: 40[123]\b|Insufficient Balance|insufficient_quota|quota_exhausted|invalid_api_key"
                       r"|authentication_error|account_deactivated", re.I)


def _judge_kw(model: str) -> dict:
    """DeepSeek and Claude judges would spend the short YES/NO budget on reasoning, so thinking is disabled."""
    return {"extra_body": {"thinking": {"type": "disabled"}}} if ("deepseek" in model or model.startswith("claude")) else {}
_LOCK = threading.Lock()
_TLS = threading.local()

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "BM25 keyword search over the fixed 100K-document web corpus. Returns top-10 "
                       "(docid, title, snippet). Bag-of-words matching — use several DISTINCT keyword "
                       "combinations across calls rather than one long sentence.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "open_doc",
        "description": "Open a document by docid; returns a 6000-character window starting at offset.",
        "parameters": {"type": "object", "properties": {"docid": {"type": "string"},
                                                        "offset": {"type": "integer", "default": 0}},
                       "required": ["docid"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final answer (short and exact: a name/date/title, not a sentence). "
                       "This ends the task.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"]}}},
]

BCP_SYS = """You are a deep-research agent answering a hard multi-constraint question using ONLY a fixed
local corpus of ~100K web documents, via the search / open_doc tools.

Method:
- Decompose the question into its constraints. Issue MULTIPLE searches with different keyword angles
  (entities, dates, quoted phrases, synonyms); BM25 is bag-of-words — short keyword queries beat sentences.
- open_doc promising hits and VERIFY every constraint in the document text before trusting a candidate.
- If a search dead-ends, pivot to other constraints rather than repeating similar keywords.
- You MUST finish by calling submit_answer with the short exact answer (name/title/date only, no sentence),
  then reply without any tool call summarizing the evidence.
"""


def _sha16(s) -> str:
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()[:16]


def _db():
    if getattr(_TLS, "db", None) is None:
        _TLS.db = sqlite3.connect(f"file:{BCP/'corpus.db'}?mode=ro", uri=True, check_same_thread=False)
    return _TLS.db


# ---- dense retrieval ----
# The query is embedded by Qwen3-Embedding-8B, served at BCP_EMB_URL (e.g. a local vLLM), and scored against the
# precomputed document embeddings in BCP_DENSE (default data/bcp/dense: corpus_emb.npy + docids.json). The result
# format is: [docid] title, then the first 300 characters of the document.
_DENSE_DIR = os.environ.get("BCP_DENSE", str(BCP / "dense"))
import numpy as _np
_D_IDS = json.load(open(f"{_DENSE_DIR}/docids.json"))
_D_TORCH = _D_EMB = None
_D_LOCK = threading.Lock()


def _dense_matrix():
    """Load the document matrix on first use. Scores are float32 on both paths. BCP_DENSE_GPU=<index> keeps the
    matrix on that GPU (a CPU matrix-vector product over the 1.6GB matrix is slow with many workers); unset = numpy
    on CPU."""
    global _D_TORCH, _D_EMB, _torch
    with _D_LOCK:
        if _D_TORCH is None and _D_EMB is None:
            if os.environ.get("BCP_DENSE_GPU"):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _D_TORCH = _torch.from_numpy(_np.load(f"{_DENSE_DIR}/corpus_emb.npy").astype(_np.float32)).to(
                            "cuda:" + os.environ["BCP_DENSE_GPU"])
                except Exception as _e:
                    print(f"[dense] torch path unavailable ({_e}); falling back to numpy", flush=True)
            if _D_TORCH is None:
                _D_EMB = _np.load(f"{_DENSE_DIR}/corpus_emb.npy").astype(_np.float32)   # [N,D] L2-normed
    return _D_TORCH, _D_EMB


BCP_EMB_URL_S = os.environ.get("BCP_EMB_URL", "http://127.0.0.1:8910/v1")
_D_CLIENT = OpenAI(base_url=BCP_EMB_URL_S, api_key="EMPTY", timeout=120)
_D_INSTR = ("Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            "Query: ")


def tool_search(q: str) -> str:
    err = None
    for att in range(3):
        try:
            v = _D_CLIENT.embeddings.create(model="Qwen/Qwen3-Embedding-8B",
                                            input=[_D_INSTR + (q or "")[:2000]]).data[0].embedding
            break
        except Exception as e:
            err = e
            time.sleep(3 * (att + 1))
    else:
        # An unreachable embedding server is not an empty search result: after six attempts the rollout raises
        # InfraError, so nothing is written and a relaunch redoes it once retrieval is back.
        for att in range(3, 6):
            try:
                v = _D_CLIENT.embeddings.create(model="Qwen/Qwen3-Embedding-8B",
                                                input=[_D_INSTR + (q or "")[:2000]]).data[0].embedding
                break
            except Exception as e:
                err = e
                time.sleep(5 * (att - 1))
        else:
            raise InfraError(f"dense retrieval down after 6 tries ({BCP_EMB_URL_S}): {str(err)[:120]}")
    v = _np.asarray(v, _np.float32)
    v /= (_np.linalg.norm(v) + 1e-9)
    mat_t, mat = _dense_matrix()
    if mat_t is not None:
        with _torch.no_grad():
            qv = _torch.from_numpy(v).to(mat_t.device, dtype=mat_t.dtype)
            top = _torch.topk(mat_t @ qv, 10).indices.cpu().tolist()
    else:
        sims = mat @ v
        top = _np.argpartition(-sims, 10)[:10]
        top = top[_np.argsort(-sims[top])]
    out = []
    for i in top:
        docid = _D_IDS[int(i)]
        rid = _rowid_of(docid)
        row = _db().execute("SELECT substr(text,1,420) FROM docs WHERE rowid = ?",
                            (rid,)).fetchone() if rid is not None else None
        head = row[0] if row else ""
        m = re.search(r"title:\s*(.+)", head)
        title = (m.group(1).strip() if m else head.replace("\n", " "))[:90]
        body = re.sub(r"\s+", " ", head)[:300]
        out.append(f"[{docid}] {title}\n    {body}")
    return "\n".join(out)


# docid is UNINDEXED in the FTS5 table, so `WHERE docid = ?` would scan the whole corpus; point lookups go
# through a docid -> rowid map instead (rowid is the table's key), saved next to the corpus after the first build.
_DOC_MAP = {}


def _rowid_of(docid):
    global _DOC_MAP
    if not _DOC_MAP:
        with _LOCK:
            if not _DOC_MAP:
                mp = BCP / "docid2row.json"
                if mp.exists():
                    _DOC_MAP = {k: int(v) for k, v in json.load(open(mp)).items()}
                else:
                    _DOC_MAP = {d: int(r) for r, d in _db().execute("SELECT rowid, docid FROM docs")}
                    try:
                        json.dump(_DOC_MAP, open(mp, "w"))
                    except OSError:
                        pass
    return _DOC_MAP.get(str(docid))


def tool_open(docid: str, offset: int = 0) -> str:
    rid = _rowid_of(docid)
    row = (_db().execute("SELECT text FROM docs WHERE rowid = ?", (rid,)).fetchone()
           if rid is not None else
           _db().execute("SELECT text FROM docs WHERE docid = ?", (str(docid),)).fetchone())
    if not row:
        return f"(no such docid: {docid})"
    t = row[0]
    w = t[offset:offset + 6000]
    return (f"[doc {docid} chars {offset}-{offset+len(w)} of {len(t)}]\n" + w +
            ("\n[... truncated; call open_doc with offset=%d for more]" % (offset + 6000)
             if offset + 6000 < len(t) else "\n[end of document]"))


_NORM_DROP = re.compile(r"\b(the|a|an|of)\b|[^a-z0-9 ]")


def norm_answer(s: str) -> str:
    return re.sub(r"\s+", " ", _NORM_DROP.sub(" ", (s or "").lower())).strip()


def grade_answer(qid: str, predicted: str) -> dict:
    gold = QUERIES[qid]["answer"]
    if not (predicted or "").strip():
        return {"ok": False, "msg": "no answer submitted"}
    if norm_answer(predicted) == norm_answer(gold):
        return {"ok": True, "msg": "exact"}
    base_url, model = SV.MODELS[JUDGE_KEY]
    client = OpenAI(base_url=base_url, api_key=SV.key_for(base_url), timeout=120)
    prompt = (f"Question: {QUERIES[qid]['query']}\n\nGold answer: {gold}\nPredicted answer: {predicted}\n\n"
              f"Does the predicted answer refer to the same entity/value as the gold answer? "
              f"Ignore formatting, casing, extra words. Reply with exactly one word: YES or NO.")
    err = ""
    for attempt in range(6):
        try:
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=8,
                                               messages=[{"role": "user", "content": prompt}], **_judge_kw(model))
            v = (r.choices[0].message.content or "").strip().upper()
            if not v.startswith(("YES", "NO")):
                raise RuntimeError(f"unparseable judge reply {v[:20]!r}")
            return {"ok": v.startswith("YES"), "msg": f"judge={v[:8]}", "grader": JUDGE_KEY}
        except Exception as e:
            err = str(e)[:160]
            if _FATAL_RE.search(err):
                break
            time.sleep(min(3 * 2 ** attempt, 60) if "429" in err else 2 * (attempt + 1))
    # An unavailable judge is not a wrong answer: ok=None + grade_error marks the record as ungraded, and
    # `bcp_collect.py regrade <runs_dir>` re-scores it from the stored answer.
    return {"ok": None, "msg": f"judge unavailable: {err}", "error": f"judge_unavailable: {err}", "grader": JUDGE_KEY}


def rollout(qid: str, model_key: str, idx: int, guidance: str | None = None,
            out_root: Path | None = None, name_tag: str = ""):
    task = f"bcp-{qid}"
    base_url, model = SV.MODELS[model_key]
    root = Path(out_root or RUNS)
    tdir = root / task
    (tdir / "artifacts").mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=base_url, api_key=SV.key_for(base_url), timeout=600)
    user = QUERIES[qid]["query"] + ("\n\n" + guidance if guidance else "")
    messages = [{"role": "system", "content": BCP_SYS}, {"role": "user", "content": user}]
    steps, answer, opened, usage = [], None, [], []
    t0 = time.time()
    nudges = 2                                            # submit reminders left (see below)
    for _ in range(MAX_STEPS):
        resp, err = None, ""
        for attempt in range(7):
            try:
                resp = client.chat.completions.create(model=model, messages=messages,
                                                      tools=TOOLS, max_tokens=4000, **DP.thinking_off_kw(model))
                break
            except Exception as e:
                err = str(e)[:140]
                if _FATAL_RE.search(err):
                    break
                time.sleep(min(3 * 2 ** attempt, 60) if "429" in err else 2 * (attempt + 1))
        if resp is None:
            raise InfraError(f"{task} step {len(steps)}: {err}")
        try:
            _u = {"in": int(resp.usage.prompt_tokens or 0), "out": int(resp.usage.completion_tokens or 0)}
            # Claude endpoints report prompt_tokens without cached input; cache reads and writes are recorded when
            # present, so the full input can be reconstructed.
            for _k, _n in (("cache_read_input_tokens", "cache_read"), ("cache_creation_input_tokens", "cache_write")):
                _v = (getattr(resp.usage, "model_extra", None) or {}).get(_k)
                if isinstance(_v, (int, float)):
                    _u[_n] = int(_v)
            usage.append(_u)
        except Exception:
            pass
        m = resp.choices[0].message
        if not m.tool_calls:
            steps.append({"think": (m.content or "")[:4000], "cmd": "", "obs": ""})
            # Submit reminder: a reply without a tool call before any answer was submitted gets up to two reminders
            # to call submit_answer (BCP_SUBMIT_NUDGE=1, the default); otherwise the rollout ends.
            if _SUBMIT_NUDGE and answer is None and nudges > 0:
                nudges -= 1
                messages.append({"role": "assistant", "content": m.content or ""})
                messages.append({"role": "user", "content":
                                 "If you have determined the final answer, submit it NOW by calling "
                                 "the submit_answer tool with the answer string. If not, continue "
                                 "searching with the tools. Do not reply in plain text."})
                continue
            break
        messages.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in m.tool_calls]})
        stop = False
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            fn = tc.function.name
            if fn == "search":
                cmd = f"search({args.get('query','')!r})"
                obs = tool_search(args.get("query", ""))
            elif fn == "open_doc":
                cmd = f"open_doc({args.get('docid')!r}, offset={args.get('offset', 0)})"
                obs = tool_open(str(args.get("docid", "")), int(args.get("offset") or 0))
                opened.append(str(args.get("docid", "")))
            elif fn == "submit_answer":
                answer = str(args.get("answer", ""))
                cmd, obs, stop = f"submit_answer({answer!r})", "Answer recorded.", True
            else:
                cmd, obs = f"{fn}({str(args)[:100]})", "(unknown tool)"
            steps.append({"think": (m.content or "")[:4000], "cmd": cmd[:4000], "obs": obs[:6000]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs[:8000]})
        if stop:
            break
    grade = grade_answer(qid, answer or "")
    final_sha = {}
    if answer is not None:
        payload = json.dumps({"answer": answer, "opened_docids": opened[:40]}, ensure_ascii=False, indent=1)
        s = _sha16(norm_answer(answer))
        final_sha["__answer__"] = s
        (tdir / "artifacts" / s).write_text(payload)
    rid = f"{model_key}_{idx}"
    rec = {"task": task, "model": model_key, "idx": idx, "steps": steps,
           "final_files": final_sha, "baseline": {}, "ok_grade": grade,
           "collector": "bcp_collect",
           "served_model": model, "secs": round(time.time() - t0, 1), "usage": usage,
           "grader": JUDGE_KEY, "guided": bool(guidance),
           "protocol": {"max_steps": MAX_STEPS, "submit_nudge": _SUBMIT_NUDGE, "dense": bool(_DENSE_DIR), "grader": JUDGE_KEY}}
    if grade.get("ok") is None:
        rec["grade_error"] = grade.get("error") or "judge unavailable"
    json.dump(rec, open(tdir / f"{rid}.json", "w"), indent=1)
    return task, rid, len(steps), answer is not None, grade["ok"]


def build_taskcache():
    import yaml
    out = ROOT / "data" / "taskcache_bcp"
    for qid, r in QUERIES.items():
        d = out / f"bcp-{qid}"
        d.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump({"instruction": r["query"], "query_id": qid,
                        "n_gold_docs": len(r["gold_docids"]), "n_evidence_docs": len(r["evidence_docids"])},
                       open(d / "task.yaml", "w"), allow_unicode=True, sort_keys=False)
    print(f"[cache] wrote {len(QUERIES)} task.yaml under {out}")


def regrade(runs_dir: str, only_key: str | None = None):
    """Re-score every record whose grade is missing (ok=None / grade_error) from its stored answer; no student calls."""
    root = Path(runs_dir)
    n = fixed = still = 0
    for f in sorted(root.glob("bcp-*/*.json")):
        d = json.load(open(f))
        if only_key and d.get("model") != only_key:
            continue
        if (d.get("ok_grade") or {}).get("ok") is not None:
            continue
        n += 1
        qid = d["task"][4:]
        sha = (d.get("final_files") or {}).get("__answer__")
        answer = ""
        if sha and (f.parent / "artifacts" / sha).exists():
            try:
                answer = json.load(open(f.parent / "artifacts" / sha)).get("answer", "")
            except Exception:
                answer = ""
        g = grade_answer(qid, answer)
        if g.get("ok") is None:
            still += 1
            continue
        d["ok_grade"] = g; d.pop("grade_error", None); d["grader"] = JUDGE_KEY
        json.dump(d, open(f, "w"), indent=1)
        fixed += 1
    print(f"[regrade] {runs_dir}: {n} ungraded records, re-scored {fixed}, still ungraded {still}", flush=True)


def main():
    if sys.argv[1] == "cache":
        build_taskcache(); return
    if sys.argv[1] == "regrade":
        regrade(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None); return
    arg = sys.argv[1]
    qids = [l.strip() for l in open(arg[1:])] if arg.startswith("@") else arg.split(",")
    qids = [q[4:] if q.startswith("bcp-") else q for q in qids if q]   # task lists name queries bcp-<query_id>
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    models = sys.argv[3].split(",")
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    for q in qids:
        assert q in QUERIES, f"unknown query_id {q}"
    jobs = []
    for q in qids:
        for mk in models:
            tdir = RUNS / f"bcp-{q}"
            existing = {int(p.stem.split("_")[-1]) for p in tdir.glob(f"{mk}_*.json")} if tdir.exists() else set()
            jobs += [(q, mk, i) for i in range(K) if i not in existing]
    print(f"[*] bcp_collect: {len(jobs)} rollouts ({len(qids)} queries x {models} x K={K}, skip-existing); "
          f"workers={workers}; MAX_STEPS={MAX_STEPS}", flush=True)

    guides, garm = {}, os.environ.get("BCP_GUIDE_ARM", "psyn")
    if os.environ.get("BCP_GUIDE_FILE"):
        guides = json.load(open(os.environ["BCP_GUIDE_FILE"]))
        print(f"    guidance: {os.environ['BCP_GUIDE_FILE']} arm={garm} ({sum(1 for t in guides if (guides[t].get('guides') or {}).get(garm))} tasks)", flush=True)
    stop = threading.Event()
    streak = [0]

    def work(j):
        q, mk, i = j
        if stop.is_set():
            return None
        g = ((guides.get(f"bcp-{q}") or {}).get("guides") or {}).get(garm, "") if guides else ""
        if guides and not g:
            print(f"  SKIP bcp-{q}/{mk}_{i}: no {garm} guidance", flush=True)
            return None
        try:
            r = rollout(q, mk, i, guidance=g or None)
            streak[0] = 0
            return r
        except InfraError as e:
            msg = str(e)
            print(f"  INFRA bcp-{q}/{mk}_{i}: NOT RECORDED: {msg[:160]}", flush=True)
            if _FATAL_RE.search(msg):
                print("BATCH-ABORT-FATAL: auth/quota/balance error from the student endpoint -> stopping the batch", flush=True)
                stop.set()
            else:
                streak[0] += 1
                if streak[0] >= 3:
                    print("BATCH-ABORT-INFRA: 3 consecutive infra errors -> stopping the batch", flush=True)
                    stop.set()
            return None
        except Exception as e:
            print(f"  ERROR bcp-{q}/{mk}_{i}: {str(e)[:160]}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(work, jobs):
            if res:
                task, rid, n, deliv, ok = res
                print(f"  done {task}/{rid}: steps={n} deliv={deliv} ok={ok}", flush=True)
    print("BATCH-DONE" if not stop.is_set() else "BATCH-STOPPED", flush=True)


if __name__ == "__main__":
    main()
