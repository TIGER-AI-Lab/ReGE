"""Terminal-Bench 2.0 collector and grader (the official prebaked task images and the official verifier).

- Environment: the docker image named in each task.toml ([environment].docker_image), with that task's CPU and
  memory limits. The agent works in /app with the shell agent of tb_collect (REASON_SYS, one bash tool); after
  every command the /app files are hashed and changed files are copied to the run's artifact store.
- Budget: TB2_MAX_STEPS model turns (30) and TB2_STUDENT_MAX_TOKENS output tokens per call (12,000), within the
  agent timeout of task.toml. A turn without a tool call that was cut at the output limit is asked again once with
  twice the limit.
- Grading: the official tests/test.sh inside the same container (reward.txt 1|0), with the task's verifier timeout.
- Guidance (an arm's guide) is placed before the instruction.

Layout: data/<TB2_OUT>/<task>/<model>_<idx>.json + artifacts/ ; data/taskcache_tb2/<task>/task.yaml
Subcommands:
  tb2_collect.py cache
  tb2_collect.py batch <all|csv|@file> <K> <model> [workers]
"""
from __future__ import annotations
import hashlib, json, os, sys, threading, time, tomllib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import tb_collect as T1
import door_probe as DP

DS = ROOT / "data" / "tb2_dataset" / "terminal-bench"
RUNS = Path(os.environ.get("TB2_OUT") or (ROOT / "data" / "runs_tb2"))
CACHE = ROOT / "data" / "taskcache_tb2"
CALL_CAP = int(os.environ.get("TB2_CALL_CAP", "480"))        # wall-clock cap per model call (seconds)


class InfraError(RuntimeError):
    """The model could not be reached for a whole rollout (7 attempts). Such a rollout is not written (a later
    relaunch retries it) instead of being booked as a task failure; three in a row stop the batch. Account-level
    failures (402/401/403) are T1.FatalInfraError and stop it at once."""


_SIDE = ThreadPoolExecutor(max_workers=64)
_PULL_LOCK = threading.Lock()
_PULLED: set[str] = set()


def universe() -> list[str]:
    return sorted(d.name for d in DS.iterdir() if (d / "task.toml").exists())


def meta(task: str) -> dict:
    m = tomllib.loads((DS / task / "task.toml").read_text())
    env = m["environment"]
    return {"image": env["docker_image"],
            "agent_s": int(m.get("agent", {}).get("timeout_sec", 900)),
            "verify_s": int(m.get("verifier", {}).get("timeout_sec", 900)),
            "cpus": str(env.get("cpus", 2)), "mem": str(env.get("memory", "4G"))}


def ensure_image(img: str):
    with _PULL_LOCK:
        done = img in _PULLED
    if done:
        return
    if T1.sh("docker", "image", "inspect", img).returncode != 0:
        p = T1.sh("docker", "pull", img, timeout=2400)
        if p.returncode != 0:
            raise RuntimeError("docker pull failed: " + (p.stderr or p.stdout)[-300:])
    with _PULL_LOCK:
        _PULLED.add(img)


STUDENT_MAX_TOKENS = int(os.environ.get("TB2_STUDENT_MAX_TOKENS", "12000"))
# A tool-less turn cut at the output limit is asked again once at twice the limit (TB2_BURN_REASK=0 ends the episode).
BURN_REASK = os.environ.get("TB2_BURN_REASK", "1") not in ("0", "false", "no")
MAX_STEPS = int(os.environ.get("TB2_MAX_STEPS", "30"))


def _llm(client, model, messages):
    last = ""
    budget = STUDENT_MAX_TOKENS
    extra = DP.thinking_off_kw(model)
    escalated = False
    for i in range(7):
        fut = _SIDE.submit(client.chat.completions.create, model=model, messages=messages,
                           tools=T1.BASH_TOOL, max_tokens=budget, **extra)
        try:
            resp = fut.result(timeout=CALL_CAP)
        except FutTimeout:
            fut.cancel()
            last = f"call-cap {CALL_CAP}s"
            time.sleep(min(2 ** i, 45))
            continue
        except Exception as e:
            last = str(e)[:200]
            if T1.fatal_infra(last):
                raise T1.FatalInfraError(last)
            time.sleep(min(2 ** i, 45))
            continue
        ch = resp.choices[0]
        if BURN_REASK and not escalated and ch.finish_reason == "length" and not ch.message.tool_calls:
            escalated = True
            budget = STUDENT_MAX_TOKENS * 2
            print(f"    [student budget-burn] {model}: finish=length with no tool call at "
                  f"{STUDENT_MAX_TOKENS} -> retry at {budget}", flush=True)
            continue
        return resp, ""
    raise InfraError(f"{model}: LLM unreachable after 7 attempts: {last}")


def verify(name: str, task: str, vtimeout: int):
    T1.sh("docker", "cp", str(DS / task / "tests"), f"{name}:/tests", timeout=120)
    T1.exec_in(name, "mkdir -p /logs/verifier", 30)
    T1.exec_in(name, "cd /app && bash /tests/test.sh > /logs/verifier/run.log 2>&1",
               timeout=vtimeout + 60)
    r = T1.exec_in(name, "cat /logs/verifier/reward.txt 2>/dev/null", 30).strip()
    ok = r.endswith("1")
    tail = T1.exec_in(name, "tail -c 600 /logs/verifier/run.log 2>/dev/null", 30)
    return ok, tail


def single(task: str, model_key: str, idx: int, out_root: Path | None = None, guide: str = "") -> dict:
    """One rollout. guide: arm guidance placed before the instruction."""
    out_root = out_root or RUNS
    m = meta(task)
    ensure_image(m["image"])
    # The container name includes a hash of the output root, so arms running the same (task, model, idx) at the
    # same time get separate containers.
    tag = hashlib.sha1(str(Path(out_root).resolve()).encode()).hexdigest()[:8]
    name = f"tb2_{task[:40]}_{model_key}_{idx}_{tag}"
    store = out_root / task / "artifacts"
    store.mkdir(parents=True, exist_ok=True)
    T1.sh("docker", "rm", "-f", name)
    T1.sh("docker", "run", "-d", "--name", name, "--shm-size=512m", "-w", "/app",
          "--cpus", m["cpus"], "--memory", m["mem"],   # resource limits of task.toml
          m["image"], "sleep", "infinity")
    t0 = time.time()
    try:
        baseline = T1.capture_state(name)
        steps = [{"cmd": None, "files": baseline}]
        from openai import OpenAI
        base, model = T1.MODELS[model_key]
        client = OpenAI(base_url=base, api_key=T1.SV.key_for(base), timeout=600)
        instr = (DS / task / "instruction.md").read_text()
        if guide:
            instr = f"{guide}\n\nTASK:\n{instr}"
        messages = [{"role": "system", "content": T1.REASON_SYS},
                    {"role": "user", "content": instr}]
        for _ in range(MAX_STEPS):
            if time.time() - t0 > m["agent_s"]:
                steps.append({"error": "agent wall-timeout"})
                break
            resp, err = _llm(client, model, messages)
            if resp is None:
                steps.append({"error": err})
                break
            msg = resp.choices[0].message
            tcs = msg.tool_calls or []
            u = getattr(resp, "usage", None)
            usage = ({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
                     if u is not None else None)
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": T1.tcs_echo(tcs)})
            if not tcs:
                steps.append({"cmd": None, "think": msg.content or "", "usage": usage})
                break
            first = True
            for t in tcs:
                try:
                    cmd = json.loads(t.function.arguments or "").get("command", "")
                except Exception:
                    cmd = ""
                out = T1.exec_in(name, cmd) if cmd else "(empty/malformed command)"
                messages.append({"role": "tool", "tool_call_id": t.id, "content": out[:1500]})
                hm = T1.capture_state(name)
                T1.save_artifacts(name, hm, baseline, store)
                steps.append({"cmd": cmd, "files": hm, "obs": out[:2000],
                              "think": (msg.content or "") if first else "",
                              "usage": usage if first else None})
                first = False
        final = T1.capture_state(name)
        T1.save_artifacts(name, final, baseline, store)
        ok, vtail = verify(name, task, m["verify_s"])
    finally:
        T1.sh("docker", "rm", "-f", name)
    rec = {"task": task, "model": model_key, "idx": idx, "baseline": baseline, "steps": steps,
           "final_files": final, "ok_grade": {"ok": ok, "msg": vtail[-300:]},
           "collector": "tb2_collect", "mode": "agent",
           "served_model": T1.MODELS.get(model_key, ("", model_key))[1],
           "student_max_tokens": STUDENT_MAX_TOKENS,
           "max_steps": MAX_STEPS,
           "secs": round(time.time() - t0, 1)}
    (out_root / task).mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out_root / task / f"{model_key}_{idx}.json", "w"), indent=1)
    return rec


def cmd_cache():
    import yaml
    CACHE.mkdir(parents=True, exist_ok=True)
    man = []
    for t in universe():
        d = CACHE / t
        d.mkdir(exist_ok=True)
        yaml.safe_dump({"instruction": (DS / t / "instruction.md").read_text(),
                        **meta(t)}, open(d / "task.yaml", "w"), allow_unicode=True, width=10000)
        man.append(t)
    json.dump(man, open(CACHE / "manifest.json", "w"), indent=1)
    print(f"cache: {len(man)} tasks")


def cmd_batch(spec: str, K: int, model_key: str, workers: int):
    ids = universe() if spec == "all" else (
        [l.strip() for l in open(spec[1:]) if l.strip()] if spec.startswith("@") else spec.split(","))
    guides = {}
    gf = os.environ.get("TB2_GUIDE_FILE")
    arm = os.environ.get("TB2_GUIDE_ARM", "psyn")
    if gf:
        raw = json.load(open(gf))
        for t, r in raw.items():
            g = r.get("guides", {}).get(arm, "") if isinstance(r, dict) else ""
            if g:
                guides[t] = g
    # One batch per output root: a second batch on the same TB2_OUT would rerun unfinished tasks in the same
    # containers, so it waits up to 30 s for the lock and then exits.
    import fcntl
    RUNS.mkdir(parents=True, exist_ok=True)
    _lock = open(RUNS / ".batch.lock", "w")
    for _i in range(30):
        try:
            fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            time.sleep(1)
    else:
        print(f"BATCH-REFUSED: another tb2_collect batch already holds {RUNS}/.batch.lock", flush=True)
        return
    jobs = [(t, k) for t in ids for k in range(K)
            if not (RUNS / t / f"{model_key}_{k}.json").exists()]
    print(f"batch: {len(jobs)} rollouts ({len(ids)} tasks x K={K}, skip-existing), "
          f"workers={workers}", flush=True)
    stop = threading.Event()

    infra_streak = [0]
    streak_lock = threading.Lock()

    def work(job):
        if stop.is_set():
            return
        t, k = job
        try:
            rec = single(t, model_key, k, guide=guides.get(t, ""))
            with streak_lock:
                infra_streak[0] = 0
            print(f"{t}#{k}: ok={rec['ok_grade']['ok']} steps={len(rec['steps'])} "
                  f"{rec['secs']}s", flush=True)
        except T1.FatalInfraError as e:
            stop.set()
            print(f"BATCH-ABORT-FATAL: {str(e)[:200]} -- nothing recorded; relaunch after fixing the account", flush=True)
        except InfraError as e:
            with streak_lock:
                infra_streak[0] += 1
                n = infra_streak[0]
            print(f"{t}#{k}: SKIPPED (infra, not recorded): {str(e)[:150]}", flush=True)
            if n >= 3:
                stop.set()
                print("BATCH-ABORT-INFRA: 3 consecutive LLM outages -- stopping; relaunch to retry", flush=True)
        except Exception as e:
            print(f"{t}#{k}: FAILED {str(e)[:150]}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))
    print("BATCH-DONE" if not stop.is_set() else "BATCH-ABORTED", flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "cache":
        cmd_cache()
    elif c == "batch":
        cmd_batch(sys.argv[2], int(sys.argv[3]), sys.argv[4],
                  int(sys.argv[5]) if len(sys.argv) > 5 else 8)
    else:
        raise SystemExit("unknown subcommand")
