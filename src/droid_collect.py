"""AndroidWorld collector and grader (official environment, T3A text agent, grading on the device state).

- Environment, task instantiation, agent and grading are official android_world code: env_launcher.load_and_setup_env,
  suite_utils.create_suite (n_task_combinations, seed) for parameters shared across arms, agents.t3a.T3A, and
  success = task.is_successful(env) after the agent signals done, within int(task.complexity * 10) steps.
- Guidance enters through T3A's own task-guideline interface, set_task_guidelines([guide]).
- Additions: a model wrapper for OpenAI-compatible endpoints (wall-clock cap and retries) and the run layout.

Runs under the AndroidWorld Python environment ($ANDROID_WORLD_PYTHON). An emulator must already be booted:
  ~/Android/sdk/emulator/emulator -avd AndroidWorldAvd(-lane) -no-window -no-audio \
      -no-snapshot -gpu swiftshader_indirect -grpc <GRPC> -port <CONSOLE>
plus android_world's one-time app setup (emulator_setup=True) on that AVD image.

Env vars: DROID_CONSOLE_PORT (5554), DROID_GRPC_PORT (8554), DROID_OUT, DROID_SEED (30),
          DROID_GUIDE_FILE + DROID_GUIDE_ARM (guidance), DROID_ARMS + DROID_ONLY_INSTANCES (interleaved arms)
Layout: data/<DROID_OUT>/<TaskName>/<model>_c<combo>_<idx>.json

usage: droid_collect.py batch <all|csv|@file> <combos> <K> <model> [lane nlanes]
"""
from __future__ import annotations
import json, os, re, sys, time
from pathlib import Path
import door_probe as DP

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("DROID_OUT") or (ROOT / "data" / "runs_droid"))
SEED = int(os.environ.get("DROID_SEED", "30"))
CONSOLE = int(os.environ.get("DROID_CONSOLE_PORT", "5554"))
GRPC = int(os.environ.get("DROID_GRPC_PORT", "8554"))
ADB = os.environ.get("ADB", "adb")

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


class InfraError(RuntimeError):
    """The model could not be reached for this rollout (7 attempts). The rollout is not recorded (a later batch
    retries it) and is never counted as a task failure."""


class FatalInfraError(InfraError):
    """Account-level failure (402 insufficient balance, 401/403 authentication): every further call would fail the
    same way, so the whole batch stops at once."""


_FATAL_RE = re.compile(r"Error code: 40[123]\b|Insufficient Balance|insufficient_quota|quota_exhausted|invalid_api_key|authentication_error|account_deactivated", re.I)


MODELS = {  # same keys and endpoints as stitch_validate.MODELS
    "gpt56luna": (os.environ.get("GPT_BASE_URL", "https://api.openai.com/v1"), "gpt-5.6-luna"),
    "claude-sonnet5": (os.environ.get("CLAUDE_SONNET5_BASE", os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1")), "claude-sonnet-5"),
    "claude-sonnet5b": (os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1"), "claude-sonnet-5"),
    "dsv4flash": ("https://api.deepseek.com/v1", "deepseek-v4-flash"),
}

_ENV = None


def get_env():
    global _ENV
    if _ENV is None:
        from android_world.env import env_launcher
        _ENV = env_launcher.load_and_setup_env(
            console_port=CONSOLE, emulator_setup=False, adb_path=ADB, grpc_port=GRPC)
    return _ENV


def registry_dict():
    from android_world import registry
    tr = registry.TaskRegistry()
    return tr.get_registry(tr.ANDROID_WORLD_FAMILY)


def template_names() -> list[str]:
    return sorted(registry_dict().keys())


class ProxyWrapper:
    """infer.LlmWrapper for an OpenAI-compatible endpoint: a wall-clock cap per call (a call over the cap is
    abandoned to a side thread and retried on a fresh connection) and up to 7 attempts with backoff."""

    _SIDE = None
    CALL_CAP = int(os.environ.get("DROID_CALL_CAP", "300"))

    def __init__(self, model_key: str):
        from openai import OpenAI
        from concurrent.futures import ThreadPoolExecutor
        base, model = MODELS[model_key]
        self.client = OpenAI(base_url=base, api_key=_api_key(base), timeout=600)
        self.model = model
        self.usage = []
        if ProxyWrapper._SIDE is None:
            ProxyWrapper._SIDE = ThreadPoolExecutor(max_workers=16)

    def predict(self, text_prompt: str):
        from android_world.agents import infer
        from concurrent.futures import TimeoutError as FutTimeout
        last = None
        for i in range(7):
            fut = ProxyWrapper._SIDE.submit(
                self.client.chat.completions.create, model=self.model,
                messages=[{"role": "user", "content": text_prompt}], **DP.thinking_off_kw(self.model))
            try:
                r = fut.result(timeout=ProxyWrapper.CALL_CAP)
                try:
                    self.usage.append({"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens})
                except Exception:
                    pass
                return (r.choices[0].message.content or ""), None, r
            except FutTimeout:
                fut.cancel()
                last = f"call-cap {ProxyWrapper.CALL_CAP}s exceeded"
            except Exception as e:
                last = e
                if _FATAL_RE.search(str(e)):          # account-level: retrying cannot help, abort the batch
                    print(f"LLM FATAL: {str(e)[:160]}", flush=True)
                    raise FatalInfraError(f"{self.model}: {str(e)[:160]}")
            time.sleep(min(2 ** i, 45))
        print(f"LLM FAILED after retries: {str(last)[:120]}", flush=True)
        # T3A's error sentinel would end the episode as a zero-step failure; report an infrastructure error instead
        raise InfraError(f"{self.model}: LLM unreachable after 7 attempts: {str(last)[:140]}")


def make_task(name: str, combo: int):
    """Official deterministic instantiation: same (name, combo) -> same params across arms."""
    from android_world import suite_utils
    reg = registry_dict()
    suite = suite_utils.create_suite(
        {name: reg[name]}, n_task_combinations=combo + 1, seed=SEED)
    return suite[name][combo]


def single(name: str, model_key: str, combo: int, k: int, guide: str = "") -> dict:
    """One rollout on instance (template, combo); k = rollout index within the pool."""
    from android_world.agents import t3a
    env = get_env()
    task = make_task(name, combo)
    rec = {"task": name, "model": model_key, "combo": combo, "k": k, "goal": task.goal,
           "params": json.dumps(task.params, default=str)[:800], "complexity": task.complexity}
    t0 = time.time()
    env.reset(go_home=True)
    task.initialize_task(env)
    llm = ProxyWrapper(model_key)
    agent = t3a.T3A(env, llm)
    if guide:
        agent.set_task_guidelines([guide])
    max_steps = int((task.complexity or 2) * 10)
    is_done = False
    summaries = []
    err = None
    infra = None
    try:
        for _ in range(max_steps):
            resp = agent.step(task.goal)
            s = (resp.data or {}).get("summary")
            if s:
                summaries.append(str(s)[:220])
            if resp.done:
                is_done = True
                break
    except InfraError as e:
        infra = e
    except Exception as e:
        err = str(e)[:200]
    if infra is None:
        try:
            rec["success"] = int(is_done and task.is_successful(env) == 1)
        except Exception as e:
            rec["success"] = 0
            rec["grade_error"] = str(e)[:150]
    try:
        task.tear_down(env)
    except Exception:
        pass
    if infra is not None:
        # the model never answered: no record on disk (a later batch retries it), not a failure
        print(f"{name}.c{combo}#{k}: NOT RECORDED (infra after {len(summaries)} steps): {str(infra)[:150]}", flush=True)
        raise infra
    rec.update({"done": is_done, "n_steps": len(summaries), "max_steps": max_steps,
                "secs": round(time.time() - t0, 1), "summaries": summaries,
                "usage": llm.usage})
    if err:
        rec["error"] = err
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out / f"{model_key}_c{combo}_{k}.json", "w"), indent=1)
    return rec


def cmd_batch(spec: str, combos: list[int], K: int, model_key: str,
              lane: int = 0, nlanes: int = 1):
    """Jobs = templates x combos x K, sharded over nlanes emulator lanes."""
    RUNS.mkdir(parents=True, exist_ok=True)
    # One batch per emulator: two batches on the same console port would drive one device at once, so a second
    # launch waits up to 30 s for the lock and then exits.
    import fcntl
    _lock = open(f"/tmp/droid_lane_{CONSOLE}.lock", "w")
    for _i in range(30):                        # 30s grace: a just-finished predecessor may still hold it
        try:
            fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            time.sleep(1)
    else:
        print(f"BATCH-REFUSED: another droid_collect batch already drives console {CONSOLE}", flush=True)
        return
    if spec == "all":
        names = template_names()
    elif spec.startswith("@"):
        names = [l.strip() for l in open(spec[1:]) if l.strip()]
    else:
        names = spec.split(",")
    # The Vlc* templates crash the emulator, so they are excluded by default (DROID_EXCLUDE='' keeps them).
    excl = [p for p in os.environ.get("DROID_EXCLUDE", "Vlc").split(",") if p]
    dropped = [n for n in names if any(n.startswith(p) for p in excl)]
    if dropped:
        names = [n for n in names if n not in dropped]
        print(f"excluded {len(dropped)} emulator-killer templates: {dropped}", flush=True)
    guides = {}
    gf = os.environ.get("DROID_GUIDE_FILE")
    arm = os.environ.get("DROID_GUIDE_ARM", "psyn")
    if gf:
        guides = json.load(open(gf))
    alljobs = [(n, c, k) for n in names for c in combos for k in range(K)]
    # The remaining work is sharded: lane = (slot + k) % nlanes, where slot numbers the (template, combo) pairs
    # that still have work, so every lane gets an equal share.
    # Interleaved arms: DROID_ARMS="control,psyn,best1,aggagent" runs every arm back to back on the same
    # instance in one process, writing each into <DROID_OUT>/<arm>/, so all arms of an instance share one emulator
    # session; the arm order rotates per instance. Without DROID_ARMS a single arm runs (the path below).
    arms_env = [a.strip() for a in os.environ.get("DROID_ARMS", "").split(",") if a.strip()]
    if arms_env:
        base = RUNS
        done_f = lambda a, n, c, k: base / a / n / f"{model_key}_c{c}_{k}.json"
        # The pipeline gives each lane an explicit instance list (DROID_ONLY_INSTANCES), so no two lanes run the
        # same instance.
        only = os.environ.get("DROID_ONLY_INSTANCES")
        if only:
            want = {l.strip() for l in open(only) if l.strip()}
            alljobs = [j for j in alljobs if f"{j[0]}__c{j[1]}" in want]
            lane, nlanes = 0, 1
        # An instance with any arm missing is redone in full, so all arms of an instance come from one session;
        # completed instances are skipped.
        todo = [j for j in alljobs if not all(done_f(a, *j).exists() for a in arms_env)]
        slots = {}
        for n, c, k in todo:
            slots.setdefault((n, c), len(slots))
        mine = [j for j in todo if (slots[(j[0], j[1])] + j[2]) % nlanes == lane]
        print(f"lane {lane}/{nlanes} console={CONSOLE}: {len(mine)} instances x arms={arms_env} "
              f"(interleaved, instance-atomic{', explicit list' if only else ''}) -> {base}", flush=True)
        consecutive_infra = 0
        for idx, (n, c, k) in enumerate(mine):
            order = arms_env[idx % len(arms_env):] + arms_env[:idx % len(arms_env)]     # rotate arm order
            for a in order:
                g = guides.get(f"{n}__c{c}") or guides.get(n) or ""
                if isinstance(g, dict):
                    g = (g.get("guides") or {}).get(a, "")
                globals()["RUNS"] = base / a
                try:
                    rec = single(n, model_key, c, k, guide=g)
                    consecutive_infra = 0
                    print(f"[{a}] {n}.c{c}#{k}: success={rec['success']} done={rec['done']} "
                          f"steps={rec['n_steps']}/{rec['max_steps']} {rec['secs']}s", flush=True)
                except FatalInfraError as e:
                    globals()["RUNS"] = base
                    print(f"BATCH-ABORT-FATAL lane {lane}: {str(e)[:200]} -- nothing recorded", flush=True)
                    return
                except InfraError as e:
                    consecutive_infra += 1
                    print(f"[{a}] {n}.c{c}#{k}: SKIPPED (infra, not recorded): {str(e)[:150]}", flush=True)
                    if consecutive_infra >= 3:
                        globals()["RUNS"] = base
                        print(f"BATCH-ABORT-INFRA lane {lane}: 3 consecutive LLM outages -- stopping", flush=True)
                        return
                except Exception as e:
                    print(f"[{a}] {n}.c{c}#{k}: FAILED {str(e)[:150]}", flush=True)
        globals()["RUNS"] = base
        print(f"LANE-{lane}-DONE", flush=True)
        return
    # ---- single arm ----
    todo = [j for j in alljobs if not (RUNS / j[0] / f"{model_key}_c{j[1]}_{j[2]}.json").exists()]
    slots = {}
    for n, c, k in todo:
        slots.setdefault((n, c), len(slots))
    mine = [j for j in todo if (slots[(j[0], j[1])] + j[2]) % nlanes == lane]
    print(f"lane {lane}/{nlanes} console={CONSOLE}: {len(mine)} rollouts "
          f"({len(names)} templates x combos={combos} x K={K}, skip-existing)", flush=True)
    consecutive_infra = 0
    for n, c, k in mine:
        try:
            g = guides.get(f"{n}__c{c}") or guides.get(n) or ""
            if isinstance(g, dict):
                g = (g.get("guides") or {}).get(arm, "")
            rec = single(n, model_key, c, k, guide=g)
            consecutive_infra = 0
            print(f"{n}.c{c}#{k}: success={rec['success']} done={rec['done']} "
                  f"steps={rec['n_steps']}/{rec['max_steps']} {rec['secs']}s", flush=True)
        except FatalInfraError as e:
            print(f"BATCH-ABORT-FATAL lane {lane}: {str(e)[:200]} -- nothing recorded; relaunch after fixing the account", flush=True)
            return
        except InfraError as e:
            consecutive_infra += 1
            print(f"{n}.c{c}#{k}: SKIPPED (infra, not recorded): {str(e)[:150]}", flush=True)
            if consecutive_infra >= 3:
                print(f"BATCH-ABORT-INFRA lane {lane}: 3 consecutive LLM outages -- stopping; relaunch to retry", flush=True)
                return
        except Exception as e:
            print(f"{n}.c{c}#{k}: FAILED {str(e)[:150]}", flush=True)
    print(f"LANE-{lane}-DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "batch":
        combos = [int(x) for x in sys.argv[3].split(",")]
        lane = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        nlanes = int(sys.argv[7]) if len(sys.argv) > 7 else 1
        cmd_batch(sys.argv[2], combos, int(sys.argv[4]), sys.argv[5], lane, nlanes)
    else:
        raise SystemExit("unknown subcommand")
