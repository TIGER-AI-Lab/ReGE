"""AggAgent (Lee/Yen/Ye/Chen, arXiv:2604.11753, github.com/princeton-pli/AggAgent) as a guidance arm over our
per-task source pools.

Their `aggagent` package runs unmodified: system and user prompts, the four tools
get_solution/search_trajectory/get_segment/finish, the agentic loop and the 100k context budget. Adaptations:
  (1) trajectory format: our pool steps {think, cmd, obs} become their ReAct message schema
      (assistant{content=think, tool_calls=[execute(cmd)]} + tool{content=obs}), with observations kept in full,
      followed by one assistant message with the attempt's final files or answer, the artifacts the ReGE coach sees;
  (2) delivery: the finish tool's `solution` text is handed to a fresh student under the same protocol as the
      other arms, since execution benchmarks have no answer string to submit;
  (3) integer tool arguments sent as strings are converted to integers before the tool is called;
  (4) with Claude, a trailing assistant turn is replaced by a user nudge in the request that is sent, because
      the API does not accept a conversation that ends with an assistant message.
The qwen_agent import is shimmed: their tools.py only uses BaseTool as a base class and the register_tool
decorator.

guidance_moa_experiment registers EXTRA_BUILDERS["aggagent"] = g_aggagent when EXTRA_ARMS includes aggagent.
"""
from __future__ import annotations
import json, os, sys, threading, time, types
from pathlib import Path

_REPO = Path(os.environ.get("AGGAGENT_REPO", Path(__file__).resolve().parent.parent / "third_party" / "AggAgent"))

# ---- shim qwen_agent before their tools.py imports it ----
if "qwen_agent.tools.base" not in sys.modules:
    _qa, _qt, _qb = (types.ModuleType(n) for n in ("qwen_agent", "qwen_agent.tools", "qwen_agent.tools.base"))

    class _BaseTool:
        def __init__(self, cfg=None):
            self.cfg = cfg or {}

    def _register_tool(name, allow_overwrite=False):
        def deco(cls):
            return cls
        return deco

    _qb.BaseTool, _qb.register_tool = _BaseTool, _register_tool
    _qa.tools, _qt.base = _qt, _qb
    sys.modules["qwen_agent"], sys.modules["qwen_agent.tools"], sys.modules["qwen_agent.tools.base"] = _qa, _qt, _qb

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from aggagent import AggAgent as _AggAgent            # their package, unmodified

import tb_collect as TC
import aggregate_deliverable as A
import door_probe as DP

_AGG_KEY = os.environ.get("AGGAGENT_MODEL_KEY")
if _AGG_KEY:
    AGG_BASE, AGG_MODEL = TC.SV.MODELS[_AGG_KEY]
else:
    AGG_BASE = os.environ.get("AGGAGENT_BASE", os.environ.get("CLAUDE_SONNET5B_BASE", "https://api.anthropic.com/v1"))
    AGG_MODEL = os.environ.get("AGGAGENT_MODEL", "claude-sonnet-5")


def _to_messages(steps, final=None):
    """Our pool step triples -> their ReAct message schema. Lossless (full obs). `final` (the attempt's final
    files/answer) becomes the last assistant message, which their get_solution tool returns."""
    msgs = []
    for i, s in enumerate(steps, 1):
        think, cmd, obs = (s.get("think") or "").strip(), s.get("cmd") or "", s.get("obs") or ""
        tc = ([{"id": f"call_{i}", "type": "function",
                "function": {"name": "execute", "arguments": json.dumps({"command": cmd}, ensure_ascii=False)}}]
              if cmd else [])
        msgs.append({"role": "assistant", "content": think, "tool_calls": tc})
        if cmd:
            msgs.append({"role": "tool", "name": "execute", "content": obs})
    if final is not None:
        msgs.append({"role": "assistant", "content": final})
    return msgs


AGGAGENT_HDR = ("## GUIDANCE (synthesized by an aggregation agent that inspected several previous "
                "independent attempts at this task; follow it but verify as you go):")


# Adaptation (3): some models emit integer tool arguments as JSON strings ("trajectory_id": "1"), which the
# upstream tools compare with ints. String or float values of integer-typed parameters (per each tool's own JSON
# schema) are converted before dispatch; everything else passes through untouched.
import aggagent.agent as _agmod_tc
_orig_call_tool = _agmod_tc.AggAgent.custom_call_tool


def _coerce_int_args(tool, tool_args):
    props = (getattr(tool, "parameters", None) or {}).get("properties") or {}
    if not isinstance(tool_args, dict):
        return tool_args
    out = dict(tool_args)
    for k, spec in props.items():
        if isinstance(spec, dict) and spec.get("type") == "integer" and k in out and not isinstance(out[k], bool):
            v = out[k]
            if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                out[k] = int(v.strip())
            elif isinstance(v, float) and v.is_integer():
                out[k] = int(v)
    return out


def _call_tool_coerced(self, tool_name, tool_args, **kwargs):
    tool = self.tool_map.get(tool_name)
    if tool is not None:
        tool_args = _coerce_int_args(tool, tool_args)
    return _orig_call_tool(self, tool_name, tool_args, **kwargs)


_agmod_tc.AggAgent.custom_call_tool = _call_tool_coerced


# Adaptation (4): the package's loop accepts only replies that call a tool; after a reply without one it appends
# {"role": "assistant", "content": "Server error"} and sends the conversation again, so the next request ends
# with an assistant turn, which the Claude API rejects. In the copy that is sent, a trailing assistant turn is
# replaced by an equivalent user nudge; the package's own message list is untouched. Only Claude models are
# affected.
_orig_call_server = _agmod_tc.AggAgent.call_server
_NUDGE = ("Your previous reply did not call a tool. Respond with exactly one tool call "
          "(get_solution / search_trajectory / get_segment / finish).")


def _call_server_prefill_safe(self, messages, max_tries=5, only_finish=False):
    msgs = list(messages)
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant" and not msgs[-1].get("tool_calls"):
        msgs = msgs[:-1] + [{"role": "user", "content": _NUDGE}]
    return _orig_call_server(self, msgs, max_tries=max_tries, only_finish=only_finish)


if "claude" in AGG_MODEL.lower():
    _agmod_tc.AggAgent.call_server = _call_server_prefill_safe


# ---- token accounting (observation only; the loop's behaviour is unchanged) ----
# _make_output is the single exit of every run, so a wrapper keeps the package's per-round statistics on the
# instance.
_orig_make_output = _agmod_tc.AggAgent._make_output


def _make_output_keep(self, result, messages, stats):
    self._routemoa_stats = stats
    return _orig_make_output(self, result, messages, stats)


_agmod_tc.AggAgent._make_output = _make_output_keep
_USAGE = threading.local()

# The package's own tally covers only the replies it keeps; every completion is also counted here, at the
# litellm call, while a run is active (thread-local; None otherwise).
_orig_litellm_completion = _agmod_tc.litellm.completion


def _completion_counted(*args, **kwargs):
    resp = _orig_litellm_completion(*args, **kwargs)
    acc = getattr(_USAGE, "calls", None)
    if acc is not None:
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                acc.append([None, None])
            else:
                # Claude endpoints report prompt_tokens without cached input; cache reads and writes are recorded
                # per call (positions 2/3) when present.
                cw = []
                for _k in ("cache_read_input_tokens", "cache_creation_input_tokens"):
                    _v = (getattr(u, "model_extra", None) or {}).get(_k)
                    if _v is None:
                        _v = getattr(u, _k, None)
                    cw.append(int(_v) if isinstance(_v, (int, float)) else None)
                acc.append([int(getattr(u, "prompt_tokens", None) or 0),
                            int(getattr(u, "completion_tokens", None) or 0)] + cw)
        except Exception:
            acc.append([None, None])
    return resp


_agmod_tc.litellm.completion = _completion_counted


def usage_reset():
    _USAGE.last = None


def usage_take():
    u = getattr(_USAGE, "last", None)
    _USAGE.last = None
    return u


def _summarize_stats(stats, calls, secs, solved):
    """stats = their per-round tally (kept replies only); calls = every litellm completion of this run.
    'in'/'out' are the total spend (all calls); the rounds-only view is kept as in_rounds/out_rounds."""
    st = stats or {}
    rounds = st.get("token_usage_each_step") or []
    r_in = [r.get("input_tokens") for r in rounds if r.get("input_tokens") is not None]
    r_out = [r.get("output_tokens") for r in rounds if r.get("output_tokens") is not None]
    calls = calls or []
    c_in = [c[0] for c in calls if c[0] is not None]
    c_out = [c[1] for c in calls if c[1] is not None]
    c_cr = [c[2] for c in calls if len(c) > 2 and c[2] is not None]
    c_cw = [c[3] for c in calls if len(c) > 3 and c[3] is not None]
    return {"iterations": st.get("iterations"), "api_calls": len(calls), "rounds_kept": len(rounds),
            "hidden_retries": max(0, len(calls) - len(rounds)),
            "calls_without_usage": len(calls) - len(c_in),
            "in": sum(c_in), "out": sum(c_out), "in_max_call": max(c_in) if c_in else 0,
            **({"cache_read": sum(c_cr), "cache_write": sum(c_cw)} if c_cr or c_cw else {}),
            "in_rounds": sum(r_in), "out_rounds": sum(r_out),
            "server_errors": st.get("server_errors"), "tool_call_errors": st.get("tool_call_errors"),
            "tool_calls": st.get("tool_calls"), "context_limit_reached": st.get("context_limit_reached"),
            "secs": round(secs, 1), "solution": bool(solved), "model": AGG_MODEL,
            "per_call": [list(c) for c in calls]}


def _run_aggagent(task, instr, rolls, store, ctx):
    agent = _AggAgent(
        model=AGG_MODEL,
        task="",                                       # generic variant of their prompts
        llm_kwargs={"model": "openai/" + AGG_MODEL,
                    "api_key": TC.SV.key_for(AGG_BASE),
                    "api_base": AGG_BASE,
                    "temperature": 1.0,
                    "max_tokens": 10000,               # per round (their default)
                    "parallel_tool_calls": False,
                    "timeout": 180,                    # per round
                    **DP.thinking_off_kw(AGG_MODEL)},
    )
    trajs = [_to_messages(r["d"].get("steps", []), "FINAL FILES:\n" + A.serialize_files(r["deliv"], store))
             for r in rolls]
    out, t0 = None, time.time()
    _USAGE.calls = []                                  # arm the litellm-boundary counter for this run
    try:
        out = agent.run(question=instr, trajectories=trajs)
    finally:                                           # record the spend even when the loop died
        calls, _USAGE.calls = getattr(_USAGE, "calls", None), None
        usage = _summarize_stats(getattr(agent, "_routemoa_stats", None), calls, time.time() - t0,
                                 bool(out and out.get("solution")))
        _USAGE.last = usage
        if isinstance(ctx, dict):
            ctx["_aggagent_usage"] = usage
    if not out.get("solution"):
        raise RuntimeError(f"aggagent produced no solution: {out.get('error')}")
    if isinstance(ctx, dict):
        ctx["_aggagent_reason"] = out.get("reason")
    return AGGAGENT_HDR + "\n\n" + out["solution"]


def g_aggagent(task, instr, rolls, store, ctx):
    return _run_aggagent(task, instr, rolls, store, ctx)
