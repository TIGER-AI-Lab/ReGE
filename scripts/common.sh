# Shared settings for the run_*.sh pipelines. Sourced, not executed.
# usage inside a pipeline: source "$(dirname "$0")/common.sh" "$@"
#   $1      model line: gpt (gpt-5.6-luna) | claude (claude-sonnet-5)
#   $2...   stages to run (default: every stage of the pipeline, in order)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs result data

PY=${REGE_PYTHON:-python3}                 # main environment (requirements.txt)
AWPY=${ANDROID_WORLD_PYTHON:-$PY}          # environment with android_world installed
WORKERS=${WORKERS:-4}                      # concurrent rollouts per collector process
PARTS=${PARTS:-4}                          # parallel guide-builder processes

LINE=${1:?usage: $0 <gpt|claude> [stage ...]}
shift
case "$LINE" in
  gpt)    KEY=gpt56luna;       MODEL=gpt-5.6-luna;    BASE=${GPT_BASE_URL:-https://api.openai.com/v1} ;;
  claude) KEY=claude-sonnet5b; MODEL=claude-sonnet-5; BASE=${CLAUDE_SONNET5B_BASE:-https://api.anthropic.com/v1} ;;
  *) echo "model line must be gpt or claude" >&2; exit 2 ;;
esac
STAGES=("$@")

# Source generation, guidance construction (coach, judge, AggAgent) and student execution use one base model.
export AGG_URL="$BASE" AGG_MODEL="$MODEL" AGGAGENT_MODEL_KEY="$KEY" EXTRA_ARMS=aggagent
export POOL_KEEP_EMPTY=1

want() {                                   # want <stage>: true when no stage was named or <stage> was named
  [ ${#STAGES[@]} -eq 0 ] && return 0
  local s
  for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

log() { echo "$(date '+%F %T') [$LINE] $*"; }

build_parts() {                            # build_parts <log prefix> <command ...>: PARTS builders, part index and count appended
  local name=$1; shift
  local p pids=()
  for p in $(seq 0 $((PARTS - 1))); do
    "$@" "$p" "$PARTS" > "logs/${name}_p${p}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}
