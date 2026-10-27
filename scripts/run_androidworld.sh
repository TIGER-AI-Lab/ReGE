#!/usr/bin/env bash
# AndroidWorld pipeline (T3A text agent, 114 templates; GPT: parameter combinations 0,1,2; Claude: combination 0):
# source pool -> house records -> ReGE-Sel judge -> guidance -> the four arms interleaved per instance.
# usage: scripts/run_androidworld.sh <gpt|claude> [pool] [house] [judge] [guides] [arms]
# Needs $LANES booted emulators (console ports 5554, 5556, ...; gRPC ports +3000) and $ANDROID_WORLD_PYTHON
# (android_world installed).
source "$(dirname "$0")/common.sh" "$@"

LANES=${LANES:-6}
if [ "$LINE" = gpt ]; then
  POOL=runs_droid; HOUSE=runs_droid_house; ARMS_DIR=runs_droid_arms_ilv0825; TAG=luna; COMBOS=0,1,2
  GUIDES=droid_guides_luna
else
  POOL=runs_droid_s5epool; HOUSE=runs_droid_house_s5e; ARMS_DIR=runs_droid_s5xarms; TAG=s5e; COMBOS=0
  GUIDES=droid_guides_s5x
fi
TASKCACHE_DIR=data/taskcache_${HOUSE#runs_}
IDS=result/$([ "$LINE" = gpt ] && echo droid_ids.txt || echo droid_s5e_c0_ids.txt)
PICKS=result/judge_picks_droid_$TAG.json

lane_env() {                              # lane_env <i>: console and gRPC port of emulator lane i
  echo "DROID_CONSOLE_PORT=$((5554 + 2 * $1)) DROID_GRPC_PORT=$((8554 + 2 * $1))"
}

if want pool; then
  log "source pool: 4 attempts per instance, $LANES emulator lanes -> data/$POOL"
  pids=()
  for i in $(seq 0 $((LANES - 1))); do
    env $(lane_env "$i") DROID_OUT=data/$POOL \
      $AWPY src/droid_collect.py batch all "$COMBOS" 4 "$KEY" "$i" "$LANES" > "logs/droid_${TAG}_pool_l$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
fi
if want house; then
  log "house records -> data/$HOUSE (task text in $TASKCACHE_DIR)"
  DROID_POOL_DIR=$POOL DROID_HOUSE_DIR=$HOUSE DROID_POOL_KEY=$KEY DROID_POOL_K=4 $PY src/droid_guides_dsE.py transcribe
  find "data/$HOUSE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$IDS"
  log "$(wc -l < "$IDS") instances -> $IDS"
fi
if want judge; then
  log "ReGE-Sel judge"
  TB_RUNS=$HOUSE TASKCACHE=$TASKCACHE_DIR $PY src/judge_sweep.py droid_$TAG @$IDS "$MODEL" "$WORKERS"
  $PY scripts/merge_parts.py picks result/judge_sweep_droid_$TAG.json "$MODEL" "$PICKS"
fi
if want guides; then
  log "guidance: ReGE-Syn, ReGE-Sel (with the reset-and-verification text), AggAgent -> result/${GUIDES}_all.json"
  TB_RUNS=$HOUSE TASKCACHE=$TASKCACHE_DIR JUDGE_PICK_FILE=$PICKS DROID_HOUSE_DIR=$HOUSE DROID_POOL_KEY=$KEY \
    DROID_GUIDE_PREFIX=$GUIDES GUIDE_IDS_FILE=$IDS DROID_NEED=psyn,best1,aggagent \
    build_parts "droid_${TAG}_guides" $PY src/droid_guides_dsE.py build
  $PY scripts/merge_parts.py guides "$GUIDES"
fi
if want arms; then
  log "students: control, psyn, best1, aggagent interleaved per instance -> data/$ARMS_DIR"
  pids=()
  for i in $(seq 0 $((LANES - 1))); do
    awk -v n="$LANES" -v i="$i" '(NR - 1) % n == i' "$IDS" > "result/droid_${TAG}_lane$i.txt"
    env $(lane_env "$i") DROID_OUT=data/$ARMS_DIR DROID_ARMS=control,psyn,best1,aggagent \
      DROID_GUIDE_FILE=result/${GUIDES}_all.json DROID_ONLY_INSTANCES=result/droid_${TAG}_lane$i.txt \
      $AWPY src/droid_collect.py batch all "$COMBOS" 1 "$KEY" "$i" "$LANES" > "logs/droid_${TAG}_arms_l$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
fi
log "done"
