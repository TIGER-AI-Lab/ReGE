#!/usr/bin/env bash
# Terminal-Bench 2.0 pipeline: source pool -> ReGE-Sel judge -> guidance -> fresh students.
# usage: scripts/run_tb2.sh <gpt|claude> [cache] [pool] [judge] [guides] [arms]
# Needs the 89 tasks under data/tb2_dataset/terminal-bench and Docker; the task images are pulled on first use.
source "$(dirname "$0")/common.sh" "$@"

IDS=result/tb2_ids.txt
if [ "$LINE" = gpt ]; then
  POOL=runs_tb2; ARMS_DIR=runs_tb2_arms; TAG=luna; POOL_KEY=$KEY
else
  POOL=runs_tb2_s5pool; ARMS_DIR=runs_tb2_s5darms; TAG=s5d; POOL_KEY=claude-sonnet5
fi
PICKS=result/judge_picks_tb2_$TAG.json
GUIDES=tb2_guides_$TAG
export TB2_MAX_STEPS=30 TB2_STUDENT_MAX_TOKENS=12000

if want cache; then
  log "task cache"
  $PY src/tb2_collect.py cache
  $PY scripts/make_task_lists.py tb2
fi
if want pool; then
  log "source pool: 4 attempts per task -> data/$POOL"
  TB2_OUT=data/$POOL $PY src/tb2_collect.py batch @$IDS 4 "$POOL_KEY" "$WORKERS"
fi
if want judge; then
  log "ReGE-Sel judge"
  TB_RUNS=$POOL TASKCACHE=data/taskcache_tb2 $PY src/judge_sweep.py tb2_$TAG @$IDS "$MODEL" "$WORKERS"
  $PY scripts/merge_parts.py picks result/judge_sweep_tb2_$TAG.json "$MODEL" "$PICKS"
fi
if want guides; then
  log "guidance: ReGE-Syn, ReGE-Sel, AggAgent -> result/${GUIDES}_all.json"
  TB_RUNS=$POOL TASKCACHE=data/taskcache_tb2 JUDGE_PICK_FILE=$PICKS TB2_GUIDE_IDS=$IDS TB2_GUIDE_PREFIX=$GUIDES \
    TB2_NEED=psyn,best1,aggagent build_parts "tb2_${TAG}_guides" $PY src/tb2_guides.py build
  $PY scripts/merge_parts.py guides "$GUIDES"
fi
if want arms; then
  for arm in control psyn best1 aggagent; do
    log "students: $arm -> data/$ARMS_DIR/$arm"
    TB2_OUT=data/$ARMS_DIR/$arm TB2_GUIDE_FILE=result/${GUIDES}_all.json TB2_GUIDE_ARM=$arm \
      $PY src/tb2_collect.py batch @$IDS 1 "$KEY" "$WORKERS"
  done
fi
log "done"
