#!/usr/bin/env bash
# BrowseComp-Plus pipeline (830 queries, fixed corpus): source pool -> ReGE-Sel judge -> guidance -> fresh students.
# usage: scripts/run_bcp.sh <gpt|claude> [cache] [pool] [judge] [guides] [arms]
# Needs data/bcp/{queries.jsonl,corpus.db,dense/} (see README), the Qwen3-Embedding-8B server at $BCP_EMB_URL
# for dense retrieval, and DEEPSEEK_API_KEY for the DeepSeek-V4-Flash equivalence judge.
source "$(dirname "$0")/common.sh" "$@"

IDS=result/bcp_luna_ids.txt
if [ "$LINE" = gpt ]; then
  POOL=runs_bcp_lunapool; ARMS_DIR=runs_bcp_lunaarms; TAG=luna
else
  POOL=runs_bcp_s5b200pool; ARMS_DIR=runs_bcp_s5b200arms; TAG=s5b200
fi
PICKS=result/judge_picks_bcp_$TAG.json
GUIDES=bcp_guides_$TAG
export BCP_MAX_STEPS=40 BCP_SUBMIT_NUDGE=1 BCP_JUDGE=dsv4flash BCP_EMB_URL=${BCP_EMB_URL:-http://127.0.0.1:8910/v1}

if want cache; then
  log "task cache and query lists"
  $PY src/bcp_collect.py cache
  $PY scripts/make_task_lists.py bcp
fi
if want pool; then
  log "source pool: 4 attempts per query -> data/$POOL"
  BCP_OUT=data/$POOL $PY src/bcp_collect.py @$IDS 4 "$KEY" "$WORKERS"
fi
if want judge; then
  log "ReGE-Sel judge"
  TB_RUNS=$POOL TASKCACHE=data/taskcache_bcp $PY src/judge_sweep.py bcp_$TAG @$IDS "$MODEL" "$WORKERS"
  $PY scripts/merge_parts.py picks result/judge_sweep_bcp_$TAG.json "$MODEL" "$PICKS"
fi
if want guides; then
  log "guidance: ReGE-Syn, ReGE-Sel, AggAgent -> result/${GUIDES}_all.json"
  TB_RUNS=$POOL TASKCACHE=data/taskcache_bcp JUDGE_PICK_FILE=$PICKS \
    build_parts "bcp_${TAG}_guides" $PY src/guides_build.py $IDS "$GUIDES" psyn,best1,aggagent
  $PY scripts/merge_parts.py guides "$GUIDES"
fi
if want arms; then
  for arm in control psyn best1 aggagent; do
    log "students: $arm -> data/$ARMS_DIR/$arm"
    BCP_OUT=data/$ARMS_DIR/$arm BCP_GUIDE_FILE=result/${GUIDES}_all.json BCP_GUIDE_ARM=$arm \
      $PY src/bcp_collect.py @$IDS 1 "$KEY" "$WORKERS"
  done
fi
log "done"
