#!/usr/bin/env bash
# Spider2-Lite pipeline (135 local SQLite instances): source pool -> house records -> ReGE-Sel judge -> guidance
# -> fresh students -> official grading.
# usage: scripts/run_spider.sh <gpt|claude> [cache] [pool] [house] [judge] [guides] [arms]
# Needs Spider2 (spider2-lite with the unpacked local databases) at $SPIDER2_LITE_DIR or third_party/Spider2/spider2-lite.
source "$(dirname "$0")/common.sh" "$@"

IDS=result/spider_ids.txt
if [ "$LINE" = gpt ]; then
  POOL=runs_spider; HOUSE=runs_spider_house; ARMS_DIR=runs_spider_arms; TAG=luna
  POOL_GRADES=spider_grade_pool_${KEY}.json; ARM_TAG() { echo "$1"; }
else
  POOL=runs_spider_s5epool; HOUSE=runs_spider_house_s5e; ARMS_DIR=runs_spider_s5earms; TAG=s5e
  POOL_GRADES=spider_grade_${POOL}_${KEY}.json; ARM_TAG() { echo "$1_s5e"; }
fi
PICKS=result/judge_picks_spider_$TAG.json
GUIDES=spider_guides_$TAG
export SPIDER_MAX_CALLS=18

if want cache; then
  log "task cache and evaluation working copy"
  $PY src/spider_collect.py cache
  $PY scripts/make_task_lists.py spider
fi
if want pool; then
  log "source pool: 4 attempts per instance -> data/$POOL"
  SPIDER_OUT=data/$POOL $PY src/spider_collect.py batch @$IDS 4 "$KEY" "$WORKERS"
  SPIDER_OUT=data/$POOL $PY src/spider_collect.py grade "$KEY" 0,1,2,3
fi
if want house; then
  log "house records -> data/$HOUSE"
  SPIDER_POOL_DIR=$POOL SPIDER_HOUSE_DIR=$HOUSE SPIDER_POOL_KEY=$KEY SPIDER_POOL_K=4 SPIDER_POOL_GRADES=$POOL_GRADES \
    $PY src/spider_guides_s5.py transcribe
fi
if want judge; then
  log "ReGE-Sel judge"
  TB_RUNS=$HOUSE TASKCACHE=data/taskcache_spider $PY src/judge_sweep.py spider_$TAG @$IDS "$MODEL" "$WORKERS"
  $PY scripts/merge_parts.py picks result/judge_sweep_spider_$TAG.json "$MODEL" "$PICKS"
fi
if want guides; then
  log "guidance: ReGE-Syn, ReGE-Sel, AggAgent -> result/${GUIDES}_all.json"
  TB_RUNS=$HOUSE TASKCACHE=data/taskcache_spider JUDGE_PICK_FILE=$PICKS SPIDER_HOUSE_DIR=$HOUSE SPIDER_POOL_KEY=$KEY \
    SPIDER_GUIDE_IDS=$IDS SPIDER_GUIDE_PREFIX=$GUIDES SPIDER_GUIDE_ALL=${GUIDES}_all.json SPIDER_NEED=psyn,best1,aggagent \
    build_parts "spider_${TAG}_guides" $PY src/spider_guides_s5.py build
  $PY scripts/merge_parts.py guides "$GUIDES"
fi
if want arms; then
  for arm in control psyn best1 aggagent; do
    log "students: $arm -> data/$ARMS_DIR/$arm"
    SPIDER_OUT=data/$ARMS_DIR/$arm SPIDER_GUIDE_FILE=result/${GUIDES}_all.json SPIDER_GUIDE_ARM=$arm \
      $PY src/spider_collect.py batch @$IDS 1 "$KEY" "$WORKERS"
    SPIDER_OUT=data/$ARMS_DIR/$arm SPIDER_GRADE_TAG=$(ARM_TAG $arm) $PY src/spider_collect.py grade "$KEY" 0
  done
fi
log "done"
