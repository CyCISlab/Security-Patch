#!/bin/bash
# rerun_v3 runner: supervisor's final runs (Commit-only benign targets). Same protocol as Phase 3
# (runs_v2/06_phase3/run_phase3_v8.sh): same train.py / evaluate_cvefixes.py commands, 10-fold ensemble, tau from
# validation, retry once and keep both logs, skip finished runs. Differences, and only these:
#   * train.py gets --comment commitonly --benign-commit-only (benign targets drawn from Commit events only); caches
#     and models therefore live in new folders (*_commitonly*), nothing from Phase 3 is overwritten
#   * evaluate_cvefixes.py gets --benign-commit-only --model-comment commitonly
#   * caches are built ahead on CPU (02_caches/prebuild_cache.py, the Phase 3 prebuild route) and each cache must
#     pass the split gate (02_caches/split_check.py: same train+val and test repos, same order, same fixes, as the
#     Phase 3 cache) before anything trains on it. train.py is never allowed to build a cache itself.
#   * checkpointing (Anonusr: run overnight until ~10:00, continue ~15:00): if rerun_v3/PAUSE_AT holds a time
#     ("2026-11-21 09:45"), no job starts unless its estimated length ends before that time. The runner then writes
#     rerun_v3/PAUSED and exits; `bash rerun_v3/resume.sh` continues. Finished runs are skipped on resume.
# Order: Task 1 (priority 1) no metadata, then Task 1 metadata, then Task 2 ablation, Task 4 full corpus, Task 3
# draws (seed 42 first), Task 5 from-scratch rebuild check. Every event goes to rerun_v3/00_RERUN_V3_SUMMARY.log.
cd "$(dirname "$0")/.."
R=rerun_v3; SUMMARY=$R/00_RERUN_V3_SUMMARY.log
log() { echo "[$(date '+%a %F %T')] $*" | tee -a $SUMMARY; }
rm -f $R/PAUSED
echo $$ > $R/runner.pid
log "=== rerun_v3 runner started (pid $$); pause at: $(cat $R/PAUSE_AT 2>/dev/null || echo none) ==="

# ---------------- gate 0: code tests must have passed ----------------
for T in test_commit_only_dataset test_commit_only_external test_split_replay; do
  F=$R/01_code_changes/tests/$T.txt
  if [ "$(grep -c 'ALL PASS' $F 2>/dev/null)" -lt 1 ] || grep -q "^FAIL" $F; then
    log "GATE FAILED: $F does not show ALL PASS; nothing run"; exit 1; fi
done
log "gate 0: code tests ALL PASS ($R/01_code_changes/tests/*.txt)"

# ---------------- checkpoint ----------------
# estimated seconds per job (Phase 3 medians x ~1.6, for CPU contention)
declare -A EST=([cache]=6000 [split]=600 [transformer]=900 [conv1d]=450 [gru]=400 [bilstm]=650 [near]=900 [unseen]=420 [cols]=1800 [fc]=5400 [draw]=1200)
can_start() {  # $1 job kind -> 0 if it may start now
  [ -f $R/PAUSE_AT ] || return 0
  local P=$(date -d "$(cat $R/PAUSE_AT)" +%s 2>/dev/null) || return 0
  [ $(( $(date +%s) + ${EST[$1]} )) -le $P ]
}
pause() { [ -f $R/PAUSED ] || { log "PAUSE: next job ($1) would not finish before $(cat $R/PAUSE_AT); stopping here. Resume: bash rerun_v3/resume.sh"; touch $R/PAUSED; }; }

# ---------------- names ----------------
expname() {  # $1 AGGNAME $2 meta(0/1) $3 seed  -> train.py exp name with --comment commitonly
  local n="Aggregate.$1_R1_B20"; [ "$2" = 1 ] && n="${n}_meta"; n="${n}_commitonly"; [ "$3" != 42 ] && n="${n}_s$3"; echo "$n"
}
aggname() { [ $1 = sym ] && echo before_cve || echo only_before; }
aggflag() { [ $1 = sym ] && echo before || echo only_before; }

# ---------------- caches (CPU, 3 workers, background) ----------------
mkdir -p $R/02_caches/build_logs $R/02_caches/incomplete
build_cache() {  # $1 sym|causal $2 seed $3 meta
  local NAME=$(expname $(aggname $1) $3 $2) L=$R/02_caches/build_logs/cache_$1_s$2_meta$3.log
  [ -f ready_data/$NAME/column_names.pkl ] && return 0
  [ -f $R/PAUSED ] && return 0
  if ! can_start cache; then pause "cache $NAME"; return 0; fi
  if [ -d ready_data/$NAME ]; then mv ready_data/$NAME $R/02_caches/incomplete/$NAME.$(date +%s); log "C  moved incomplete ready_data/$NAME to $R/02_caches/incomplete/"; fi
  [ -f $L ] && mv $L ${L%.log}.interrupted.$(date +%s).log
  log "C  START $NAME -> $L"; local T0=$(date +%s)
  if CUDA_VISIBLE_DEVICES= nice -n 5 python3 $R/02_caches/prebuild_cache.py $(aggflag $1) $2 $3 > $L 2>&1 \
     && [ -f ready_data/$NAME/column_names.pkl ]; then
    log "C  DONE  $NAME in $(( $(date +%s)-T0 ))s | $(grep PREBUILD $L)"
  else log "C  FAIL  $NAME (see $L)"; fi
}
cworker() { for J in "$@"; do IFS=: read W S M <<< "$J"; build_cache $W $S $M; done; }
if ! pgrep -f "rerun_v3/02_caches/prebuild_cache.py" > /dev/null; then
( cworker sym:42:0 sym:123:0 sym:7:0 sym:256:0 sym:999:0 causal:256:1 causal:999:1 &
  cworker causal:42:0 causal:123:0 causal:7:0 causal:256:0 causal:999:0 sym:999:1 causal:7:1 &
  cworker sym:42:1 causal:42:1 sym:123:1 causal:123:1 sym:7:1 sym:256:1 &
  wait; log "C  cache workers finished" ) &
fi
cache_ready() { [ -f ready_data/$(expname $(aggname $1) $3 $2)/column_names.pkl ]; }

# split gate, once per cache: 0 PASS, 1 FAIL, 2 not ready / paused
split_ok() {  # $1 sym|causal $2 seed $3 meta
  local NAME; NAME=$(expname $(aggname $1) $3 $2)
  local V=$R/02_caches/split_checks/split_$NAME.txt   # fix 1: was in the same local as NAME (expanded empty)
  cache_ready $1 $2 $3 || return 2
  if [ ! -f $V ]; then
    can_start split || { pause "split check $NAME"; return 2; }
    CUDA_VISIBLE_DEVICES= python3 $R/02_caches/split_check.py $(aggflag $1) $2 $3 > $R/02_caches/split_checks/run_$NAME.log 2>&1
    log "SPLIT $NAME: $(tr '\n' ' ' < $V 2>/dev/null || echo 'no verdict, see run log')"
  fi
  grep -q "^VERDICT: PASS" $V 2>/dev/null && return 0 || return 1
}

# ---------------- one run (same as Phase 3 run_with_retry / run_config) ----------------
run_with_retry() {  # LOG REGEX OUTFILE cmd...
  local L=$1 RX=$2 OUT=$3; shift 3
  for ATT in 1 2; do
    local MARK=$(mktemp); local T0=$(date +%s)
    "$@" > "$L" 2>&1; local RC=$?
    if [ $RC -eq 0 ] && grep -q "$RX" "$L" && { [ -z "$OUT" ] || [ "$OUT" -nt "$MARK" ]; }; then
      rm -f $MARK; LAST_DT=$(( $(date +%s)-T0 )); return 0; fi
    rm -f $MARK; log "    attempt $ATT FAILED rc=$RC ($L)"
    [ $ATT = 1 ] && mv "$L" "${L%.log}.attempt1.log"
  done
  return 1
}

# returns 0 = finished (or skipped as done), 2 = waiting (cache not ready), 3 = paused, 1 = failed
run_config() {  # $1 DIR $2 MODEL $3 WIN $4 seed $5 meta [$6 ablation group]
  local DIR=$1 MODEL=$2 WIN=$3 S=$4 META=$5 G=$6
  local AGGR=$(aggflag $WIN) POOL; [ $WIN = sym ] && POOL=center || POOL=last
  local MF="" EF="" SUF="_nometa"; [ $META = 1 ] && { MF="--metadata"; EF="--meta"; SUF=""; }
  local EXP=$(expname $(aggname $WIN) $META $S) TAG="${MODEL}_${WIN}_b20_s${S}${SUF}" AF="" COLS=""
  if [ -n "$G" ]; then
    TAG="${TAG}_ablate-$G"; COLS=$R/03_ablation_columns/columns_${EXP}.txt; AF="--ablate $G --columns-file $COLS"
  fi
  local XP=$EXP${G:+_ablate-$G}
  mkdir -p $DIR
  if [ -n "$G" ]; then [ -f $DIR/preds_$TAG.npz ] && return 0
  else [ -f $DIR/preds_$TAG.npz ] && [ -f $DIR/ext_near_$TAG.npz ] && [ -f $DIR/ext_unseen_$TAG.npz ] && return 0; fi
  [ -f $DIR/FAILED_$TAG ] && return 1
  split_ok $WIN $S $META; local SP=$?
  [ $SP = 2 ] && { [ -f $R/PAUSED ] && return 3 || return 2; }
  if [ $SP = 1 ]; then log "FAIL  $TAG not run: split check failed ($R/02_caches/split_checks/split_$EXP.txt)"; touch $DIR/FAILED_$TAG; return 1; fi
  if [ -n "$G" ] && ! grep -q "VERDICT: VERIFIED" $COLS 2>/dev/null; then return 2; fi
  if [ ! -f $DIR/preds_$TAG.npz ]; then
    can_start $MODEL || { pause "train $TAG"; return 3; }
    [ -f $DIR/logs_$TAG.log ] && { mv $DIR/logs_$TAG.log $DIR/logs_$TAG.interrupted.log; log "kept interrupted log $DIR/logs_$TAG.interrupted.log"; }
    log "START $TAG  (exp $XP)"
    if ! run_with_retry $DIR/logs_$TAG.log "\[LEAKAGE-FREE\]" results/preds_${XP}_${MODEL}.npz \
         python3 train.py -a $AGGR --model $MODEL --pooling $POOL -b 20 -r 1 -e 50 --batch 32 -k 10 $MF \
           --data-location data_collection --cache-location ready_data --cache --seed $S \
           --comment commitonly --benign-commit-only $AF; then
      log "FAIL  $TAG training failed twice"; touch $DIR/FAILED_$TAG; return 1
    fi
    if ! grep -q "Loading Dataset $EXP" $DIR/logs_$TAG.log; then
      log "FAIL  $TAG: train.py did not load the prebuilt cache $EXP (see log); result not used"; touch $DIR/FAILED_$TAG; return 1; fi
    cp results/preds_${XP}_${MODEL}.npz $DIR/preds_$TAG.npz
    cp results/${XP}_${MODEL}.txt $DIR/results_$TAG.txt 2>/dev/null
    mv fold_accuracies_${XP}_${MODEL}_*.json $DIR/ 2>/dev/null
    local NF=$(ls models/$XP/${MODEL}_folds/fold_*.keras 2>/dev/null | wc -l)
    log "DONE  train $TAG in ${LAST_DT}s, fold models=$NF | $(grep -h '\[LEAKAGE-FREE\]' $DIR/logs_$TAG.log | sed 's/.*CRITICAL //')"
  fi
  [ -n "$G" ] && return 0
  for KIND in near unseen; do
    [ -f $DIR/ext_${KIND}_$TAG.npz ] && continue
    can_start $KIND || { pause "ext $KIND $TAG"; return 3; }
    local XF=""; [ $KIND = unseen ] && XF="--exclude-train-overlap"
    [ -f $DIR/ext_${KIND}_$TAG.log ] && mv $DIR/ext_${KIND}_$TAG.log $DIR/ext_${KIND}_$TAG.interrupted.log
    if run_with_retry $DIR/ext_${KIND}_$TAG.log "^\[RESULT\]" $DIR/ext_${KIND}_$TAG.npz \
         python3 -u evaluate_cvefixes.py -a $AGGR --model $MODEL --backs 20 $EF --seed $S --enriched $XF \
           --benign-commit-only --model-comment commitonly \
           --tau-from-npz $DIR/preds_$TAG.npz --save-preds $DIR/ext_${KIND}_$TAG.npz; then
      log "DONE  ext $KIND $TAG in ${LAST_DT}s | $(grep -h '^\[COMMIT-LIST\]' $DIR/ext_${KIND}_$TAG.log) | $(grep -h '^\[RESULT\]' $DIR/ext_${KIND}_$TAG.log)"
    else log "FAIL  ext $KIND $TAG failed twice"; fi
  done
  return 0
}

# run a list of configurations in priority order, each as soon as its cache is ready; returns 3 if paused
run_queue() {  # args: "DIR MODEL WIN SEED META [G]" strings
  while true; do
    local waiting=0 ran=0
    for C in "$@"; do
      run_config $C; local RC=$?
      [ $RC = 3 ] && return 3
      [ $RC = 2 ] && { waiting=1; continue; }
    done
    [ $waiting = 0 ] && return 0
    [ -f $R/PAUSED ] && return 3
    sleep 60
  done
}

SEEDS="42 123 7 256 999"
need() { [ -f $1 ] || { log "FAIL  $1 missing; stopping before this task"; touch $R/STOPPED_MISSING_SCRIPT; exit 1; }; }
# ---------------- Task 1 (priority 1) ----------------
Q=""
for S in $SEEDS; do for M in transformer conv1d; do for W in sym causal; do Q="$Q|$R/task1_b20_nometa $M $W $S 0"; done; done
  for M in gru bilstm; do Q="$Q|$R/task1_b20_nometa $M causal $S 0"; done; done
for S in $SEEDS; do for M in transformer conv1d; do for W in sym causal; do Q="$Q|$R/task1_b20_meta $M $W $S 1"; done; done; done
IFS='|' read -ra T1 <<< "${Q#|}"
log "=== Task 1: ${#T1[@]} configurations (train + near + unseen), no metadata first ==="
run_queue "${T1[@]}"; [ $? = 3 ] && exit 0
log "=== Task 1 finished ==="

# ---------------- Task 2: ablation column lists (verifier, CPU) + 30 ablation runs ----------------
mkdir -p $R/03_ablation_columns
for S in $SEEDS; do for W in sym causal; do
  EXP=$(expname $(aggname $W) 0 $S); T=$R/03_ablation_columns/columns_$EXP.txt
  grep -q "VERDICT: VERIFIED" $T 2>/dev/null && continue
  can_start cols || { pause "column list $EXP"; exit 0; }
  nice -n 5 python3 runs_v2/03_feature_check/reconstruct_and_verify_columns.py ready_data/$EXP $(aggflag $W) 20 0 $T > ${T%.txt}.run.log 2>&1
  log "COLS  $EXP: $(grep VERDICT $T 2>/dev/null || echo 'NO VERDICT, see run log')"
done; done
Q=""
for S in $SEEDS; do for W in sym causal; do for G in no_new no_graphql no_temporal; do Q="$Q|$R/task2_ablation_nometa transformer $W $S 0 $G"; done; done; done
IFS='|' read -ra T2 <<< "${Q#|}"
log "=== Task 2: ${#T2[@]} ablation runs ==="
run_queue "${T2[@]}"; [ $? = 3 ] && exit 0
log "=== Task 2 finished ==="

# ---------------- Task 4: full corpus, Commit events only (chunked, as Phase 3 v9) ----------------
need $R/task4_fullcorpus.sh; source $R/task4_fullcorpus.sh
for M in transformer conv1d; do fullcorpus_commit $M || { [ -f $R/PAUSED ] && exit 0; }; done
log "=== Task 4 finished ==="

# ---------------- Task 3: 20 benign redraws (scoring only), seed-42 models first ----------------
need $R/task3_draws.sh; source $R/task3_draws.sh
run_draws; [ -f $R/PAUSED ] && exit 0
log "=== Task 3 finished ==="

# ---------------- Task 5: rebuild one Phase 3 cache from scratch and compare ----------------
need $R/task5_rebuild.sh; source $R/task5_rebuild.sh
task5_rebuild; [ -f $R/PAUSED ] && exit 0
log "=== rerun_v3 queue finished ==="; touch $R/RERUN_V3_DONE
wait
