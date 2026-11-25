# rerun_v3 Task 4: full corpus restricted to Commit events, Task 1 Transformer and Conv1D, seed 42, sym, no metadata.
# Sourced by run_v3.sh. fc_run copied unchanged from runs_v2/06_phase3/run_task2_transformer_v9.sh (watchdog 20 GB);
# scoring in 36-repo chunk processes + merge, the method that passed runs_v2/12_oom_fix/determinism_check (NOTES s18).
# Both models are chunked here (Phase 3 scored Conv1D in one process), so the two models share one method.
# Only change to the command: --benign-commit-only --model-comment commitonly, tau from the rerun_v3 Task 1 npz.
WD_MAX_RSS_KB=20971520
fc_run() {  # $1 LOG  $2 OUT  $3 success regex in LOG  cmd...   (sets LAST_DT, FC_MAXRSS_MB)
  local L=$1 OUT=$2 RX=$3; shift 3
  for ATT in 1 2; do
    local MARK=$(mktemp) T0=$(date +%s) FLAG=$L.watchdog; rm -f $FLAG; FC_MAXRSS_MB=0
    "$@" > "$L" 2>&1 &
    local PID=$!
    while kill -0 $PID 2>/dev/null; do
      local C=$(grep -c dot_merger "$L" 2>/dev/null) RSSKB=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null)
      [ $(( ${RSSKB:-0} / 1024 )) -gt $FC_MAXRSS_MB ] && FC_MAXRSS_MB=$(( ${RSSKB:-0} / 1024 ))
      if [ "${RSSKB:-0}" -gt $WD_MAX_RSS_KB ]; then
        echo "compiles=$C rss_kb=$RSSKB at $(grep -E '^\s*\[[0-9]+/214\]' "$L" | tail -1)" > $FLAG
        kill $PID; break
      fi
      sleep 20
    done
    wait $PID; local RC=$?
    if [ $RC -eq 0 ] && grep -q "$RX" "$L" && [ "$OUT" -nt "$MARK" ]; then
      rm -f $MARK; LAST_DT=$(( $(date +%s)-T0 )); return 0
    fi
    rm -f $MARK
    if [ -f $FLAG ]; then
      log "    attempt $ATT STOPPED BY WATCHDOG ($(cat $FLAG)); not retried ($L -> ${L%.log}.watchdog.log)"
      mv "$L" "${L%.log}.watchdog.log"; return 1
    fi
    log "    attempt $ATT FAILED rc=$RC ($L)"
    [ $ATT = 1 ] && mv "$L" "${L%.log}.attempt1.log"
  done
  return 1
}
fullcorpus_commit() {  # $1 MODEL ; returns 1 if paused or failed
  local MODEL=$1 TAG=${1}_sym_b20_s42_nometa D=$R/task4_fullcorpus_commits
  local CD=$D/chunks_$TAG NCSV=$(ls data_collection/cvefixes_test/enriched/*.csv | wc -l) CS=36 LIST="" T0=$(date +%s)
  local NPZ=$R/task1_b20_nometa/preds_$TAG.npz
  [ -f $D/fullcorpus_preds_$TAG.npz ] && return 0
  [ -f $NPZ ] || { log "FAIL  task4 $TAG: Task 1 npz $NPZ missing"; return 1; }
  local ARGS="-a before --model $MODEL --backs 20 --seed 42 --enriched --realistic --benign-commit-only --model-comment commitonly --tau-from-npz $NPZ"
  mkdir -p $CD
  log "START task4 fullcorpus commits $TAG ($NCSV repos, chunks of $CS -> $CD/)"
  for ((A=0; A<NCSV; A+=CS)); do
    local B=$((A+CS)); [ $B -gt $NCSV ] && B=$NCSV
    local P=$CD/chunk_${A}_${B}.pkl L=$CD/chunk_${A}_${B}.log
    LIST="${LIST:+$LIST,}$P"
    if [ -f $P ] && grep -q "^\[CHUNK\]" $L 2>/dev/null; then continue; fi
    can_start draw || { pause "task4 chunk $A:$B $TAG"; return 1; }
    [ -f $L ] && mv $L ${L%.log}.interrupted.log
    if ! fc_run $L $P "^\[CHUNK\]" python3 -u evaluate_cvefixes.py $ARGS --repo-range $A:$B --chunk-out $P; then
      log "FAIL  task4 $TAG: chunk $A:$B failed; not merged"; return 1; fi
    log "  chunk $A:$B done in ${LAST_DT}s, peak RSS ${FC_MAXRSS_MB} MB | $(grep -h '^\[CHUNK\]' $L | sed 's/ -> .*//')"
  done
  can_start split || { pause "task4 merge $TAG"; return 1; }
  if fc_run $D/realistic_$TAG.log $D/fullcorpus_preds_$TAG.npz "\[COMMIT-LIST\]" \
       python3 -u evaluate_cvefixes.py $ARGS --merge-chunks $LIST --save-preds $D/fullcorpus_preds_$TAG.npz; then
    log "DONE  task4 fullcorpus commits $TAG in $(( $(date +%s)-T0 ))s | $(grep -h '^\[MERGE\]\|Total:\|\[COMMIT-LIST\]' $D/realistic_$TAG.log | tr '\n' ' ')"
    python3 operational_metrics.py --preds $D/fullcorpus_preds_$TAG.npz --repos 212 --days 365 \
      --budgets 100 500 1000 5000 --tie-seed 0 --out-prefix $D/${TAG}_ > $D/operational_metrics_$TAG.txt 2>&1 \
      && log "DONE  task4 operational metrics $TAG -> $D/operational_metrics_$TAG.txt" \
      || log "FAIL  task4 operational metrics $TAG (see $D/operational_metrics_$TAG.txt)"
  else log "FAIL  task4 $TAG: merge failed"; return 1; fi
}
