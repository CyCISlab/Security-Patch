# rerun_v3 Task 3: 20 further Commit-only benign draws for near-domain and unseen, scoring only (no training).
# Sourced by run_v3.sh. Same evaluate_cvefixes.py command as Task 1 external scoring, only --sample-seed changes
# (draw seeds 1..20; draw 0 is the Task 1 result itself). Models: Task 1 Transformer and Conv1D, sym and causal,
# no metadata; seed-42 models first, then the other seeds as time allows (the brief does not name a seed).
T3_SEEDS=${T3_SEEDS:-"42"}   # Anonusr 2026-11-22: seed-42 models only (supervisor named no seed; all seeds ~60 h more, past the deadline)
run_draws() {
  local D=$R/task3_draws; mkdir -p $D
  for S in $T3_SEEDS; do for M in transformer conv1d; do for W in sym causal; do
    local AGGR=$(aggflag $W) TAG=${M}_${W}_b20_s${S}_nometa NPZ=$R/task1_b20_nometa/preds_${M}_${W}_b20_s${S}_nometa.npz
    [ -f $NPZ ] || { log "FAIL  task3 $TAG: Task 1 npz missing"; continue; }
    for DS in $(seq 1 20); do for KIND in near unseen; do
      local O=$D/ext_${KIND}_${TAG}_draw$DS
      [ -f $O.npz ] && continue
      can_start $KIND || { pause "task3 $KIND $TAG draw $DS"; return; }
      local XF=""; [ $KIND = unseen ] && XF="--exclude-train-overlap"
      [ -f $O.log ] && mv $O.log $O.interrupted.log
      if run_with_retry $O.log "^\[RESULT\]" $O.npz python3 -u evaluate_cvefixes.py -a $AGGR --model $M --backs 20 \
           --seed $S --enriched $XF --benign-commit-only --model-comment commitonly --sample-seed $DS \
           --tau-from-npz $NPZ --save-preds $O.npz; then
        log "DONE  task3 $KIND $TAG draw $DS in ${LAST_DT}s | $(grep -h '^\[COMMIT-LIST\]' $O.log) | $(grep -h '^\[RESULT\]' $O.log)"
      else log "FAIL  task3 $KIND $TAG draw $DS failed twice"; fi
    done; done
  done; done; done
}
