# rerun_v3 Task 5 rebuild check (CPU). Sourced by run_v3.sh.
task5_rebuild() {
  [ -f $R/05_task5/rebuild_check.txt ] && return
  can_start cache || { pause "task5 rebuild"; return; }
  [ -d ready_data/Aggregate.before_cve_R1_B20_rebuildcheck ] && mv ready_data/Aggregate.before_cve_R1_B20_rebuildcheck $R/02_caches/incomplete/rebuildcheck.$(date +%s)
  mkdir -p $R/05_task5; log "START task5 from-scratch rebuild of Aggregate.before_cve_R1_B20"
  CUDA_VISIBLE_DEVICES= nice -n 5 python3 $R/task5_rebuild.py > $R/05_task5/rebuild_check.run.log 2>&1
  log "DONE  task5 | $(tr '\n' ' ' < $R/05_task5/rebuild_check.txt 2>/dev/null || echo 'no output, see run log')"
}
