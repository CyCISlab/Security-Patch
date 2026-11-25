#!/bin/bash
# Waits until all 160 seed-42 Task 3 draws exist (4 models x 20 draws x near/unseen), then makes the running runner
# pause before its next job (PAUSE_AT = now; no job is killed), waits for it to exit, and resumes it. On resume
# task3_draws.sh has T3_SEEDS=42, so Task 3 is complete and Task 5 runs. Anonusr 2026-11-22: "yes" (seed 42 only).
cd "$(dirname "$0")/.."
R=rerun_v3; S=$R/00_RERUN_V3_SUMMARY.log
log() { echo "[$(date '+%a %F %T')] $*" | tee -a $S; }
log "T3-SWITCH: watcher started (pid $$); Task 3 limited to seed-42 models"
until [ $(ls $R/task3_draws/ext_*_s42_nometa_draw*.npz 2>/dev/null | wc -l) -ge 160 ]; do sleep 60; done
date '+%F %T' > $R/PAUSE_AT
log "T3-SWITCH: all 160 seed-42 draws present; PAUSE_AT set to now so the runner stops before the next job"
P=$(cat $R/runner.pid); while ps -p $P > /dev/null 2>&1; do sleep 20; done
log "T3-SWITCH: runner exited; resuming without pause"
bash $R/resume.sh
