#!/bin/bash
# Emergency stop (use only if you must shut down before the planned pause). Stops the runner and its jobs by pid.
# Interrupted runs are redone on resume; their logs are kept as *.interrupted.log. Then: bash rerun_v3/resume.sh
cd "$(dirname "$0")/.."
P=$(pgrep -f "rerun_v3/run_v3.sh"; pgrep -f "rerun_v3/02_caches/(prebuild_cache|split_check).py"; pgrep -f "^python3 (-u )?(train|evaluate_cvefixes)\.py.*commitonly")
[ -n "$P" ] && kill $P 2>/dev/null; sleep 10; [ -n "$P" ] && kill -9 $P 2>/dev/null
echo "[$(date '+%a %F %T')] STOP_NOW: stopped pids $(echo $P | tr '\n' ' ')" | tee -a rerun_v3/00_RERUN_V3_SUMMARY.log
touch rerun_v3/PAUSED
