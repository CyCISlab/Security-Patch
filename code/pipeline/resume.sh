#!/bin/bash
# Continue rerun_v3 after a pause. Usage: bash rerun_v3/resume.sh ["YYYY-MM-DD HH:MM"]   (optional new pause time)
cd "$(dirname "$0")/.."
P=$(cat rerun_v3/runner.pid 2>/dev/null); if [ -n "$P" ] && [ "$(ps -o args= -p $P 2>/dev/null)" = "bash rerun_v3/run_v3.sh" ]; then echo "runner already running (pid $P)"; exit 1; fi
if [ -n "$1" ]; then echo "$1" > rerun_v3/PAUSE_AT; else rm -f rerun_v3/PAUSE_AT; fi
rm -f rerun_v3/PAUSED
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch --who=rerun_v3 --why="rerun_v3 runs" \
  bash rerun_v3/run_v3.sh >> rerun_v3/runner_stdout.log 2>&1 < /dev/null &
echo "resumed; pause at: $(cat rerun_v3/PAUSE_AT 2>/dev/null || echo none); log: rerun_v3/00_RERUN_V3_SUMMARY.log"
