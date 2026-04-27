#!/usr/bin/env bash
# Standardized experiment runner.
#
#   scripts/run_exp.sh <python_script> [args...]
#
# - Forces unbuffered Python output (`python -u`).
# - Tees stdout+stderr to experiments/_scratch_logs/<basename>_<ts>.log.
# - Background-friendly: prints the log path on first line so a Monitor can
#   tail it. Also writes the PID to <log>.pid for clean kill.
# - On exit, prints a summary with elapsed time and exit code.
#
# Recommended Monitor usage:
#   Monitor command: until [ ! -d /proc/$(cat <log>.pid) ] 2>/dev/null;
#                    do sleep 30; done; tail -50 <log>
set -e
if [ -z "$1" ]; then
  echo "usage: $0 <python_script> [args...]"; exit 2
fi
SCRIPT="$1"; shift
[ -f "$SCRIPT" ] || { echo "not found: $SCRIPT"; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/experiments/_scratch_logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
NAME="$(basename "$SCRIPT" .py)"
LOG="$LOGDIR/${NAME}_${TS}.log"

echo "log: $LOG"
echo "$$ start $(date -Iseconds)" > "$LOG.pid"

START=$(date +%s)
{
  echo "==== $NAME @ $TS ===="
  echo "cmd: uv run python -u $SCRIPT $*"
  uv run python -u "$SCRIPT" "$@"
  RC=$?
  END=$(date +%s)
  echo "==== exit $RC after $((END - START))s ===="
  exit $RC
} 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
rm -f "$LOG.pid"
exit $RC
