#!/usr/bin/env bash
# Battery watchdog: pauses (SIGSTOP) every ans.train process tree while the Mac runs on battery and
# resumes it (SIGCONT) when AC power returns. Covers queue runs and runs started by hand.
#
#   nohup bash scripts/power_watchdog.sh >/dev/null 2>&1 &     # from ml/
#   kill "$(cat runs/power_watchdog.pid)"                      # stop (resumes anything it paused)
#   ALLOW_BATTERY=1 ...                                         # disable pausing
#
# Logs to runs/queue.log. Single instance via runs/power_watchdog.pid.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/power_lib.sh"
LOG_TAG=power
INTERVAL="${WATCHDOG_INTERVAL:-30}"
PIDFILE="${WATCHDOG_PIDFILE:-$ML_DIR/runs/power_watchdog.pid}"

mkdir -p "$(dirname "$PIDFILE")"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && [[ "$(cat "$PIDFILE")" != "$$" ]]; then
  echo "watchdog already running (pid $(cat "$PIDFILE"))"
  exit 0
fi
echo $$ >"$PIDFILE"

paused=""   # pids we stopped

resume_all() {
  local pid
  for pid in $paused; do
    cont_tree "$pid"
  done
  [[ -n "$paused" ]] && log "resumed pids:$paused"
  paused=""
}

cleanup() {
  trap - INT TERM HUP EXIT
  resume_all
  [[ "$(cat "$PIDFILE" 2>/dev/null)" == "$$" ]] && rm -f "$PIDFILE"
  log "watchdog stopped (pid $$)"
  exit 0
}
trap cleanup INT TERM HUP EXIT

log "watchdog started (pid $$, interval ${INTERVAL}s, ALLOW_BATTERY=${ALLOW_BATTERY:-0})"
state=ac
while true; do
  if on_battery; then
    new=""
    for pid in $(trainer_pids); do
      stop_tree "$pid"   # idempotent; also catches workers spawned since the last check
      [[ " $paused " == *" $pid "* ]] || new="$new $pid"
    done
    if [[ -n "$new" ]]; then
      paused="$paused$new"
      log "paused (on battery): pids$new"
    elif [[ $state == ac ]]; then
      log "on battery: no training process to pause"
    fi
    state=battery
  else
    if [[ $state == battery ]]; then
      if [[ -n "$paused" ]]; then resume_all; fi
      log "resumed (on AC)"
    fi
    paused=""
    state=ac
  fi
  sleep "$INTERVAL" &
  wait $!
done
