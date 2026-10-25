# Shared helpers for run_queue.sh and power_watchdog.sh (sourced, not executed).
#
#   ALLOW_BATTERY=1          never pause / never wait for AC
#   ANS_FAKE_POWER=ac|battery            stub the power source (tests)
#   ANS_FAKE_POWER_FILE=<file>           same, read from a file on every check (can change at runtime)
#   ANS_WATCH_PATTERN=<regex>            processes to pause (default: python processes running ans.train)

ML_DIR="${ML_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
QUEUE_LOG="${QUEUE_LOG:-$ML_DIR/runs/queue.log}"
ANS_WATCH_PATTERN="${ANS_WATCH_PATTERN:-ans\.train}"

log() {
  mkdir -p "$(dirname "$QUEUE_LOG")"
  printf '%s [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${LOG_TAG:-queue}" "$*" >>"$QUEUE_LOG"
}

# Prints "ac" or "battery". Unknown / no battery -> "ac".
power_source() {
  if [[ -n "${ANS_FAKE_POWER_FILE:-}" && -f "$ANS_FAKE_POWER_FILE" ]]; then
    tr -d '[:space:]' <"$ANS_FAKE_POWER_FILE"
    return
  fi
  if [[ -n "${ANS_FAKE_POWER:-}" ]]; then
    echo "$ANS_FAKE_POWER"
    return
  fi
  if command -v pmset >/dev/null 2>&1 && pmset -g batt 2>/dev/null | head -1 | grep -q "Battery Power"; then
    echo battery
  else
    echo ac
  fi
}

on_battery() {
  [[ "${ALLOW_BATTERY:-0}" != "1" && "$(power_source)" == "battery" ]]
}

# Blocks until on AC power (or ALLOW_BATTERY=1). $1 = what is waiting (for the log).
wait_for_ac() {
  local what="$1" logged=0
  while on_battery; do
    if [[ $logged == 0 ]]; then
      log "on battery: not starting $what until AC power returns"
      logged=1
    fi
    sleep "${POWER_POLL_SECONDS:-30}"
  done
  [[ $logged == 1 ]] && log "AC power back: starting $what"
  return 0
}

# All descendants of a pid (depth first, children before grandchildren).
descendants() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    echo "$child"
    descendants "$child"
  done
}

# Main training processes: python interpreters whose command line matches the pattern
# (skips the bash/uv/caffeinate wrappers whose command lines also contain it).
trainer_pids() {
  local pid comm
  for pid in $(pgrep -f "$ANS_WATCH_PATTERN" 2>/dev/null); do
    [[ "$pid" == "$$" ]] && continue
    comm="$(ps -o comm= -p "$pid" 2>/dev/null)"
    [[ "$comm" == *python* ]] && echo "$pid"
  done
}

# Is any trainer (other than those in the optional exclude list) alive?
other_trainers_alive() {
  local pid
  for pid in $(trainer_pids); do
    [[ " ${*:-} " == *" $pid "* ]] && continue
    return 0
  done
  return 1
}

# SIGSTOP a trainer then its whole tree (dataloader workers, metric pools).
stop_tree() {
  local pid="$1" d
  kill -STOP "$pid" 2>/dev/null || return 0
  for d in $(descendants "$pid"); do kill -STOP "$d" 2>/dev/null; done
}

# SIGCONT the tree bottom-up first, then the trainer.
cont_tree() {
  local pid="$1" d
  for d in $(descendants "$pid"); do kill -CONT "$d" 2>/dev/null; done
  kill -CONT "$pid" 2>/dev/null || true
}
