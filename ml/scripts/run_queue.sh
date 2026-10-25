#!/usr/bin/env bash
# Serial training queue (PLAN.md §6.2). From ml/:
#
#   nohup bash scripts/run_queue.sh configs/queue_m4.txt >/dev/null 2>&1 &
#
# * One run at a time, under `caffeinate -is`; everything is logged to runs/queue.log, each run's output
#   to runs/<run>_train.log and its evaluation to runs/<run>_eval.log.
# * Before each run it waits while any other `ans.train` process is alive (e.g. hand-started runs).
# * Skips runs whose checkpoints/<run>/meta.json says completed; resumes interrupted runs (--resume).
# * Prerequisites (teacher checkpoint of a *completed* teacher run for KD runs, noise/RIR manifests for
#   augmentation runs) are checked by `python -m ans.queue check`; a run whose prerequisite is missing is
#   deferred to the end of the queue once, then skipped with the reason logged.
# * A run that exits non-zero (sleep, crash, kill) is retried once with --resume, then skipped.
# * After each completed run, evaluates it on CPU in the background (vbd + whichever OOD sets exist,
#   --merge into ml/exports/results.json and web/public/results.json) while the next run trains.
# * Battery: starts scripts/power_watchdog.sh (pauses training on battery) and does not start a run or
#   an evaluation while on battery. ALLOW_BATTERY=1 disables both.
#
# Queue file: one config per line, optional extra ans.train args after it; '#' starts a comment.
# Stop:   kill "$(cat runs/queue.pid)"   (stops the queue and its current run; resumable later)
#         touch runs/queue.stop          (lets the current run finish, then stops)
# Env:    QUEUE_POLL_SECONDS (60), QUEUE_SKIP_EVAL=1, QUEUE_EVAL_EXTRA="...", ALLOW_BATTERY=1

main() {
  set -u
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=power_lib.sh
  source "$here/power_lib.sh"
  cd "$ML_DIR" || exit 1
  LOG_TAG=queue
  local queue_file="${1:?usage: run_queue.sh <queue file>}"
  [[ -f "$queue_file" ]] || { echo "no such queue file: $queue_file" >&2; exit 1; }

  if [[ -z "${ANS_QUEUE_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
    export ANS_QUEUE_CAFFEINATED=1
    exec caffeinate -is bash "$here/run_queue.sh" "$@"
  fi

  mkdir -p runs
  local pidfile="${QUEUE_PIDFILE:-runs/queue.pid}"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null && [[ "$(cat "$pidfile")" != "$$" ]]; then
    echo "queue already running (pid $(cat "$pidfile"))" >&2
    exit 1
  fi
  echo $$ >"$pidfile"
  rm -f runs/queue.stop

  CURRENT_PID=""
  EVAL_PID=""
  trap 'on_signal' INT TERM HUP

  ensure_watchdog

  # Read the queue (bash 3.2: no mapfile / associative arrays).
  local items=() line
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -n "$line" ]] && items+=("$line")
  done <"$queue_file"
  log "queue started (pid $$, ${#items[@]} entries from $queue_file)"

  local deferred=" " i=0
  while [[ $i -lt ${#items[@]} ]]; do
    local entry="${items[$i]}"
    i=$((i + 1))
    if [[ -f runs/queue.stop ]]; then
      log "runs/queue.stop found: stopping before $entry"
      break
    fi
    # shellcheck disable=SC2086
    set -- $entry
    local cfg="$1"
    shift
    local extra="$*"

    wait_for_other_training
    local status run reason
    IFS=$'\t' read -r status run reason < <(uv run --quiet python -m ans.queue check "$cfg" 2>>runs/queue_errors.log)
    case "$status" in
      done)
        log "skip $run: already completed"
        continue
        ;;
      missing)
        if [[ "$deferred" != *" $entry "* ]]; then
          deferred="$deferred$entry "
          items+=("$entry")
          log "defer $run to the end of the queue: $reason"
        else
          log "SKIP $run: prerequisite still missing: $reason"
        fi
        continue
        ;;
      resume | fresh) ;;
      *)
        log "SKIP $cfg: could not check status (see runs/queue_errors.log)"
        continue
        ;;
    esac

    local attempt=0 rc
    while :; do
      wait_for_ac "run $run"
      wait_for_other_training
      local args=(--config "$cfg")
      [[ "$status" == resume ]] && args+=(--resume)
      # shellcheck disable=SC2206
      [[ -n "$extra" ]] && args+=($extra)
      log "start $run ($status, attempt $((attempt + 1))): ans.train ${args[*]}"
      local t0=$SECONDS
      uv run --quiet python -m ans.train "${args[@]}" >>"runs/${run}_train.log" 2>&1 &
      CURRENT_PID=$!
      while kill -0 "$CURRENT_PID" 2>/dev/null; do sleep 10; done
      wait "$CURRENT_PID"
      rc=$?
      CURRENT_PID=""
      IFS=$'\t' read -r status run reason < <(uv run --quiet python -m ans.queue check "$cfg" 2>>runs/queue_errors.log)
      if [[ $rc -eq 0 && "$status" == done ]]; then
        log "finished $run in $(((SECONDS - t0) / 60)) min"
        start_eval "$cfg" "$run"
        break
      fi
      local why="exit code $rc"
      [[ $rc -gt 128 ]] && why="$why (signal $((rc - 128)))"
      [[ $rc -eq 0 ]] && why="exit 0 but meta.json not completed"
      if [[ $attempt -eq 0 ]]; then
        attempt=1
        [[ -f "checkpoints/$run/last.pt" ]] && status=resume || status=fresh
        log "run $run failed: $why; retrying once ($status) - see runs/${run}_train.log"
        continue
      fi
      log "FAILED $run twice ($why); moving on - see runs/${run}_train.log"
      break
    done
  done

  if [[ -n "$EVAL_PID" ]]; then
    log "waiting for the last evaluation (pid $EVAL_PID)"
    wait "$EVAL_PID" 2>/dev/null
  fi
  log "queue finished"
  rm -f "$pidfile"
}


ensure_watchdog() {
  [[ "${ALLOW_BATTERY:-0}" == "1" ]] && { log "ALLOW_BATTERY=1: battery watchdog not started"; return; }
  local wpid="runs/power_watchdog.pid"
  if [[ -f "$wpid" ]] && kill -0 "$(cat "$wpid")" 2>/dev/null; then
    log "battery watchdog already running (pid $(cat "$wpid"))"
    return
  fi
  nohup bash scripts/power_watchdog.sh >/dev/null 2>&1 &
  log "started battery watchdog (pid $!)"
}

# Waits until no ans.train process (any wrapper included) has been seen for 3 consecutive polls, so the
# gap between two chained hand-started runs (crn64 -> crn128) is not mistaken for "idle".
wait_for_other_training() {
  local quiet=0 logged=0 poll="${QUEUE_POLL_SECONDS:-60}"
  while :; do
    local others
    others="$(pgrep -f "$ANS_WATCH_PATTERN" 2>/dev/null | tr '\n' ' ')"
    if [[ -n "${others// /}" ]]; then
      if [[ $logged -eq 0 ]]; then
        log "waiting: other ans.train processes alive (pids $others)"
        logged=1
      fi
      quiet=0
    else
      quiet=$((quiet + 1))
      [[ $quiet -ge 3 || ($logged -eq 0 && $quiet -ge 2) ]] && break
    fi
    if [[ $logged -eq 0 ]]; then sleep 10; else sleep "$poll"; fi
  done
  [[ $logged -eq 1 ]] && log "no other ans.train process; continuing"
  return 0
}

start_eval() {
  local cfg="$1" run="$2"
  [[ "${QUEUE_SKIP_EVAL:-0}" == "1" ]] && { log "QUEUE_SKIP_EVAL=1: not evaluating $run"; return; }
  if [[ -n "$EVAL_PID" ]] && kill -0 "$EVAL_PID" 2>/dev/null; then
    log "waiting for the previous evaluation (pid $EVAL_PID) before evaluating $run"
    wait "$EVAL_PID" 2>/dev/null
  fi
  wait_for_ac "evaluation of $run"
  local eargs
  eargs="$(uv run --quiet python -m ans.queue evalargs "$cfg" 2>>runs/queue_errors.log)"
  log "evaluate $run: $eargs (CPU, background)"
  (
    # shellcheck disable=SC2086
    uv run --quiet python -m ans.evaluate $eargs --device cpu --workers 4 --merge --decompose ${QUEUE_EVAL_EXTRA:-} \
      >>"runs/${run}_eval.log" 2>&1
    log "evaluation of $run finished (exit $?) - see runs/${run}_eval.log"
  ) &
  EVAL_PID=$!
}

kill_tree() {
  local pid="$1" d
  for d in $(descendants "$pid"); do kill -TERM "$d" 2>/dev/null; done
  kill -TERM "$pid" 2>/dev/null
}

on_signal() {
  trap - INT TERM HUP
  log "queue received a stop signal"
  if [[ -n "$CURRENT_PID" ]]; then
    # resume the tree first in case the watchdog paused it, then terminate it (last.pt allows --resume)
    for d in $(descendants "$CURRENT_PID"); do kill -CONT "$d" 2>/dev/null; done
    kill_tree "$CURRENT_PID"
    log "terminated the current run (pid $CURRENT_PID); it will resume from last.pt next time"
  fi
  rm -f "${QUEUE_PIDFILE:-runs/queue.pid}"
  exit 143
}

main "$@"
exit $?
