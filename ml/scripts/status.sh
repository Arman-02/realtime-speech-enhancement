#!/usr/bin/env bash
# One-shot overview of the training queue: processes, power, current run, finished runs.
# Usage (from anywhere): bash ml/scripts/status.sh
cd "$(dirname "$0")/.." || exit 1

echo "== Power"
pmset -g batt | head -2 | tail -1

echo
echo "== Processes"
for name in run_queue power_watchdog; do
  pid=$(pgrep -f "$name.sh" | head -1)
  [[ -n $pid ]] && echo "  $name: running (pid $pid)" || echo "  $name: NOT running"
done
train=$(pgrep -f "python.* -m ans.train" | head -1)
if [[ -n $train ]]; then
  state=$(ps -o state= -p "$train" | tr -d ' ')
  [[ $state == T* ]] && note="PAUSED (on battery)" || note="training"
  echo "  ans.train: $note (pid $train) $(ps -o args= -p "$train" | grep -oE 'configs/[^ ]+|--run-name [^ ]+' | tr '\n' ' ')"
else
  echo "  ans.train: none running"
fi

echo
echo "== Current progress"
latest=$(ls -t runs/*_train.log 2>/dev/null | head -1)
[[ -n $latest ]] && echo "  $(basename "$latest" _train.log): $(tr '\r' '\n' < "$latest" | grep -E 'epoch [0-9]+:|done:' | tail -1 | cut -c1-110)"

echo
echo "== Finished runs (best val metrics)"
for log in runs/*_train.log; do
  line=$(grep -h "^done:" "$log" 2>/dev/null | tail -1)
  [[ -n $line ]] && echo "  $(basename "$log" _train.log): $(echo "$line" | grep -oE "best=\{[^}]*\}")"
done

echo
echo "== Queue log (last 8 lines)"
tail -8 runs/queue.log 2>/dev/null | sed 's/^/  /'
