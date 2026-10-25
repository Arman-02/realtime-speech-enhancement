# Training runbook

The training queue runs on its own, independently of any Claude session. This page shows you how to check on it, pause it, stop it and restart it. Run every command from `ml/` (`cd ~/Documents/Desktop2/ActiveNoiseSuppression/ml`).

## Check status
```bash
bash scripts/status.sh                         # one-screen overview: power, processes, progress, finished runs
tail -f runs/queue.log                         # live queue events (run start/finish, battery pause/resume, retries)
tail -f runs/<run>_train.log | tr '\r' '\n'    # live progress bar of one run
uv run tensorboard --logdir runs               # graphs at http://localhost:6006; watch val/pesq
```
`status.sh` shows `ans.train: PAUSED (on battery)` when the battery watchdog has frozen the current run. Plugging the Mac back in resumes it automatically.

## What the queue runs
The queue runs `configs/queue_m4.txt` in order, one run at a time. A finished run is skipped, and an interrupted run resumes from its last epoch. Each run takes about 2.5–5 h. Runs that need the teacher wait for it to finish.

| Run | Config | What it is |
|---|---|---|
| R1 | `teacher_bigru512` | Non-causal teacher for distillation |
| R4–R6 | `crn64_kd`, `crn128_kd`, `crn256_kd` | Distilled students |
| R7, R8 | `crn128_seed1`, `crn128_kd_seed1` | Second seed (for significance tests) |
| R9 | `crn128_kd_b05` | Distillation with β = 0.5 |
| R10, R11 | `crn256_la1`, `crn256_la2` | Lookahead of 10 ms and 20 ms |
| R12–R14 | `crn256_*_50ep` | Augmentation ablations |

Checkpoints are in `checkpoints/<run>/`, logs in `runs/<run>_train.log`, and results are merged into `exports/results.json` and `../web/public/results.json`.

## Battery and sleep
- The queue keeps the Mac awake with `caffeinate`, but **only on AC power**.
- On battery, the watchdog pauses training (it is frozen, not killed) and won't start new runs or evaluations. To override, set `ALLOW_BATTERY=1` before starting the queue.
- With the lid closed, the Mac may still sleep unless it is on power with an external display. After sleep, training continues on wake. If it crashed, the queue retries it once from the last checkpoint.

## Stop or pause
```bash
touch runs/queue.stop            # graceful: let the current run finish, then stop
kill "$(cat runs/queue.pid)"     # immediate: stop the queue and the current run (resumable)
```

## After a reboot, or if status.sh says the queue is NOT running
```bash
rm -f runs/queue.stop
nohup bash scripts/run_queue.sh configs/queue_m4.txt >/dev/null 2>&1 &
bash scripts/status.sh           # confirm: run_queue running, power_watchdog running
```
Completed runs are skipped and the interrupted run resumes, so restarting is always safe.

## Run a single config by hand (only while the queue is stopped; one GPU job at a time)
```bash
uv run python -m ans.train --config configs/crn128.yaml --resume   # e.g. finish crn128 if it was interrupted
```

## Check that a result is healthy
- `val/pesq` in TensorBoard should rise quickly and then plateau.
- Reference best val PESQ values (the val set is harder than test): crn_small (256) 2.52, crn64 2.42.
- Test-set PESQ for crn_small is 2.76.
