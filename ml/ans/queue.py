"""Helpers for ``scripts/run_queue.sh`` (PLAN.md §6.2).

uv run python -m ans.queue check configs/crn128_kd.yaml     # -> "<status>\\t<run>\\t<reason>"
uv run python -m ans.queue evalargs configs/crn128_kd.yaml  # -> evaluate.py arguments for this run

Status: ``done`` (meta.json completed), ``resume`` (last.pt exists), ``fresh``, or ``missing`` (a
prerequisite is absent: the teacher checkpoint of a KD run must exist *and* its run must be completed;
augmentation runs need their noise/RIR manifests).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ans.data import ML_ROOT
from ans.runmeta import read_meta


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ML_ROOT / p


def check(config: str, ckpt_root: Path | None = None) -> tuple[str, str, str]:
    from ans.train import load_config, resolve_sections

    cfg = resolve_sections(load_config(resolve(config)))
    run = cfg["run_name"]
    run_dir = (ckpt_root or ML_ROOT / "checkpoints") / run
    meta = read_meta(run_dir)
    if meta and meta.get("completed"):
        return "done", run, "meta.json completed"
    missing = []
    teacher = (cfg.get("distill") or {}).get("teacher_ckpt")
    if teacher:
        tp = resolve(teacher)
        tmeta = read_meta(tp.parent)
        if not tp.exists():
            missing.append(f"teacher checkpoint {teacher} not found")
        elif not (tmeta and tmeta.get("completed")):
            missing.append(f"teacher run {tp.parent.name} not completed")
    aug = (cfg.get("data") or {}).get("augment")
    if aug:
        from ans.augment import AugmentConfig, find_data_manifest

        for key, name in AugmentConfig.from_dict(aug).manifests().items():
            if find_data_manifest(name) is None:
                missing.append(f"{key} manifest {name}.csv not found")
    if missing:
        return "missing", run, "; ".join(missing)
    if (run_dir / "last.pt").exists():
        return "resume", run, "last.pt exists"
    return "fresh", run, ""


def eval_args(config: str) -> list[str]:
    from ans.testsets import available_sets
    from ans.train import load_config

    cfg = load_config(resolve(config))
    sets = available_sets()
    return ["--sets", *sets, "--models", f"crn:checkpoints/{cfg['run_name']}/best.pt"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "evalargs"])
    ap.add_argument("config")
    args = ap.parse_args()
    if args.cmd == "check":
        print("\t".join(check(args.config)))
    else:
        print(" ".join(eval_args(args.config)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
