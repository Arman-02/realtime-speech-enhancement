"""Training entry point (PLAN.md §4.5).

uv run python -m ans.train --config configs/crn_small.yaml [key.sub=value ...]
uv run python -m ans.train --config configs/crn_small.yaml --synthetic --max-steps 200   # smoke run
uv run python -m ans.train --config configs/crn_small.yaml --resume                        # from last.pt
uv run python -m ans.train --config configs/crn128_kd.yaml                                  # distillation

Optional config sections (all off by default; the default VB-D path is unchanged):
  distill: {teacher_ckpt, beta, gamma}          knowledge distillation (ans/distill.py)
  data.augment: {extra_noise, reverb, mic, level} augmentation (ans/augment.py)
  model.kwargs.lookahead: k                     lookahead-k student (§4.11)
Every run writes checkpoints/<run>/meta.json (completed: true once training finishes).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ans.data import (
    ML_ROOT,
    LoaderConfig,
    MixConfig,
    SyntheticDataset,
    VBDDataset,
    collate,
    make_loader,
    manifest_path,
)
from ans.losses import EnhancementLoss, LossConfig
from ans.metrics import compute_metrics_batch, mean_ci
from ans.models import build_model, count_params

DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "crn_small",
    "seed": 0,
    "device": "auto",
    "model": {"name": "ans_crn", "kwargs": {}},
    "data": {
        "manifests": "data/vbd16k/manifests",
        "segment": 32000,
        "remix_prob": 0.5,
        "snr_range": [-5.0, 20.0],
        "gain_prob": 0.5,
        "gain_dbfs": [-25.0, -5.0],
        "extra_noise_dir": None,
        "num_workers": None,
        "synthetic_items": 2048,
    },
    "loss": {"spec_weight": 1.0, "compress": 0.3, "alpha": 0.3, "mrstft_weight": 0.0, "sisdr_weight": 0.0},
    "optim": {"lr": 1e-3, "weight_decay": 1e-5, "betas": [0.9, 0.999], "grad_clip": 5.0, "ema_decay": 0.999},
    "schedule": {"epochs": 100, "warmup_epochs": 2, "min_lr_ratio": 0.01},
    "batch_size": 32,
    "val": {"every_epochs": 1, "metrics_every_epochs": 2, "subset": 200, "batch_size": 8, "workers": None},
    "log_every": 25,
    "max_steps": None,
    "display_name": None,
    "distill": None,
}


# ------------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------------


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_override(cfg: dict, override: str) -> None:
    """``a.b.c=value`` (value parsed as YAML)."""
    key, _, raw = override.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got {override!r}")
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        if not isinstance(node.get(p), dict):
            node[p] = {}
        node = node[p]
    node[leaf] = yaml.safe_load(raw)


def read_yaml_config(path: str | Path) -> dict:
    """A YAML config; ``base: other.yaml`` (relative to this file) is loaded first and updated."""
    path = Path(path)
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    base = d.pop("base", None)
    return deep_update(read_yaml_config(path.parent / base), d) if base else d


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        cfg = deep_update(cfg, read_yaml_config(path))
    for o in overrides or []:
        apply_override(cfg, o)
    return cfg


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------------------------------------


class EMA:
    """Exponential moving average of model weights (with a short warmup on the decay)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, p in zip(self.model.parameters(), model.parameters(), strict=True):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers(), strict=True):
            e.copy_(b)


def lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def train_step(
    model: nn.Module,
    batch: dict,
    loss_fn: EnhancementLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = 5.0,
    amp: bool = False,
    distiller=None,
) -> tuple[float, dict[str, float], float]:
    """One optimisation step (bf16 autocast if ``amp``, CUDA only); returns (loss, terms, grad norm).

    With a :class:`ans.distill.Distiller` the loss is the §4.4 distillation loss."""
    model.train()
    noisy = batch["noisy"].to(device, non_blocking=True)
    clean = batch["clean"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    if distiller is None:
        with torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext():
            est = model.enhance(noisy)
        loss, terms = loss_fn(est.float(), clean, mask)
        params = model.parameters()
    else:
        loss, terms = distiller(model, noisy, clean, mask, loss_fn)
        params = [*model.parameters(), *distiller.trainable_parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip or float("inf"))
    optimizer.step()
    return float(loss.detach()), terms, float(gnorm)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    dataset,
    loss_fn: EnhancementLoss,
    device: torch.device,
    batch_size: int = 8,
    metrics: bool = True,
    workers: int | None = None,
) -> dict[str, float]:
    """Val loss over ``dataset`` (full utterances) and optionally PESQ/STOI/SI-SDR."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    losses, pairs = [], []
    for batch in loader:
        noisy, clean, mask = (batch[k].to(device) for k in ("noisy", "clean", "mask"))
        est = model.enhance(noisy)
        loss, _ = loss_fn(est, clean, mask)
        losses.append(float(loss) * len(noisy))
        if metrics:
            est_np, clean_np = est.cpu().numpy(), batch["clean"].numpy()
            for i, n in enumerate(batch["lengths"].tolist()):
                pairs.append((clean_np[i, :n], est_np[i, :n]))
    out = {"loss": sum(losses) / len(dataset)}
    if metrics and pairs:
        rows = compute_metrics_batch(pairs, workers=workers, which=("pesq", "stoi", "si_sdr"))
        for k in ("pesq", "stoi", "si_sdr"):
            out[k] = mean_ci([r[k] for r in rows])[0]
    return out


def build_datasets(cfg: dict, synthetic: bool):
    d = cfg["data"]
    if synthetic:
        seg = d["segment"]
        return (
            SyntheticDataset(d.get("synthetic_items", 2048), train=True, segment=seg, seed=cfg["seed"]),
            SyntheticDataset(min(cfg["val"]["subset"], 32), train=False, seed=cfg["seed"] + 1),
        )
    mix = MixConfig(
        segment=d["segment"],
        remix_prob=d["remix_prob"],
        snr_range=tuple(d["snr_range"]),
        gain_prob=d["gain_prob"],
        gain_dbfs=tuple(d["gain_dbfs"]),
        extra_noise_dir=d.get("extra_noise_dir"),
    )
    train_csv, val_csv = manifest_path("train", d["manifests"]), manifest_path("val", d["manifests"])
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found — run scripts/prepare_vbd.py or pass --synthetic")
    return (
        VBDDataset(train_csv, train=True, mix=mix, seed=cfg["seed"], augment=d.get("augment")),
        VBDDataset(val_csv, train=False, max_items=cfg["val"]["subset"]),
    )


def run_manifests(cfg: dict, synthetic: bool) -> dict[str, Path | None]:
    """Data manifests (and the teacher checkpoint) a run depends on, for meta.json hashes."""
    if synthetic:
        return {}
    d = cfg["data"]
    out: dict[str, Path | None] = {s: manifest_path(s, d["manifests"]) for s in ("train", "val")}
    if d.get("augment"):
        from ans.augment import AugmentConfig, find_data_manifest

        for key, name in AugmentConfig.from_dict(d["augment"]).manifests().items():
            out[f"augment.{key}"] = find_data_manifest(name)
    if (cfg.get("distill") or {}).get("teacher_ckpt"):
        from ans.distill import resolve_ckpt

        out["teacher_ckpt"] = resolve_ckpt(cfg["distill"]["teacher_ckpt"], ML_ROOT)
    return out


def save_checkpoint(path: Path, **state: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------------------------


def train(
    cfg: dict,
    synthetic: bool = False,
    resume: str | None = None,
    ckpt_root: Path | None = None,
    log_root: Path | None = None,
) -> dict[str, Any]:
    cfg = resolve_sections(cfg)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device(cfg.get("device", "auto"))
    run = cfg["run_name"]
    ckpt_dir = (ckpt_root or ML_ROOT / "checkpoints") / run
    log_dir = (log_root or ML_ROOT / "runs") / run

    train_ds, val_ds = build_datasets(cfg, synthetic)
    loader = make_loader(
        train_ds,
        LoaderConfig(batch_size=cfg["batch_size"], num_workers=cfg["data"].get("num_workers")),
    )
    steps_per_epoch = len(loader)
    epochs = cfg["schedule"]["epochs"]
    max_steps = cfg.get("max_steps")
    total_steps = min(epochs * steps_per_epoch, max_steps or math.inf)
    warmup = min(int(cfg["schedule"]["warmup_epochs"] * steps_per_epoch), int(total_steps) // 10 or 1)

    model_cfg = cfg["model"]
    model = build_model(model_cfg["name"], **model_cfg.get("kwargs", {})).to(device)
    ema = EMA(model, cfg["optim"]["ema_decay"])
    loss_fn = EnhancementLoss(LossConfig(**cfg["loss"])).to(device)
    distiller = None
    if (dcfg := _distill_config(cfg)) is not None:
        from ans.distill import Distiller, resolve_ckpt
        from ans.models import load_checkpoint

        teacher = load_checkpoint(resolve_ckpt(dcfg.teacher_ckpt, ML_ROOT), device)
        distiller = Distiller(teacher, model, dcfg).to(device)
        print(f"distillation: teacher {dcfg.teacher_ckpt} beta={dcfg.beta} gamma={dcfg.gamma}")
    o = cfg["optim"]
    params = model.parameters() if distiller is None else [*model.parameters(), *distiller.trainable_parameters()]
    optimizer = torch.optim.AdamW(params, lr=o["lr"], weight_decay=o["weight_decay"], betas=o["betas"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(warmup, int(total_steps), cfg["schedule"]["min_lr_ratio"])
    )
    amp = device.type == "cuda"

    step, start_epoch, best = 0, 0, -math.inf
    best_metrics: dict | None = None
    if resume:
        path = Path(resume) if resume != "auto" else ckpt_dir / "last.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        ema.model.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        best, best_metrics = ck.get("best_score", -math.inf), ck.get("best_metrics")
        if distiller is not None and distiller.proj is not None and ck.get("distill_proj") is not None:
            distiller.proj.load_state_dict(ck["distill_proj"])
        ema.updates = step
        print(f"resumed from {path} at epoch {start_epoch}, step {step}")

    n_params = count_params(model)
    print(f"device={device} model={model_cfg['name']} params={n_params:,} steps/epoch={steps_per_epoch}")
    writer = SummaryWriter(str(log_dir))
    writer.add_text("config", f"```\n{yaml.safe_dump(cfg)}\n```", step)
    (ckpt_dir).mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    from ans import runmeta

    meta = runmeta.start_meta(
        ckpt_dir,
        cfg,
        run_manifests(cfg, synthetic),
        {"synthetic": synthetic, "device": str(device), "params": n_params, "lookahead": _lookahead(model)},
    )
    writer.add_text("meta", f"```\n{json.dumps({k: meta[k] for k in ('git', 'manifests', 'pip_freeze')})}\n```", step)

    history: list[float] = []
    val_hist: list[dict] = []

    def ckpt_state(epoch: int) -> dict:
        return {
            "model_name": model_cfg["name"],
            "model_kwargs": model_cfg.get("kwargs", {}),
            "state_dict": model.state_dict(),
            "ema_state_dict": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": sched.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_score": best,
            "best_metrics": best_metrics,
            "params": n_params,
            "config": cfg,
            "lookahead": _lookahead(model),
            "display_name": cfg.get("display_name"),
            "distilled": distiller is not None,
            "distill_proj": distiller.proj.state_dict()
            if distiller is not None and distiller.proj is not None
            else None,
        }

    t_start = time.time()
    done = False
    epoch = start_epoch
    for epoch in range(start_epoch, epochs):
        t0, n0 = time.time(), step
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            loss, terms, gnorm = train_step(model, batch, loss_fn, optimizer, device, o["grad_clip"], amp, distiller)
            sched.step()
            ema.update(model)
            step += 1
            history.append(loss)
            if step % cfg["log_every"] == 0:
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/grad_norm", gnorm, step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                for k, v in terms.items():
                    writer.add_scalar(f"train/{k}", v, step)
                pbar.set_postfix(loss=f"{loss:.4f}", it_s=f"{(step - n0) / (time.time() - t0):.2f}")
            if max_steps and step >= max_steps:
                done = True
                break
        it_s = (step - n0) / max(time.time() - t0, 1e-9)
        writer.add_scalar("train/it_per_s", it_s, step)
        print(f"epoch {epoch}: step {step}, loss {np.mean(history[-max(1, step - n0) :]):.4f}, {it_s:.2f} it/s")

        v = cfg["val"]
        last_epoch = done or epoch == epochs - 1
        if last_epoch or (epoch + 1) % v["every_epochs"] == 0:
            with_metrics = last_epoch or (epoch + 1) % v["metrics_every_epochs"] == 0
            res = run_validation(ema.model, val_ds, loss_fn, device, v["batch_size"], with_metrics, v["workers"])
            res["epoch"], res["step"] = epoch, step
            val_hist.append(res)
            for k, val in res.items():
                if k not in ("epoch", "step"):
                    writer.add_scalar(f"val/{k}", val, step)
            print("  val " + json.dumps({k: round(x, 4) for k, x in res.items()}))
            score = res.get("pesq", math.nan)
            if not math.isnan(score) and score > best:
                best = score
                best_metrics = {k: res[k] for k in ("pesq", "stoi", "si_sdr") if k in res}
                save_checkpoint(ckpt_dir / "best.pt", **ckpt_state(epoch))
        save_checkpoint(ckpt_dir / "last.pt", **ckpt_state(epoch))
        if done:
            break
    if not (ckpt_dir / "best.pt").exists():
        save_checkpoint(ckpt_dir / "best.pt", **ckpt_state(epoch))
    writer.close()
    runmeta.finish_meta(
        ckpt_dir,
        meta,
        steps=step,
        epochs_done=epoch + 1,
        best_metrics=best_metrics,
        train_seconds=round(time.time() - t_start, 1),
    )
    return {
        "steps": step,
        "loss_history": history,
        "val": val_hist,
        "best": best_metrics,
        "seconds": time.time() - t_start,
        "ckpt_dir": str(ckpt_dir),
    }


def resolve_sections(cfg: dict) -> dict:
    """Fill defaults of the optional ``distill`` / ``data.augment`` sections (for config.yaml/meta.json)."""
    from dataclasses import asdict

    cfg = copy.deepcopy(cfg)
    if (cfg.get("distill") or {}).get("teacher_ckpt"):
        from ans.distill import DistillConfig

        cfg["distill"] = asdict(DistillConfig.from_dict(cfg["distill"]))
    if cfg["data"].get("augment"):
        from ans.augment import AugmentConfig

        aug = asdict(AugmentConfig.from_dict(cfg["data"]["augment"]))
        cfg["data"]["augment"] = yaml.safe_load(yaml.safe_dump(json.loads(json.dumps(aug))))
    return cfg


def _distill_config(cfg: dict):
    if not (cfg.get("distill") or {}).get("teacher_ckpt"):
        return None
    from ans.distill import DistillConfig

    return DistillConfig.from_dict(cfg["distill"])


def _lookahead(model: nn.Module) -> int:
    return int(getattr(getattr(model, "cfg", None), "lookahead", 0) or 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ML_ROOT / "configs" / "crn_small.yaml"))
    ap.add_argument("--synthetic", action="store_true", help="train on synthetic data (smoke test)")
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device")
    ap.add_argument("--run-name")
    ap.add_argument("--extra-noise-dir", help="directory of extra noise wavs for dynamic mixing")
    ap.add_argument("--resume", nargs="?", const="auto", help="checkpoint path (default: <run>/last.pt)")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. optim.lr=5e-4 batch_size=16")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.epochs:
        cfg["schedule"]["epochs"] = args.epochs
    if args.device:
        cfg["device"] = args.device
    if args.extra_noise_dir:
        cfg["data"]["extra_noise_dir"] = args.extra_noise_dir
    if args.run_name:
        cfg["run_name"] = args.run_name
    elif args.synthetic:
        cfg["run_name"] = f"{cfg['run_name']}_synthetic"
    out = train(cfg, synthetic=args.synthetic, resume=args.resume)
    h = out["loss_history"]
    k = max(1, min(20, len(h) // 5))
    print(
        f"done: {out['steps']} steps in {out['seconds']:.0f}s; "
        f"loss first{k}={np.mean(h[:k]):.4f} last{k}={np.mean(h[-k:]):.4f}; best={out['best']}; "
        f"checkpoints in {out['ckpt_dir']}"
    )


if __name__ == "__main__":
    main()
