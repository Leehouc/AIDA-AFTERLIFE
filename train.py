from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lhconnect6.dataset import Connect6NpzDataset, resolve_dataset_paths
from lhconnect6.model import build_model
from lhconnect6.training import run_epoch
from lhconnect6.utils import choose_device, dump_json, ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight LHconnect6 competition model.")
    parser.add_argument("--data-dir", required=True, help="Directory containing train.npz and optional val.npz.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs.")
    parser.add_argument("--blocks", type=int, default=20, help="Number of residual blocks.")
    parser.add_argument("--channels", type=int, default=128, help="Trunk width.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--value-loss-weight", type=float, default=0.25, help="Multiplier for value loss.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Cross entropy label smoothing.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--augment-symmetry", action="store_true", help="Enable random 8-way symmetry augmentation.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    best_metric: float,
) -> dict:
    return {
        "epoch": epoch,
        "model_type": getattr(model, "model_type", "res"),
        "model_param": list(getattr(model, "model_param", (args.blocks, args.channels))),
        "state_dict": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_args": vars(args),
        "best_metric": best_metric,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = ensure_dir(args.output_dir)
    train_path, val_path = resolve_dataset_paths(args.data_dir)
    device = choose_device(args.device)

    train_dataset = Connect6NpzDataset(train_path, augment_symmetry=args.augment_symmetry)
    val_dataset = Connect6NpzDataset(val_path, augment_symmetry=False) if val_path is not None else None

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    model = build_model(blocks=args.blocks, channels=args.channels)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_metric = float("inf")
    metrics_log_path = Path(output_dir) / "metrics.jsonl"

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    dump_json(Path(output_dir) / "train_args.json", vars(args))

    with metrics_log_path.open("a", encoding="utf-8") as metrics_out:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                value_loss_weight=args.value_loss_weight,
                label_smoothing=args.label_smoothing,
                use_amp=args.amp,
            )
            scheduler.step()

            val_metrics = None
            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = run_epoch(
                        model=model,
                        loader=val_loader,
                        device=device,
                        optimizer=None,
                        scaler=None,
                        value_loss_weight=args.value_loss_weight,
                        label_smoothing=args.label_smoothing,
                        use_amp=args.amp,
                    )

            monitor_metric = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
            is_best = monitor_metric < best_metric
            if is_best:
                best_metric = monitor_metric

            checkpoint = checkpoint_payload(model, optimizer, scheduler, epoch, args, best_metric)
            torch.save(checkpoint, Path(output_dir) / "last.pt")
            if is_best:
                torch.save(checkpoint, Path(output_dir) / "best.pt")

            line = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "best_metric": best_metric,
            }
            metrics_out.write(json.dumps(line, ensure_ascii=False) + "\n")
            metrics_out.flush()

            if val_metrics is None:
                print(
                    f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                    f"policy_top1={train_metrics['policy_top1']:.4f} "
                    f"mse={train_metrics['mse']:.4f}"
                )
            else:
                print(
                    f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_policy_top1={val_metrics['policy_top1']:.4f} "
                    f"val_mse={val_metrics['mse']:.4f}"
                )


if __name__ == "__main__":
    main()
