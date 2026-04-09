from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from lhconnect6.dataset import Connect6NpzDataset, resolve_dataset_paths
from lhconnect6.model import load_checkpoint
from lhconnect6.training import run_epoch
from lhconnect6.utils import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an LHconnect6 checkpoint on NPZ data.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path produced by train.py or export_competition_model.py.")
    parser.add_argument("--data-dir", required=True, help="Directory containing train.npz and optional val.npz.")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--value-loss-weight", type=float, default=0.25, help="Value loss multiplier.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Loss label smoothing.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)

    train_path, val_path = resolve_dataset_paths(args.data_dir)
    if args.split == "val":
        if val_path is None:
            raise FileNotFoundError("val.npz does not exist in the provided data directory")
        data_path = val_path
    else:
        data_path = train_path

    dataset = Connect6NpzDataset(data_path, augment_symmetry=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        metrics = run_epoch(
            model=model,
            loader=loader,
            device=device,
            optimizer=None,
            scaler=None,
            value_loss_weight=args.value_loss_weight,
            label_smoothing=args.label_smoothing,
            use_amp=args.amp,
        )

    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "samples": len(dataset),
        "mse": metrics.get("mse"),
        "metrics": metrics,
        "model_param": checkpoint.get("model_param"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
