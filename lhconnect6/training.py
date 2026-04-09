from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import AverageMeter


def compute_batch_metrics(
    value_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    value_targets: torch.Tensor,
    policy_targets: torch.Tensor,
    value_loss_weight: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_loss = F.cross_entropy(policy_logits, policy_targets, label_smoothing=label_smoothing)
    value_loss = F.cross_entropy(value_logits, value_targets, label_smoothing=label_smoothing)
    total_loss = policy_loss + value_loss_weight * value_loss

    value_probs = torch.softmax(value_logits, dim=1)
    value_targets_scalar = value_targets.float() - 1.0
    value_pred_scalar = value_probs[:, 2] - value_probs[:, 0]
    value_mse = torch.mean((value_pred_scalar - value_targets_scalar) ** 2).item()

    policy_top1 = (policy_logits.argmax(dim=1) == policy_targets).float().mean().item()
    policy_top5 = (
        policy_logits.topk(k=min(5, policy_logits.shape[1]), dim=1)
        .indices.eq(policy_targets.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    value_acc = (value_logits.argmax(dim=1) == value_targets).float().mean().item()
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "mse": float(value_mse),
        "policy_top1": policy_top1,
        "policy_top5": policy_top5,
        "value_acc": value_acc,
    }
    return total_loss, metrics


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    value_loss_weight: float,
    label_smoothing: float,
    use_amp: bool,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    meters = {
        key: AverageMeter()
        for key in [
            "loss",
            "policy_loss",
            "value_loss",
            "mse",
            "policy_top1",
            "policy_top5",
            "value_acc",
        ]
    }

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        policy_targets = batch["policy_target"].to(device, non_blocking=True)
        value_targets = batch["value_target"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        use_autocast = use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_autocast):
            value_logits, policy_logits = model(inputs)
            total_loss, metrics = compute_batch_metrics(
                value_logits=value_logits,
                policy_logits=policy_logits,
                value_targets=value_targets,
                policy_targets=policy_targets,
                value_loss_weight=value_loss_weight,
                label_smoothing=label_smoothing,
            )

        if is_training:
            if scaler is not None and use_autocast:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        batch_size = inputs.shape[0]
        for key, value in metrics.items():
            meters[key].update(value, batch_size)

    return {key: meter.avg for key, meter in meters.items()}
