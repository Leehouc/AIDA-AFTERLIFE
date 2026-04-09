from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .symmetry import apply_symmetry_index, apply_symmetry_planes


class Connect6NpzDataset(Dataset):
    def __init__(
        self,
        npz_path: str | Path,
        augment_symmetry: bool = False,
    ) -> None:
        super().__init__()
        data = np.load(npz_path)
        self.inputs = data["inputs"]
        self.policy_targets = data["policy_targets"]
        self.value_targets = data["value_targets"]
        self.players = data["players"] if "players" in data else np.zeros(len(self.inputs), dtype=np.int8)
        self.stages = data["stages"] if "stages" in data else np.zeros(len(self.inputs), dtype=np.int8)
        self.augment_symmetry = augment_symmetry

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        planes = self.inputs[index].astype(np.float32, copy=False)
        policy_target = int(self.policy_targets[index])
        if self.augment_symmetry:
            sym = int(np.random.randint(8))
            planes = apply_symmetry_planes(planes, sym)
            policy_target = apply_symmetry_index(policy_target, sym)

        return {
            "input": torch.from_numpy(np.ascontiguousarray(planes)),
            "policy_target": torch.tensor(policy_target, dtype=torch.long),
            "value_target": torch.tensor(int(self.value_targets[index]), dtype=torch.long),
            "player": torch.tensor(int(self.players[index]), dtype=torch.long),
            "stage": torch.tensor(int(self.stages[index]), dtype=torch.long),
        }


def resolve_dataset_paths(data_dir: str | Path) -> tuple[Path, Path | None]:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    return train_path, val_path if val_path.exists() else None
