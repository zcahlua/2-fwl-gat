from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.datasets import QM9


@dataclass
class TargetStats:
    mean: float
    std: float


TARGET_DIM = 19


def load_qm9(root: str):
    return QM9(root=root)


def make_splits(num_graphs: int, seed: int, train_size: int = 100000, val_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_graphs, generator=g)

    train_end = min(train_size, num_graphs)
    val_end = min(train_end + val_size, num_graphs)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]
    return train_idx, val_idx, test_idx


def maybe_subset(indices: torch.Tensor, subset: int | None) -> torch.Tensor:
    if subset is None or subset <= 0:
        return indices
    return indices[: min(subset, indices.numel())]


def get_target_values(dataset, indices: torch.Tensor, target: int) -> torch.Tensor:
    values = []
    for i in indices.tolist():
        y = dataset[i].y
        if y.dim() == 2:
            values.append(y[0, target].item())
        else:
            values.append(y[target].item())
    return torch.tensor(values, dtype=torch.float32)


def compute_train_stats(dataset, train_idx: torch.Tensor, target: int) -> TargetStats:
    y = get_target_values(dataset, train_idx, target)
    mean = float(y.mean().item())
    std = float(y.std(unbiased=False).item())
    if std < 1e-12:
        std = 1.0
    return TargetStats(mean=mean, std=std)


def make_subset(dataset, indices: torch.Tensor) -> Subset:
    return Subset(dataset, indices.tolist())
