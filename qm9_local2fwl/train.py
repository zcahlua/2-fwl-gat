from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from .data import compute_train_stats, load_qm9, make_splits, make_subset, maybe_subset
from .model import Local2FWLGAT
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QM9 Local 2-FWL-style GAT training")
    p.add_argument("--root", type=str, default="./data/qm9")
    p.add_argument("--target", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--pair-layers", type=int, default=3)
    p.add_argument("--gat-layers", type=int, default=2)
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--subset", type=int, default=0, help="Use at most this many samples per split (debug)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="checkpoints/local2fwl_qm9")
    return p.parse_args()


def get_targets(data, target_idx: int) -> torch.Tensor:
    y = data.y
    if y.dim() == 1:
        y = y.unsqueeze(0)
    return y[:, target_idx : target_idx + 1]


def evaluate(model, loader, device, target_idx: int, mean: float, std: float) -> float:
    model.eval()
    mae_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            target = get_targets(batch, target_idx)
            pred = pred_norm * std + mean
            mae_sum += (pred - target).abs().sum().item()
            count += target.numel()
    return mae_sum / max(count, 1)


def train_one_epoch(model, loader, optimizer, device, target_idx: int, mean: float, std: float) -> float:
    model.train()
    loss_sum = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        target = get_targets(batch, target_idx)
        target_norm = (target - mean) / std

        pred_norm = model(batch)
        loss = nn.functional.mse_loss(pred_norm, target_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return loss_sum / max(n, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    dataset = load_qm9(args.root)
    train_idx, val_idx, test_idx = make_splits(len(dataset), args.seed)

    subset = args.subset if args.subset > 0 else None
    train_idx = maybe_subset(train_idx, subset)
    val_idx = maybe_subset(val_idx, subset)
    test_idx = maybe_subset(test_idx, subset)

    stats = compute_train_stats(dataset, train_idx, args.target)

    train_ds = make_subset(dataset, train_idx)
    val_ds = make_subset(dataset, val_idx)
    test_ds = make_subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Local2FWLGAT(
        node_in_dim=dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        pair_layers=args.pair_layers,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = ensure_dir(args.save_dir)
    ckpt_path = Path(save_dir) / f"best_target{args.target}.pt"

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.target, stats.mean, stats.std)
        train_mae = evaluate(model, train_loader, device, args.target, stats.mean, stats.std)
        val_mae = evaluate(model, val_loader, device, args.target, stats.mean, stats.std)

        if val_mae < best_val:
            best_val = val_mae
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "target": args.target,
                    "mean": stats.mean,
                    "std": stats.std,
                    "val_mae": val_mae,
                },
                ckpt_path,
            )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"train_MAE={train_mae:.6f} | val_MAE={val_mae:.6f}"
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    final_train_mae = evaluate(model, train_loader, device, args.target, stats.mean, stats.std)
    final_val_mae = evaluate(model, val_loader, device, args.target, stats.mean, stats.std)
    final_test_mae = evaluate(model, test_loader, device, args.target, stats.mean, stats.std)

    print("Best checkpoint:", ckpt_path)
    print(f"Final MAE | train={final_train_mae:.6f} | val={final_val_mae:.6f} | test={final_test_mae:.6f}")


if __name__ == "__main__":
    main()
