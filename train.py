"""
Train xLSTM Large for multivariate, multi-step forecasting on a CSV file.

Example:
    python train.py --csv data.csv --time-column timestamp \
        --seq-len 256 --horizon 24 --epochs 50 --batch-size 16

By default, all numeric columns are used as both input features and prediction
targets. Use --feature-cols and --target-cols to select columns explicitly.
The split is chronological: train -> validation -> test.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_columns(value):
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def load_dataframe(args):
    df = pd.read_csv(args.csv)

    if args.time_column:
        if args.time_column not in df.columns:
            raise ValueError(f"Time column '{args.time_column}' was not found.")
        df[args.time_column] = pd.to_datetime(df[args.time_column], errors="coerce")
        df = df.dropna(subset=[args.time_column]).sort_values(args.time_column)

    feature_cols = parse_columns(args.feature_cols)
    target_cols = parse_columns(args.target_cols)

    if feature_cols is None:
        excluded = {args.time_column} if args.time_column else set()
        feature_cols = [
            c for c in df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")

    if target_cols is None:
        target_cols = list(feature_cols)

    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")

    # Convert selected columns to numeric and remove rows with missing values.
    selected = list(dict.fromkeys(feature_cols + target_cols))
    for col in selected:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=selected).reset_index(drop=True)

    if len(df) < args.seq_len + args.horizon + 10:
        raise ValueError(
            f"Only {len(df)} usable rows remain. Increase the dataset size "
            "or reduce --seq-len / --horizon."
        )

    return df, feature_cols, target_cols


class WindowDataset(Dataset):
    def __init__(self, features, targets, start, end, seq_len, horizon):
        self.features = features
        self.targets = targets
        self.start = start
        self.seq_len = seq_len
        self.horizon = horizon
        self.count = end - start - seq_len - horizon + 1
        if self.count <= 0:
            raise ValueError(
                "A data split is too short for the selected sequence length "
                f"({seq_len}) and forecast horizon ({horizon})."
            )

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        i = self.start + idx
        x = self.features[i:i + self.seq_len]
        y = self.targets[i + self.seq_len:i + self.seq_len + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


class XLSTMMultiStepForecaster(nn.Module):
    def __init__(
        self,
        n_features,
        n_targets,
        horizon,
        embedding_dim=512,
        num_heads=4,
        num_blocks=6,
        vocab_size=2048,
        chunkwise_kernel="chunkwise--triton_xl_chunk",
        sequence_kernel="native_sequence__triton",
        step_kernel="triton",
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_features, embedding_dim)

        config = xLSTMLargeConfig(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            return_last_states=False,
            mode="train",
            chunkwise_kernel=chunkwise_kernel,
            sequence_kernel=sequence_kernel,
            step_kernel=step_kernel,
        )

        # Use the xLSTM Large backbone with continuous-valued tabular inputs.
        # The token embedding and language-model head are not used.
        self.xlstm = xLSTMLarge(config)
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, horizon * n_targets),
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        # x: [batch, input_length, n_features]
        x = self.input_projection(x)
        hidden, _ = self.xlstm.backbone(x)
        context = hidden[:, -1, :]
        forecast = self.forecast_head(context)
        return forecast.view(-1, self.horizon, self.n_targets)


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred = model(x)
            loss = criterion(pred, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV file.")
    parser.add_argument("--time-column", default=None,
                        help="Optional timestamp column used to sort rows.")
    parser.add_argument("--feature-cols", default=None,
                        help="Comma-separated input columns; default: all numeric columns.")
    parser.add_argument("--target-cols", default=None,
                        help="Comma-separated prediction columns; default: feature columns.")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Number of historical timesteps supplied to the model.")
    parser.add_argument("--horizon", type=int, default=24,
                        help="Number of future timesteps to predict.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints/xlstm_forecasting")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1
            and args.train_ratio + args.val_ratio < 1):
        raise ValueError("train-ratio and val-ratio must be positive and sum to less than 1.")

    set_seed(args.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    df, feature_cols, target_cols = load_dataframe(args)
    n_rows = len(df)
    train_end = int(n_rows * args.train_ratio)
    val_end = int(n_rows * (args.train_ratio + args.val_ratio))

    if min(train_end, val_end - train_end, n_rows - val_end) < args.seq_len + args.horizon:
        raise ValueError(
            "At least one chronological split is too short. Reduce --seq-len / "
            "--horizon or change the split ratios."
        )

    feature_raw = df[feature_cols].to_numpy(dtype=np.float32)
    target_raw = df[target_cols].to_numpy(dtype=np.float32)

    # Fit normalization using only the training period.
    x_mean = feature_raw[:train_end].mean(axis=0)
    x_std = feature_raw[:train_end].std(axis=0)
    y_mean = target_raw[:train_end].mean(axis=0)
    y_std = target_raw[:train_end].std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_std[y_std < 1e-8] = 1.0

    feature_scaled = ((feature_raw - x_mean) / x_std).astype(np.float32)
    target_scaled = ((target_raw - y_mean) / y_std).astype(np.float32)

    train_ds = WindowDataset(feature_scaled, target_scaled, 0, train_end,
                             args.seq_len, args.horizon)
    val_ds = WindowDataset(feature_scaled, target_scaled, train_end, val_end,
                           args.seq_len, args.horizon)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = XLSTMMultiStepForecaster(
        n_features=len(feature_cols),
        n_targets=len(target_cols),
        horizon=args.horizon,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        vocab_size=args.vocab_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Rows: {n_rows} | train: {train_end} | val: {val_end-train_end} | test: {n_rows-val_end}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Targets ({len(target_cols)}): {target_cols}")
    print(f"Training windows: {len(train_ds)} | validation windows: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": best_state,
                "model_config": {
                    "n_features": len(feature_cols),
                    "n_targets": len(target_cols),
                    "horizon": args.horizon,
                    "embedding_dim": args.embedding_dim,
                    "num_heads": args.num_heads,
                    "num_blocks": args.num_blocks,
                    "vocab_size": args.vocab_size,
                },
                "data_config": {
                    "feature_cols": feature_cols,
                    "target_cols": target_cols,
                    "seq_len": args.seq_len,
                    "horizon": args.horizon,
                    "time_column": args.time_column,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "x_mean": x_mean.tolist(),
                    "x_std": x_std.tolist(),
                    "y_mean": y_mean.tolist(),
                    "y_std": y_std.tolist(),
                    "train_end": train_end,
                    "val_end": val_end,
                },
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    print(f"Best validation MSE: {best_val:.6f}")
    print("Training complete. Run eval.py with the same CSV and checkpoint.")


if __name__ == "__main__":
    main()
