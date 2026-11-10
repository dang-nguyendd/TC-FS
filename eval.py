"""
Evaluate a trained xLSTM Large multi-step forecasting checkpoint.

Example:
    python eval.py --csv data.csv \
        --checkpoint checkpoints/xlstm_forecasting/best_model.pt \
        --output-dir results/xlstm_forecasting

The test period is reconstructed using the chronological split stored in the
checkpoint. Metrics are calculated in the original units (after inverse scaling).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge


class WindowDataset(Dataset):
    def __init__(self, features, targets, start, end, seq_len, horizon):
        self.features = features
        self.targets = targets
        self.start = start
        self.seq_len = seq_len
        self.horizon = horizon
        self.count = end - start - seq_len - horizon + 1
        if self.count <= 0:
            raise ValueError("Test split is too short for the saved sequence length and horizon.")

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        i = self.start + idx
        x = self.features[i:i + self.seq_len]
        y = self.targets[i + self.seq_len:i + self.seq_len + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


class XLSTMMultiStepForecaster(nn.Module):
    def __init__(
        self, n_features, n_targets, horizon, embedding_dim=512,
        num_heads=4, num_blocks=6, vocab_size=2048
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
            chunkwise_kernel="chunkwise--triton_xl_chunk",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
        )
        self.xlstm = xLSTMLarge(config)
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, horizon * n_targets),
        )
        self.horizon = horizon
        self.n_targets = n_targets

    def forward(self, x):
        x = self.input_projection(x)
        hidden, _ = self.xlstm.backbone(x)
        context = hidden[:, -1, :]
        forecast = self.forecast_head(context)
        return forecast.view(-1, self.horizon, self.n_targets)


def regression_metrics(y_true, y_pred):
    # Inputs are [samples, horizon, targets], in original data units.
    error = y_pred - y_true
    mae = np.mean(np.abs(error), axis=(0, 1))
    rmse = np.sqrt(np.mean(error ** 2, axis=(0, 1)))

    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2, axis=0)
    ss_tot = np.sum((y_true_flat - y_true_flat.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)

    return mae, rmse, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="results/xlstm_forecasting")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dc = checkpoint["data_config"]
    mc = checkpoint["model_config"]

    df = pd.read_csv(args.csv)
    time_col = dc.get("time_column")
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)

    selected = list(dict.fromkeys(dc["feature_cols"] + dc["target_cols"]))
    for col in selected:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the CSV.")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=selected).reset_index(drop=True)

    feature_raw = df[dc["feature_cols"]].to_numpy(dtype=np.float32)
    target_raw = df[dc["target_cols"]].to_numpy(dtype=np.float32)

    x_mean = np.asarray(dc["x_mean"], dtype=np.float32)
    x_std = np.asarray(dc["x_std"], dtype=np.float32)
    y_mean = np.asarray(dc["y_mean"], dtype=np.float32)
    y_std = np.asarray(dc["y_std"], dtype=np.float32)

    feature_scaled = ((feature_raw - x_mean) / x_std).astype(np.float32)
    target_scaled = ((target_raw - y_mean) / y_std).astype(np.float32)

    test_start = int(dc["val_end"])
    test_end = len(df)
    test_ds = WindowDataset(
        feature_scaled, target_scaled, test_start, test_end,
        int(dc["seq_len"]), int(dc["horizon"])
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = XLSTMMultiStepForecaster(**mc).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    actuals = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).cpu().numpy()
            predictions.append(pred)
            actuals.append(y.numpy())

    pred_scaled = np.concatenate(predictions, axis=0)
    true_scaled = np.concatenate(actuals, axis=0)

    # Return values to the original units.
    pred = pred_scaled * y_std[None, None, :] + y_mean[None, None, :]
    true = true_scaled * y_std[None, None, :] + y_mean[None, None, :]

    mae, rmse, r2 = regression_metrics(true, pred)
    metrics = pd.DataFrame({
        "target": dc["target_cols"],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    # Horizon-wise metrics show how forecast quality changes with lead time.
    horizon_rows = []
    for h in range(true.shape[1]):
        yt = true[:, h, :]
        yp = pred[:, h, :]
        for j, target_name in enumerate(dc["target_cols"]):
            err = yp[:, j] - yt[:, j]
            ss_tot = np.sum((yt[:, j] - yt[:, j].mean()) ** 2)
            r2_h = 1.0 - np.sum(err ** 2) / ss_tot if ss_tot > 1e-12 else np.nan
            horizon_rows.append({
                "horizon_step": h + 1,
                "target": target_name,
                "MAE": np.mean(np.abs(err)),
                "RMSE": np.sqrt(np.mean(err ** 2)),
                "R2": r2_h,
            })
    horizon_metrics = pd.DataFrame(horizon_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "test_metrics_by_target.csv", index=False)
    horizon_metrics.to_csv(output_dir / "test_metrics_by_horizon.csv", index=False)

    # Save all forecast windows in long format, with each lead time and target.
    prediction_rows = []
    for sample_idx in range(pred.shape[0]):
        for h in range(pred.shape[1]):
            row = {
                "window_index": sample_idx,
                "horizon_step": h + 1,
            }
            for j, target_name in enumerate(dc["target_cols"]):
                row[f"{target_name}_actual"] = true[sample_idx, h, j]
                row[f"{target_name}_predicted"] = pred[sample_idx, h, j]
            prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "test_predictions.csv", index=False
    )

    print(f"Device: {device}")
    print(f"Test windows: {len(test_ds)}")
    print("\nOverall test metrics (original units):")
    print(metrics.to_string(index=False))
    print(f"\nSaved metrics and predictions to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
