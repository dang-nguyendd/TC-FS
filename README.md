# xLSTM Large: tabular multi-step forecasting

## Files
- `train.py`: chronological train/validation/test split, training, early stopping, checkpoint saving.
- `eval.py`: loads the best checkpoint and evaluates only the held-out test period.

## Install
Install the xLSTM package and its `mlstm_kernels` dependency in the environment
used for your xLSTM Large example. You also need `numpy`, `pandas`, and `torch`.

## Train
```bash
python train.py --csv path/to/data.csv --time-column timestamp \
    --seq-len 256 --horizon 24 --epochs 50 --batch-size 16
```

## Evaluate
```bash
python eval.py --csv path/to/data.csv \
    --checkpoint checkpoints/xlstm_forecasting/best_model.pt
```

By default, all numeric columns are input features and prediction targets. Select
columns with comma-separated lists if needed:
```bash
python train.py --csv data.csv --time-column timestamp \
    --feature-cols temp,humidity,wind --target-cols temp,humidity \
    --seq-len 96 --horizon 12
```

The model receives `[batch, sequence_length, features]` continuous-valued inputs.
A linear input projection maps the tabular features to the xLSTM embedding dimension.
The final historical hidden state is mapped to `[batch, horizon, targets]`.

The split is chronological (70% train, 15% validation, 15% test by default).
Normalization is fitted using the training period only. The test period is not
used during training or model selection.

Note: the xLSTM Large language-model wrapper normally accepts integer token IDs.
These scripts use its xLSTM Large backbone directly after a numerical input
projection, rather than quantizing continuous data into token IDs.
