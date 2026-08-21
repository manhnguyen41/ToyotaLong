# Service-parts forecasting

Reproducible PyTorch implementation of a **single global LSTM** for all 8,605 service parts. It uses causal per-window mean scaling and direct four-month forecasts. Validation and test each use three rolling four-month blocks; only observed ground truth is revealed between blocks.

## Setup

Python 3.10+ is required. On the training machine, install a CUDA-enabled PyTorch build appropriate for its driver first, then install this package:

```bash
conda create -n service-parts python=3.11 -y
conda activate service-parts
# Install the correct CUDA PyTorch wheel from https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
```

Do not commit or modify the source workbook. Pass its path at runtime.

## Check data and run

```bash
python -m service_parts_forecasting.cli.inspect_data \
  --data "data/04_20260724_Historical order data and forecasting results for_8605 service parts.xlsx"

python -m service_parts_forecasting.cli.train \
  --config configs/lstm.yaml --data "data/04_20260724_Historical order data and forecasting results for_8605 service parts.xlsx" \
  --stage validation --smoke-test

python -m service_parts_forecasting.cli.train \
  --config configs/experiments/lstm_context12.yaml --data /path/to/workbook.xlsx \
  --stage validation

python -m service_parts_forecasting.cli.evaluate \
  --config configs/experiments/lstm_context12.yaml --data /path/to/workbook.xlsx \
  --stage test
```

Set `seeds: [52, 62, 72, 82, 92]` for the reported five-seed experiment. `device: auto` selects CUDA when available. Keep `warm_start: false` for the main protocol. Compare validation runs using their combined 12-month `paper_score_mean`, then copy the chosen run's `suggested_final_epochs_median` into `final_epochs` before running test.

Each invocation writes to a unique `outputs/<model>/<run_id>/` directory. Smoke runs include `smoke` in the run ID and therefore cannot overwrite full experiments.

## Reference baseline

```bash
python -m service_parts_forecasting.cli.compare_baseline \
  --data /path/to/workbook.xlsx --output-dir outputs/reference_baseline
```

This only reads the EEMD-DMD sheet and recomputes scores; its predictions never enter training.

## Add a forecasting model

Subclass `BaseForecastModel`, decorate it with `@register_model("name")`, and import its module from `models/__init__.py`. The forward method must return `[batch, 4]` normalized forecasts. No dataset, trainer, rolling evaluator, metrics, or writer changes are needed.

Run tests with `pytest`.

## Hyperparameter tuning

Install the tuning extra and run Optuna against the rolling **validation** schedule:

```bash
pip install -e ".[tuning]"

python -m service_parts_forecasting.cli.tune \
  --config configs/tuning/lstm_optuna.yaml \
  --data /path/to/workbook.xlsx
```

The default search covers context length, hidden size, layer count, dropout, learning rate, weight decay, and batch size. It runs 30 TPE trials using seed 52 for selection. Keep `n_jobs: 1` when trials share one GPU. Use `--n-trials N` to override the budget, or add resumable SQLite storage:

```bash
python -m service_parts_forecasting.cli.tune \
  --config configs/tuning/lstm_optuna.yaml \
  --data /path/to/workbook.xlsx \
  --n-trials 50 \
  --storage sqlite:///outputs/tuning/global_lstm_validation.db
```

Every trial is evaluated on the combined 12-month validation paper score. The tuner writes `tuning_results.csv`, `tuning_summary.json`, complete trial artifacts, and a production `best_config.yaml` with the median selected epoch count and final seeds `[52, 62, 72, 82, 92]`. Run that generated config with the test CLI only after tuning is complete.

A fast pipeline check is available; its `smoke_best_config.yaml` is deliberately named so it is not mistaken for a production selection:

```bash
python -m service_parts_forecasting.cli.tune \
  --config configs/tuning/lstm_optuna.yaml \
  --data /path/to/workbook.xlsx \
  --smoke-test --n-trials 2
```
