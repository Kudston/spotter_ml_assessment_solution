# Freight Rate Prediction Challenge

This solution trains a scikit-learn `Pipeline` with LightGBM regression and generates
the required November and December predictions. Optuna hyperparameter tuning is
disabled by default; enable it with `--tune`. Tuning uses a chronological holdout of
the final two complete months so model selection is evaluated on future dates rather
than a random sample. Because those holdout results are used for model selection,
they are not an independent final test estimate. When tuning is enabled, the selected
model is refit on all training rows before generating predictions.

## Environment

Use Python 3.10–3.13. Install the training and scorer dependencies:

```powershell
python -m pip install -r requirements.txt
```

In the supplied Anaconda installation, the assessment environment can be invoked
without activating it first:

```powershell
C:\Users\kudston\anaconda3\envs\fgrand\python.exe -m pip install -r requirements.txt
```

## Run

Run from the directory containing the CSV files:

```powershell
python train.py
# Enable Optuna hyperparameter tuning
python train.py --tune
# Optionally set the number of Optuna trials
python train.py --tune --trials 50
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
python make_report.py
```

`train.py` writes:

- `validation_predictions.csv` — exactly `load_id,predicted_rate`, in template order
- `december_chart_inputs.csv` — the supplied seven columns with predicted rates filled
- `holdout_metrics.json` — holdout metrics, split dates, and data-quality counts
  including Optuna's best parameters and holdout MAE

The scorer validates the prediction files and creates
`scorer_results/candidate_december.png`. `make_report.py` embeds that chart and the
holdout results in `assessment_report.pdf`.

Optional input/output paths are available through `python train.py --help`.
`--tune` is off by default and must be supplied to run Optuna. The default search
uses 30 trials; use `--tune --trials 50` to run 50 trials.
