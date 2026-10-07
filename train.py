from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


TARGET = "posted_rate"
ID_COLUMN = "load_id"
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
OPTIONAL_NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
]
NUMERIC_FEATURES = [
    "distance",
    "weight",
    *OPTIONAL_NUMERIC_FEATURES,
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_index",
    "latitude_delta",
    "longitude_delta",
]
REQUIRED_FEATURES = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the same fixed feature set for training and both prediction datasets."""
    missing = sorted(set(REQUIRED_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {missing}")

    features = frame.reindex(
        columns=[
            *CATEGORICAL_FEATURES,
            "distance",
            "weight",
            *OPTIONAL_NUMERIC_FEATURES,
            "date",
        ]
    ).copy()

    for column in OPTIONAL_NUMERIC_FEATURES:
        if column not in frame.columns:
            features[column] = np.nan
    numeric_columns = ["distance", "weight", *OPTIONAL_NUMERIC_FEATURES]
    for column in numeric_columns:
        original_values = features[column]
        parsed_values = pd.to_numeric(original_values, errors="coerce")
        invalid_values = original_values.notna() & parsed_values.isna()
        if invalid_values.any():
            raise ValueError(f"Input data contains invalid numeric values in {column}")
        features[column] = parsed_values
    if features["distance"].isna().any() or (features["distance"] <= 0).any():
        raise ValueError("Input data must contain positive, non-missing distances")

    dates = pd.to_datetime(features.pop("date"), errors="raise")
    if dates.isna().any():
        raise ValueError("Input data contains missing dates")

    day_of_year = dates.dt.dayofyear
    day_of_week = dates.dt.dayofweek
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["day_index"] = (dates - pd.Timestamp("2025-01-01")).dt.days.astype(float)
    features["latitude_delta"] = features["delivery_lat"] - features["pickup_lat"]
    features["longitude_delta"] = features["delivery_lon"] - features["pickup_lon"]

    return features[[*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]]


def make_pipeline(regressor_params: dict[str, Any] | None = None) -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=5,
                    sparse_output=True,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    preprocessing = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    regressor_parameters: dict[str, Any] = {
        "objective": "regression_l1",
        "n_estimators": 1200,
        "learning_rate": 0.01,
        "num_leaves": 41,
        "max_depth": 12, 
        "min_child_samples": 60,
        "colsample_bytree": 0.76508,
        "reg_lambda": 0.3265,
        "reg_alpha": 3.57e-5,
        "subsample": 0.903,        
        "subsample_freq": 1,
        "random_state": 42,
        "n_jobs": max(1, min(4, os.cpu_count() or 1)),
        "verbosity": -1,
    }
    if regressor_params is not None:
        regressor_parameters.update(regressor_params)
    regressor = LGBMRegressor(**regressor_parameters)
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features, validate=False)),
            ("preprocessing", preprocessing),
            ("regressor", regressor),
        ]
    )


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mape_percent": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }


def chronological_holdout(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    latest_month = data["date"].max().to_period("M")
    holdout_start = (latest_month - 1).to_timestamp()
    train_mask = data["date"] < holdout_start
    holdout_mask = ~train_mask
    if not train_mask.any() or not holdout_mask.any():
        raise ValueError("Could not form a non-empty chronological holdout")
    return train_mask, holdout_mask, holdout_start


def tune_hyperparameters(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    holdout_features: pd.DataFrame,
    holdout_target: pd.Series,
    n_trials: int,
) -> tuple[dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        parameters: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 80),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=5),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        }
        pipeline = make_pipeline(parameters)
        pipeline.fit(training_features, training_target)
        predictions = pipeline.predict(holdout_features)
        require_positive_predictions(predictions, "Tuning holdout predictions")
        return float(mean_absolute_error(holdout_target, predictions))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params | {"subsample_freq": 1}, float(study.best_value)


def require_positive_predictions(predictions: np.ndarray, label: str) -> None:
    if not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError(f"{label} contains non-finite or non-positive predicted rates")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the freight-rate pipeline, validate it chronologically, and predict."
    )
    parser.add_argument("--train", type=Path, default=Path("train-test.csv"))
    parser.add_argument("--validation", type=Path, default=Path("validation.csv"))
    parser.add_argument(
        "--validation-template",
        type=Path,
        default=Path("validation-predictions-template.csv"),
    )
    parser.add_argument(
        "--december-inputs", type=Path, default=Path("december-chart-inputs.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna hyperparameter trials (default: 30)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Tune model hyperparameters with Optuna before fitting the final model",
    )
    args = parser.parse_args()
    if args.tune and args.trials < 1:
        raise ValueError("--trials must be at least 1")

    data = pd.read_csv(args.train)
    required_training_columns = {ID_COLUMN, TARGET, *REQUIRED_FEATURES}
    missing_training_columns = sorted(required_training_columns - set(data.columns))
    if missing_training_columns:
        raise ValueError(f"Training data is missing columns: {missing_training_columns}")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    if (
        data[TARGET].isna().any()
        or not np.isfinite(data[TARGET].to_numpy(dtype=float)).all()
        or (data[TARGET] <= 0).any()
    ):
        raise ValueError("Training target must contain only finite, positive rates")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].isna().any():
        raise ValueError("Training data contains missing dates")

    train_mask, holdout_mask, holdout_start = chronological_holdout(data)
    best_params: dict[str, Any] | None = None
    best_holdout_mae: float | None = None
    if args.tune:
        best_params, best_holdout_mae = tune_hyperparameters(
            data.loc[train_mask],
            data.loc[train_mask, TARGET],
            data.loc[holdout_mask],
            data.loc[holdout_mask, TARGET],
            args.trials,
        )
    holdout_pipeline = make_pipeline(best_params)
    holdout_pipeline.fit(data.loc[train_mask], data.loc[train_mask, TARGET])
    holdout_predictions = holdout_pipeline.predict(data.loc[holdout_mask])
    require_positive_predictions(holdout_predictions, "Holdout predictions")

    train_actual = data.loc[train_mask, TARGET]
    train_predictions = holdout_pipeline.predict(data.loc[train_mask])
    train_metrics = calculate_metrics(train_actual, train_predictions)

    holdout_actual = data.loc[holdout_mask, TARGET]
    holdout_metrics = calculate_metrics(holdout_actual, holdout_predictions)
    baseline_rate_per_mile = float(
        (data.loc[train_mask, TARGET] / data.loc[train_mask, "distance"]).median()
    )
    baseline_predictions = (
        data.loc[holdout_mask, "distance"].to_numpy() * baseline_rate_per_mile
    )
    baseline_mae = float(mean_absolute_error(holdout_actual, baseline_predictions))

    final_pipeline = make_pipeline(best_params)
    final_pipeline.fit(data, data[TARGET])

    validation = pd.read_csv(args.validation)
    template = pd.read_csv(args.validation_template)
    if ID_COLUMN not in validation or ID_COLUMN not in template:
        raise ValueError("Validation data and template must both include load_id")
    if validation[ID_COLUMN].isna().any() or validation[ID_COLUMN].duplicated().any():
        raise ValueError("Validation data contains missing or duplicate load_id values")
    if template[ID_COLUMN].isna().any() or template[ID_COLUMN].duplicated().any():
        raise ValueError("Validation template contains missing or duplicate load_id values")
    if set(validation[ID_COLUMN]) != set(template[ID_COLUMN]):
        raise ValueError("Validation IDs do not match the prediction template")

    validation_predictions = final_pipeline.predict(validation)
    require_positive_predictions(validation_predictions, "Validation predictions")
    predictions_by_id = pd.Series(validation_predictions, index=validation[ID_COLUMN])
    final_validation = template.copy()
    final_validation["predicted_rate"] = final_validation[ID_COLUMN].map(predictions_by_id)
    if final_validation["predicted_rate"].isna().any():
        raise ValueError("Could not match every validation prediction to the template")

    december = pd.read_csv(args.december_inputs)
    if "predicted_rate" not in december:
        raise ValueError("December inputs must contain a predicted_rate column")
    december_predictions = final_pipeline.predict(december)
    require_positive_predictions(december_predictions, "December predictions")
    december["predicted_rate"] = december_predictions

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / "validation_predictions.csv"
    december_path = args.output_dir / "data" / "december_chart_inputs.csv"
    metrics_path = args.output_dir / "holdout_metrics.json"
    final_validation.to_csv(validation_path, index=False, float_format="%.2f")
    december.to_csv(december_path, index=False, float_format="%.2f")

    metrics: dict[str, Any] = {
        "model": "LightGBM regression_l1 in a scikit-learn preprocessing Pipeline",
        "split": "Chronological: train before the first day of the penultimate month; hold out the final two months",
        "training_rows": int(train_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "training_end": data.loc[train_mask, "date"].max().date().isoformat(),
        "holdout_start": holdout_start.date().isoformat(),
        "holdout_end": data.loc[holdout_mask, "date"].max().date().isoformat(),
        "holdout": holdout_metrics,
        "optuna": {
            "enabled": args.tune,
            "objective": "holdout_mae",
            "direction": "minimize",
            "trials": args.trials if args.tune else 0,
            "best_holdout_mae": best_holdout_mae,
            "best_params": best_params,
        },
        "mileage_baseline_mae": baseline_mae,
        "mae_improvement_over_mileage_baseline_percent": float(
            (1 - holdout_metrics["mae"] / baseline_mae) * 100
        ),
        "training_missing_weight": int(data["weight"].isna().sum()),
        "training_missing_market_index": int(data["market_index"].isna().sum())
        if "market_index" in data
        else None,
        "training_negative_weight": int((data["weight"] < 0).sum()),
        "validation_rows": len(final_validation),
        "validation_prediction_min": float(final_validation["predicted_rate"].min()),
        "validation_prediction_max": float(final_validation["predicted_rate"].max()),
        "december_rows": len(december),
        "december_prediction_min": float(december["predicted_rate"].min()),
        "december_prediction_max": float(december["predicted_rate"].max()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Train MAE: ${train_metrics['mae']:,.2f}")
    print(f"Train RMSE: ${train_metrics['rmse']:,.2f}")
    print(f"Train R2: {train_metrics['r2']:.4f}")

    print(f"Holdout MAE: ${holdout_metrics['mae']:,.2f}")
    print(f"Holdout RMSE: ${holdout_metrics['rmse']:,.2f}")
    print(f"Holdout R2: {holdout_metrics['r2']:.4f}")
    if args.tune:
        print(f"Optuna best holdout MAE: ${best_holdout_mae:,.2f} ({args.trials} trials)")
        print(f"Optuna best parameters: {json.dumps(best_params, sort_keys=True)}")
    print(f"Mileage-baseline MAE: ${baseline_mae:,.2f}")
    print(f"Saved validation predictions: {validation_path}")
    print(f"Saved December predictions: {december_path}")
    print(f"Saved holdout metrics: {metrics_path}")




if __name__ == "__main__":
    main()
