"""Train the freight rate model and report validation metrics.

Split strategy
--------------
The validation.csv we must ultimately predict covers Nov-Dec 2025, i.e. it
is entirely in the future relative to the labeled train_test.csv (Jan-Oct
2025). To get a trustworthy read on out-of-sample performance we mirror
that gap with a TIME-BASED holdout instead of a random split: the last 15%
of dates in train_test.csv (by chronological order) are held out for
validation, and the model only ever sees earlier dates during training.
A random split would leak same-day market conditions between train and
test rows and overstate accuracy relative to the real Nov/Dec forecasting
task.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, add_features, clean_base, market_index_trend

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_DIR.mkdir(exist_ok=True)

HOLDOUT_FRACTION = 0.15
RANDOM_STATE = 42


def load_and_split():
    df = pd.read_csv(DATA_DIR / "train_test.csv")
    df = clean_base(df)
    df = df.sort_values("date").reset_index(drop=True)

    cutoff_idx = int(len(df) * (1 - HOLDOUT_FRACTION))
    cutoff_date = df.loc[cutoff_idx, "date"]

    train_df = df[df["date"] < cutoff_date].copy()
    holdout_df = df[df["date"] >= cutoff_date].copy()
    return train_df, holdout_df, cutoff_date


def fit_transform(train_df: pd.DataFrame, holdout_df: pd.DataFrame):
    weight_median = train_df["weight"].median()
    trend_fn, daily_index = market_index_trend(train_df)

    train_feat = add_features(train_df, trend_fn, daily_index, weight_median)
    holdout_feat = add_features(holdout_df, trend_fn, daily_index, weight_median)

    return train_feat, holdout_feat, weight_median, trend_fn, daily_index


def train_model(train_feat: pd.DataFrame, holdout_feat: pd.DataFrame):
    X_train, y_train = train_feat[FEATURE_COLS], train_feat["posted_rate"]
    X_hold, y_hold = holdout_feat[FEATURE_COLS], holdout_feat["posted_rate"]

    cat_idx = [X_train.columns.get_loc(c) for c in ["equipment", "pickup", "delivery"]]

    train_set = lgb.Dataset(X_train, label=np.log1p(y_train), categorical_feature=cat_idx, free_raw_data=False)
    valid_set = lgb.Dataset(X_hold, label=np.log1p(y_hold), reference=train_set, categorical_feature=cat_idx, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred_log = model.predict(X_hold, num_iteration=model.best_iteration)
    preds = np.expm1(pred_log)

    metrics = {
        "mae": float(mean_absolute_error(y_hold, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_hold, preds))),
        "mape": float(mean_absolute_percentage_error(y_hold, preds)),
        "best_iteration": int(model.best_iteration),
        "n_train": int(len(X_train)),
        "n_holdout": int(len(X_hold)),
    }
    return model, metrics, preds


def main():
    train_df, holdout_df, cutoff_date = load_and_split()
    train_feat, holdout_feat, weight_median, trend_fn, daily_index = fit_transform(train_df, holdout_df)
    model, metrics, preds = train_model(train_feat, holdout_feat)

    print(f"Time-based split cutoff date: {cutoff_date.date()}")
    print(f"Train rows: {metrics['n_train']:,}  Holdout rows: {metrics['n_holdout']:,}")
    print(f"Holdout MAE:  ${metrics['mae']:.2f}")
    print(f"Holdout RMSE: ${metrics['rmse']:.2f}")
    print(f"Holdout MAPE: {metrics['mape']*100:.2f}%")

    model.save_model(str(MODEL_DIR / "lgbm_freight_rate.txt"))
    with open(MODEL_DIR / "weight_median.json", "w") as f:
        json.dump({"weight_median": float(weight_median), "cutoff_date": str(cutoff_date.date())}, f)
    daily_index.to_csv(MODEL_DIR / "daily_market_index.csv", header=["market_index"])
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # feature importance for the report
    importances = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(MODEL_DIR / "feature_importance.csv", header=["gain"])
    print("\nTop features by gain:")
    print(importances.head(8))


if __name__ == "__main__":
    main()
