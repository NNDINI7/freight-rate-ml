"""Fit the final production model on all labeled data and score both
required outputs: validation_predictions.csv and the filled December
chart-input file used by score.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLS, clean_base

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

LANE_PICKUP = "Lexington"
LANE_DELIVERY = "Fort Wayne"
LANE_EQUIPMENT = "Dry Van"


def combined_daily_market_index(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.Series:
    """Date-level mean of market_index using every observed value in both
    train_test.csv and validation.csv (validation's market_index is a
    provided *input* feature, not the target, so using it here is not
    leakage). This gives exact daily coverage through Dec 31, 2025.
    """
    comb = pd.concat([
        train_df[["date", "market_index"]],
        val_df[["date", "market_index"]],
    ])
    return comb.dropna().groupby("date")["market_index"].mean()


def lane_quote_signal_estimate(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """quote_signal is noisy per-shipment (not a smooth date trend, see
    report), so for the December fixed-route rows -- which don't carry a
    quote_signal at all -- we fall back to the historical median
    quote_signal observed on this exact lane+equipment combination.
    """
    comb = pd.concat([
        train_df[(train_df.pickup == LANE_PICKUP) & (train_df.delivery == LANE_DELIVERY)],
        val_df[(val_df.pickup == LANE_PICKUP) & (val_df.delivery == LANE_DELIVERY)],
    ])
    lane_equip = comb[comb.equipment == LANE_EQUIPMENT]
    return float(lane_equip["quote_signal"].median())


def apply_features(df: pd.DataFrame, daily_index: pd.Series, weight_median: float, quote_fallback: float) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    if "weight" not in out.columns:
        out["weight"] = weight_median
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(weight_median)

    if "market_index" not in out.columns:
        out["market_index"] = np.nan
    mapped = out["date"].map(daily_index)
    out["market_index"] = out["market_index"].where(out["market_index"].notna(), mapped)
    out["market_index"] = out["market_index"].fillna(daily_index.mean())

    if "quote_signal" not in out.columns:
        out["quote_signal"] = quote_fallback
    out["quote_signal"] = out["quote_signal"].fillna(quote_fallback)

    out["distance_x_quote"] = out["distance"] * out["quote_signal"]

    for col in ["equipment", "pickup", "delivery"]:
        out[col] = out[col].astype("category")
    return out


def main():
    train_df = clean_base(pd.read_csv(DATA_DIR / "train_test.csv"))
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    val_df["date"] = pd.to_datetime(val_df["date"])
    val_df["weight"] = val_df["weight"]  # keep raw here; abs() applied in apply_features

    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    best_iteration = metrics["best_iteration"]

    daily_index = combined_daily_market_index(train_df, val_df)
    weight_median = train_df["weight"].abs().median()
    quote_fallback = lane_quote_signal_estimate(train_df, val_df)
    print(f"Lane fallback quote_signal (Lexington->Fort Wayne, Dry Van): {quote_fallback:.5f}")

    # ---- retrain final model on ALL labeled rows ----
    train_feat = apply_features(train_df, daily_index, weight_median, quote_fallback)
    X_all, y_all = train_feat[FEATURE_COLS], np.log1p(train_feat["posted_rate"])
    cat_idx = [X_all.columns.get_loc(c) for c in ["equipment", "pickup", "delivery"]]
    full_set = lgb.Dataset(X_all, label=y_all, categorical_feature=cat_idx)

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
        "seed": 42,
    }
    # small headroom over the early-stopped iteration count since we now
    # train on ~18% more rows (the former holdout) than during validation
    final_rounds = int(best_iteration * 1.15)
    final_model = lgb.train(params, full_set, num_boost_round=final_rounds)
    final_model.save_model(str(MODEL_DIR / "lgbm_final.txt"))

    # ---- validation.csv -> validation_predictions.csv ----
    val_feat = apply_features(val_df, daily_index, weight_median, quote_fallback)
    val_preds = np.expm1(final_model.predict(val_feat[FEATURE_COLS]))
    val_preds = np.clip(val_preds, 1.0, None)

    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    pred_map = dict(zip(val_df["load_id"], val_preds))
    template["predicted_rate"] = template["load_id"].map(pred_map).round(2)
    assert template["predicted_rate"].isna().sum() == 0, "missing predictions for some load_id"
    out_path = ROOT / "validation_predictions.csv"
    template.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(template):,} rows)")

    # ---- december_chart_inputs.csv -> filled predicted_rate ----
    dec_df = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    dec_df["date"] = pd.to_datetime(dec_df["date"])
    dec_feat = apply_features(dec_df, daily_index, weight_median, quote_fallback)
    dec_preds = np.expm1(final_model.predict(dec_feat[FEATURE_COLS]))
    dec_out = dec_df.copy()
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out["predicted_rate"] = np.round(dec_preds, 2)
    dec_path = DATA_DIR / "december_chart_inputs_filled.csv"
    dec_out.to_csv(dec_path, index=False)
    print(f"Wrote {dec_path}")
    print(dec_out[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
