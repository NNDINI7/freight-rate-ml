"""Shared feature engineering for the freight rate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["equipment", "pickup", "delivery"]
NUMERIC_COLS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "distance_x_quote",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues without touching the target."""
    out = df.copy()

    # weight: sign-flip errors (~0.6% of rows) -> true magnitude is positive
    out["weight"] = out["weight"].abs()

    out["date"] = pd.to_datetime(out["date"])
    return out


def build_date_market_index(train_df: pd.DataFrame) -> pd.Series:
    """Per-date mean of market_index, learned ONLY from training rows.

    market_index behaves like a shared daily market factor (std ~0.025
    within a date vs. a mean that swings from ~0.75 to ~1.45 across the
    year) so the best imputation/extrapolation for a missing or
    out-of-sample date is the date-level average, smoothed against a
    day-of-year trend for dates the model has never seen (validation is
    Nov-Dec, outside the Jan-Oct training window).
    """
    daily = train_df.dropna(subset=["market_index"]).groupby("date")["market_index"].mean()
    return daily


def market_index_trend(train_df: pd.DataFrame):
    """Fit a smooth day-of-year trend to extrapolate market_index into Nov/Dec."""
    daily = build_date_market_index(train_df)
    doy = daily.index.dayofyear.values.astype(float)
    y = daily.values
    # low-degree polynomial is enough to capture the slow seasonal drift
    coeffs = np.polyfit(doy, y, deg=3)
    return np.poly1d(coeffs), daily


def add_features(df: pd.DataFrame, market_trend_fn, daily_index: pd.Series, weight_median: float) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["day_of_year"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # weight: impute missing with the equipment-level median from training
    out["weight"] = out["weight"].fillna(weight_median)

    # market_index: use the same-date average if we saw that date in
    # training, otherwise fall back to the fitted day-of-year trend
    mapped = out["date"].map(daily_index)
    trend_vals = market_trend_fn(out["date"].dt.dayofyear.values.astype(float))
    out["market_index"] = out["market_index"].where(out["market_index"].notna(), mapped)
    out["market_index"] = out["market_index"].where(out["market_index"].notna(), trend_vals)

    out["distance_x_quote"] = out["distance"] * out["quote_signal"]

    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype("category")

    return out
