"""Calibrated forecast ranges (spec step 5): split-conformal shifts per model and category.

Each decile gets a one-sided shift learned on Phase 4's train-period folds. Errors are scaled by
the forecast's own spread (q90 - q10), so wide and narrow forecasts are adjusted in step.
"""

import numpy as np
import pandas as pd

from fresh_replenish.metrics import DECILE_COLUMNS, DECILES

KEYS = ["store_id", "product_id"]
CALIBRATED = [f"cal_{c}" for c in DECILE_COLUMNS]
SHIFTS = [f"{c}_shift" for c in DECILE_COLUMNS]
MIN_SERIES = 10  # smaller categories share one pooled group
POOLED = -1


def group_table(ids: pd.DataFrame, min_series: int = MIN_SERIES) -> pd.DataFrame:
    """Calibration group per series: its first-level category, or POOLED when that is small."""
    category = ids.first_category_id
    small = category.map(category.value_counts()) < min_series
    return ids[KEYS].assign(group=category.where(~small, POOLED))


def _spread(frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(frame.q90 - frame.q10, 1e-6).to_numpy()[:, None]


def shifts(calibration: pd.DataFrame, group_of: pd.DataFrame) -> pd.DataFrame:
    """Conformal shift per model, group and decile, in units of forecast spread (wide table)."""
    frame = calibration.dropna(subset=DECILE_COLUMNS).merge(group_of, on=KEYS)
    rows = []
    for (model, group), part in frame.groupby(["model", "group"]):
        errors = part.actual.to_numpy()[:, None] - part[DECILE_COLUMNS].to_numpy()
        scores = np.sort(errors / _spread(part), axis=0)  # each decile's column sorted
        n = len(part)
        rank = np.minimum(np.ceil((n + 1) * DECILES).astype(int), n) - 1  # ceil((n+1)tau)-th
        shift = dict(zip(DECILE_COLUMNS, scores[rank, np.arange(len(DECILES))], strict=True))
        rows.append({"model": model, "group": group, **shift})
    return pd.DataFrame(rows)


def calibrate(
    forecasts: pd.DataFrame, shift_table: pd.DataFrame, group_of: pd.DataFrame
) -> pd.DataFrame:
    """forecasts plus calibrated deciles cal_q10 ... cal_q90 (NaN for point forecasts)."""
    frame = forecasts.merge(group_of, on=KEYS, how="left").merge(
        shift_table, on=["model", "group"], how="left", suffixes=("", "_shift")
    )
    adjusted = frame[DECILE_COLUMNS].to_numpy() + frame[SHIFTS].to_numpy() * _spread(frame)
    frame[CALIBRATED] = np.sort(np.maximum(adjusted, 0), axis=1)
    return frame.drop(columns=SHIFTS)
