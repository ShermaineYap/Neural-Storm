"""Normalised split-conformal prediction intervals.

Haze forecasts are far less certain than clean-air ones, so a single global
interval width is too narrow during haze and too wide on clear days. The
nonconformity score is scaled by a difficulty estimate known at issue time:

    scale = 3 + std_24 (station's volatility over the last 24 h) + 0.2 * |forecast|

and the interval is  forecast +/- q * scale, with q the conformal quantile of
|y - forecast| / scale on a held-out calibration window. The scale formula is
fixed in advance, not tuned on test data.
"""
from __future__ import annotations

import numpy as np

BANDS = [(0.0, 50.0, "pred<50"), (50.0, 100.0, "50-100"), (100.0, np.inf, ">100")]


def difficulty(pred, std_24) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    s = np.nan_to_num(np.asarray(std_24, dtype=float), nan=5.0)
    return 3.0 + s + 0.2 * np.abs(pred)


def fit(pred: np.ndarray, y: np.ndarray, std_24: np.ndarray, alpha: float = 0.2) -> dict:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    m = ~(np.isnan(pred) | np.isnan(y))
    scores = np.abs(y[m] - pred[m]) / difficulty(pred[m], np.asarray(std_24)[m])
    n = len(scores)
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    q = float(np.sort(scores)[k - 1])
    return {"alpha": alpha, "q": q, "n_cal": int(n),
            "n_cal_pred_over_100": int((pred[m] > 100).sum()),
            "scale": "3 + std_24 + 0.2*|forecast|"}


def interval(pred, cal: dict, std_24) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    hw = cal["q"] * difficulty(pred, std_24)
    return np.maximum(pred - hw, 0.0), pred + hw


def coverage_report(pred, y, lo, hi) -> dict:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    ok = ~np.isnan(y)
    inside = (y >= lo) & (y <= hi)
    out = {"empirical_coverage": float(inside[ok].mean()), "mean_width": float(np.mean((hi - lo)[ok])),
           "by_predicted_band": {}}
    for a, b, lab in BANDS:
        s = ok & (pred >= a) & (pred < b)
        if s.sum():
            out["by_predicted_band"][lab] = {"coverage": float(inside[s].mean()),
                                             "mean_width": float(np.mean((hi - lo)[s])), "n": int(s.sum())}
    s = ok & (y > 100)
    if s.sum():
        out["coverage_when_actual_over_100"] = float(inside[s].mean())
    return out
