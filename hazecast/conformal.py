"""Split-conformal prediction intervals, Mondrian by predicted API band.

Haze-level forecasts are far less certain than clean-air ones, so a single
global interval width would be too wide on good days and too narrow during
haze. We calibrate one width per predicted-API band.
"""
from __future__ import annotations

import numpy as np

BANDS = [0.0, 50.0, 100.0, np.inf]
MIN_BAND_N = 200


def _q(abs_res: np.ndarray, alpha: float) -> float:
    n = len(abs_res)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(abs_res)[k - 1])


def fit(pred: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> dict:
    """Return per-band half-widths for a (1 - alpha) interval."""
    m = ~(np.isnan(pred) | np.isnan(y))
    pred, y = pred[m], y[m]
    res = np.abs(y - pred)
    glob = _q(res, alpha)
    widths = []
    counts = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        sel = (pred >= lo) & (pred < hi)
        counts.append(int(sel.sum()))
        widths.append(_q(res[sel], alpha) if sel.sum() >= MIN_BAND_N else glob)
    return {"alpha": alpha, "bands": [b if np.isfinite(b) else 1e9 for b in BANDS],
            "half_widths": widths, "global_half_width": glob, "counts": counts}


def interval(pred: np.ndarray, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    bands = np.array(cal["bands"], dtype=float)
    idx = np.clip(np.searchsorted(bands, pred, side="right") - 1, 0, len(cal["half_widths"]) - 1)
    hw = np.array(cal["half_widths"])[idx]
    return np.maximum(pred - hw, 0.0), pred + hw


def coverage(pred, y, cal) -> float:
    lo, hi = interval(pred, cal)
    m = ~np.isnan(y)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
