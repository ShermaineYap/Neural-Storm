"""Feature engineering for station-level API forecasting.

Every feature at issue time t uses only readings at or before t (no leakage).
Targets look forward: API at t+H and "Unhealthy (API > 100) at any hour in (t, t+24]".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stations import STATIONS, nearest_neighbours

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
UNHEALTHY = 100.0

FEATURES = (
    ["api"]
    + [f"lag_{l}" for l in LAGS]
    + ["d_3", "d_6", "d_24",
       "mean_6", "mean_24", "mean_72", "max_24", "std_24", "missing_24",
       "nbr_mean", "nbr_max", "nbr_d_6",
       "reg_mean", "reg_d_24",
       "nat_median", "nat_p90", "nat_frac_over_100", "nat_frac_over_70", "nat_d_24",
       "hour", "dow", "month_sin", "month_cos", "sw_monsoon", "station_code"]
)

STATION_CODES = {s: i for i, s in enumerate(sorted(STATIONS))}


def _stack(df: pd.DataFrame, name: str) -> pd.Series:
    s = df.stack(future_stack=True)
    s.name = name
    return s


def build_feature_frames(wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute each feature as a wide (time x station) frame."""
    w = wide
    f: dict[str, pd.DataFrame] = {"api": w}
    for l in LAGS:
        f[f"lag_{l}"] = w.shift(l)
    f["d_3"] = w - w.shift(3)
    f["d_6"] = w - w.shift(6)
    f["d_24"] = w - w.shift(24)
    f["mean_6"] = w.rolling(6, min_periods=3).mean()
    f["mean_24"] = w.rolling(24, min_periods=6).mean()
    f["mean_72"] = w.rolling(72, min_periods=12).mean()
    f["max_24"] = w.rolling(24, min_periods=6).max()
    f["std_24"] = w.rolling(24, min_periods=6).std()
    f["missing_24"] = w.isna().rolling(24, min_periods=1).mean()

    nn = nearest_neighbours(3)
    nbr_mean = pd.DataFrame(index=w.index, columns=w.columns, dtype="float32")
    nbr_max = pd.DataFrame(index=w.index, columns=w.columns, dtype="float32")
    for s in w.columns:
        cols = [c for c in nn[s] if c in w.columns]
        nbr_mean[s] = w[cols].mean(axis=1)
        nbr_max[s] = w[cols].max(axis=1)
    f["nbr_mean"] = nbr_mean
    f["nbr_max"] = nbr_max
    f["nbr_d_6"] = nbr_mean - nbr_mean.shift(6)

    region_of = {s: STATIONS[s][3] for s in w.columns}
    reg_mean = pd.DataFrame(index=w.index, columns=w.columns, dtype="float32")
    for r in set(region_of.values()):
        cols = [s for s in w.columns if region_of[s] == r]
        m = w[cols].mean(axis=1)
        for s in cols:
            reg_mean[s] = m
    f["reg_mean"] = reg_mean
    f["reg_d_24"] = reg_mean - reg_mean.shift(24)

    def bcast(series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(np.repeat(series.values[:, None], w.shape[1], axis=1),
                            index=w.index, columns=w.columns)

    nat_median = w.median(axis=1)
    valid = w.notna().sum(axis=1).replace(0, np.nan)
    f["nat_median"] = bcast(nat_median)
    f["nat_p90"] = bcast(w.quantile(0.9, axis=1))
    f["nat_frac_over_100"] = bcast((w > UNHEALTHY).sum(axis=1) / valid)
    f["nat_frac_over_70"] = bcast((w > 70).sum(axis=1) / valid)
    f["nat_d_24"] = bcast(nat_median - nat_median.shift(24))

    idx = w.index
    f["hour"] = bcast(pd.Series(idx.hour, index=idx).astype("float32"))
    f["dow"] = bcast(pd.Series(idx.dayofweek, index=idx).astype("float32"))
    f["month_sin"] = bcast(pd.Series(np.sin(2 * np.pi * idx.month / 12), index=idx))
    f["month_cos"] = bcast(pd.Series(np.cos(2 * np.pi * idx.month / 12), index=idx))
    f["sw_monsoon"] = bcast(pd.Series(idx.month.isin([6, 7, 8, 9]).astype("float32"), index=idx))
    f["station_code"] = pd.DataFrame(
        np.repeat(np.array([[STATION_CODES[s] for s in w.columns]], dtype="float32"), len(idx), axis=0),
        index=idx, columns=w.columns)
    return f


def build_targets(wide: pd.DataFrame, horizons=(24, 48)) -> dict[str, pd.DataFrame]:
    t = {}
    for h in horizons:
        t[f"y_{h}"] = wide.shift(-h)
    # max over the next 24 hours (t+1 .. t+24)
    fut_max = wide[::-1].rolling(24, min_periods=6).max()[::-1].shift(-1)
    alert = (fut_max > UNHEALTHY).astype("float32").where(fut_max.notna())
    t["alert_24"] = alert
    t["fut_max_24"] = fut_max
    return t


def make_long(wide: pd.DataFrame, with_targets: bool = True, dropna_api: bool = True) -> pd.DataFrame:
    """Long table: one row per (time, station) with features (+ targets)."""
    feats = build_feature_frames(wide)
    cols = [_stack(feats[k], k) for k in FEATURES]
    if with_targets:
        tg = build_targets(wide)
        cols += [_stack(v, k) for k, v in tg.items()]
    long = pd.concat(cols, axis=1)
    long.index.names = ["time", "station"]
    if dropna_api:
        long = long[long["api"].notna()]
    return long.astype("float32")
