"""Out-of-sample forecasting for any issue hour.

For an issue time inside a walk-forward test period, the model of *that* fold is
used, so every forecast shown in the app (including the 2019 haze replay) comes
from a model that never saw that period. After the last test period the latest
fold's model is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import conformal
from .data import load_wide
from .features import FEATURES, build_feature_frames

ART = Path(__file__).resolve().parent.parent / "artifacts"
HISTORY_H = 200

FOLD_PERIODS = [
    ("F1_2019", "2019-01-01", "2020-01-01"),
    ("F2_2020", "2020-01-01", "2021-01-01"),
    ("F3_2021-22", "2021-01-01", "2100-01-01"),
]

DRIVER_GROUPS = {
    "Own station: current level": ["api", "lag_1", "lag_2", "lag_3", "mean_6"],
    "Own station: recent trend": ["d_3", "d_6", "d_24", "std_24", "max_24", "mean_24", "mean_72",
                                  "lag_6", "lag_12", "lag_24", "lag_48", "lag_72", "lag_168"],
    "Neighbouring stations": ["nbr_mean", "nbr_max", "nbr_d_6"],
    "Regional haze": ["reg_mean", "reg_d_24"],
    "Nationwide haze": ["nat_median", "nat_p90", "nat_frac_over_100", "nat_frac_over_70", "nat_d_24"],
    "Season & time of day": ["hour", "dow", "month_sin", "month_cos", "sw_monsoon"],
    "Station baseline": ["station_code"],
    "Data gaps": ["missing_24"],
}
FEATURE_GROUP = {f: g for g, fs in DRIVER_GROUPS.items() for f in fs}
WIDESPREAD_GROUPS = {"Neighbouring stations", "Regional haze", "Nationwide haze"}


def fold_for(t: pd.Timestamp) -> str:
    for name, a, b in FOLD_PERIODS:
        if pd.Timestamp(a) <= t < pd.Timestamp(b):
            return name
    return FOLD_PERIODS[0][0] if t < pd.Timestamp(FOLD_PERIODS[0][1]) else FOLD_PERIODS[-1][0]


class Forecaster:
    def __init__(self, wide: pd.DataFrame | None = None, art_dir: Path | str = ART):
        self.art = Path(art_dir)
        self.wide = wide if wide is not None else load_wide()
        self.cal = json.loads((self.art / "conformal.json").read_text())
        self.metrics = json.loads((self.art / "metrics.json").read_text()) if (self.art / "metrics.json").exists() else {}
        self.thr = self.metrics.get("config", {}).get("alert_threshold", 0.2)
        self._models: dict[str, lgb.Booster] = {}

    def model(self, key: str) -> lgb.Booster:
        if key not in self._models:
            self._models[key] = lgb.Booster(model_file=str(self.art / "models" / f"{key}.txt"))
        return self._models[key]

    def _features(self, t_from: pd.Timestamp, t_to: pd.Timestamp, stations=None) -> pd.DataFrame:
        w = self.wide.loc[t_from - pd.Timedelta(hours=HISTORY_H): t_to]
        frames = build_feature_frames(w)
        sl = slice(t_from, t_to)
        cols = stations if stations is not None else list(w.columns)
        parts = {k: frames[k].loc[sl, cols].stack(future_stack=True) for k in FEATURES}
        X = pd.DataFrame(parts)[FEATURES]
        X.index.names = ["time", "station"]
        return X.astype("float32")

    def _predict(self, X: pd.DataFrame, fold: str, contrib: bool = False) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        out["api_now"] = X["api"].to_numpy()
        for h in (24, 48):
            # models predict the change from the current reading
            p = X["api"].to_numpy(dtype=float) + self.model(f"{fold}_reg{h}").predict(X)
            lo, hi = conformal.interval(p, self.cal[f"{fold}_reg{h}"])
            out[f"pred_{h}"], out[f"lo_{h}"], out[f"hi_{h}"] = p, lo, hi
        out["p_alert"] = self.model(f"{fold}_alert24").predict(X)
        out["missing_24"] = X["missing_24"].to_numpy()
        out["fold"] = fold
        if contrib:
            c = self.model(f"{fold}_reg24").predict(X, pred_contrib=True)
            cdf = pd.DataFrame(c[:, :-1], columns=FEATURES, index=X.index)
            grouped = cdf.T.groupby(FEATURE_GROUP).sum().T
            out = out.join(grouped.add_prefix("drv::"))
            out["base_value"] = c[:, -1]  # average predicted change (API points)
        return out

    def forecast(self, t, contrib: bool = True) -> pd.DataFrame:
        """All stations, one issue hour."""
        t = pd.Timestamp(t)
        X = self._features(t, t)
        res = self._predict(X, fold_for(t), contrib=contrib)
        return res.droplevel("time")

    def series(self, station: str, t_from, t_to) -> pd.DataFrame:
        """One station, every issue hour in [t_from, t_to]; fold chosen per hour."""
        t_from, t_to = pd.Timestamp(t_from), pd.Timestamp(t_to)
        X = self._features(t_from, t_to, stations=[station])
        times = X.index.get_level_values("time")
        parts = []
        for fold in sorted({fold_for(t) for t in times.unique()}):
            m = np.array([fold_for(t) == fold for t in times])
            if m.any():
                parts.append(self._predict(X[m], fold))
        return pd.concat(parts).droplevel("station").sort_index()
