"""Metrics: regression skill, alert classification, calibration, lead time."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score

UNHEALTHY = 100.0


def reg_metrics(y, pred, mask_haze=None) -> dict:
    m = ~(np.isnan(y) | np.isnan(pred))
    e = pred[m] - y[m]
    out = {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "n": int(m.sum())}
    h = m & (y > UNHEALTHY)
    if h.sum():
        out["MAE_haze"] = float(np.mean(np.abs(pred[h] - y[h])))
        out["n_haze"] = int(h.sum())
    return out


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with quantile-free equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            tot += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(tot)


def reliability(y, p, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        s = idx == b
        if s.sum() >= 20:
            rows.append({"p_mean": float(p[s].mean()), "freq": float(y[s].mean()), "n": int(s.sum())})
    return rows


def clf_metrics(y, p, thr) -> dict:
    m = ~np.isnan(y)
    y, p = y[m].astype(int), p[m]
    yhat = (p >= thr).astype(int)
    return {
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "F1": float(f1_score(y, yhat, zero_division=0)),
        "PR_AUC": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "ECE": ece(y, np.clip(p, 0, 1)),
        "positives": int(y.sum()),
        "n": int(len(y)),
        "alerts": int(yhat.sum()),
    }


def onset_events(api: pd.DataFrame, quiet_hours: int = 48) -> list[tuple[pd.Timestamp, str]]:
    """Haze onsets: first hour a station exceeds 100 after >= quiet_hours at or below 100."""
    events = []
    for s in api.columns:
        x = api[s]
        over = (x > UNHEALTHY).fillna(False).to_numpy()
        prev_any = (pd.Series(over, index=x.index).shift(1)
                    .rolling(quiet_hours, min_periods=1).max().fillna(0).to_numpy())
        idx = np.where(over & (prev_any == 0))[0]
        events += [(x.index[i], s) for i in idx if i >= quiet_hours]
    return events


def lead_times(events, alert_flags: pd.DataFrame, window: int = 24) -> dict:
    """For each onset, earliest alert in the `window` hours before it.

    alert_flags: wide bool frame (time x station) of issued alerts at issue time t.
    """
    leads, detected = [], 0
    for t, s in events:
        if s not in alert_flags.columns:
            continue
        w = alert_flags[s].loc[t - pd.Timedelta(hours=window): t - pd.Timedelta(hours=1)]
        hits = w[w.fillna(False).astype(bool)]
        if len(hits):
            detected += 1
            leads.append((t - hits.index[0]) / pd.Timedelta(hours=1))
    n = len(events)
    return {
        "events": n,
        "detected": detected,
        "detection_rate": detected / n if n else float("nan"),
        "median_lead_h": float(np.median(leads)) if leads else 0.0,
        "mean_lead_h": float(np.mean(leads)) if leads else 0.0,
    }
