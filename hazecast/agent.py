"""HazeAgent: turns raw forecasts into decisions and a plain-language bulletin.

For every station it decides one of four actions, and it is allowed to refuse:

  INSUFFICIENT DATA  - more than half of the last 24 h is missing, or no current
                       reading. The agent abstains rather than guess.
  ALERT              - P(API > 100 within 24 h) >= threshold (0.2, from a 4:1
                       miss-vs-false-alarm cost ratio).
  WATCH              - probability below threshold, but the upper end of the
                       80% interval for t+24 h reaches the Unhealthy band.
  CLEAR              - otherwise.

It also labels the *kind* of episode from the SHAP driver groups: if the
neighbouring / regional / nationwide signals push the forecast up more than the
station's own history does, the episode looks widespread (a transboundary-haze
signature); otherwise it looks like local build-up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import api_category
from .forecast import WIDESPREAD_GROUPS
from .stations import STATIONS

ACTIONS = ["ALERT", "WATCH", "CLEAR", "INSUFFICIENT DATA"]
ACTION_COLOURS = {"ALERT": "#C62828", "WATCH": "#EF6C00", "CLEAR": "#2E7D32", "INSUFFICIENT DATA": "#757575"}
MAX_MISSING = 0.5


@dataclass
class Decision:
    station: str
    action: str
    reason: str
    episode: str
    drivers: list[tuple[str, float]]


def decide_row(station: str, r: pd.Series, thr: float) -> Decision:
    drivers = sorted(((k[5:], float(v)) for k, v in r.items() if str(k).startswith("drv::")),
                     key=lambda kv: -abs(kv[1]))
    if np.isnan(r["api_now"]) or r["missing_24"] > MAX_MISSING:
        return Decision(station, "INSUFFICIENT DATA",
                        f"{r['missing_24']:.0%} of the last 24 h is missing; not forecasting.", "-", drivers)
    up = [(g, v) for g, v in drivers if v > 0]
    wide_push = sum(v for g, v in up if g in WIDESPREAD_GROUPS)
    own_push = sum(v for g, v in up if g.startswith("Own station"))
    episode = ("Widespread (transboundary-haze signature)" if wide_push > own_push and wide_push > 5
               else "Local build-up" if own_push > 5 else "No episode")
    p = r["p_alert"]
    if p >= thr:
        action = "ALERT"
        reason = f"{p:.0%} chance of API > 100 in the next 24 h (alert threshold {thr:.0%})."
    elif r["hi_24"] > 100:
        action = "WATCH"
        reason = (f"Only {p:.0%} chance of Unhealthy, but the 80% interval for tomorrow reaches "
                  f"{r['hi_24']:.0f}.")
    else:
        action = "CLEAR"
        reason = f"{p:.0%} chance of Unhealthy; tomorrow's 80% interval tops out at {r['hi_24']:.0f}."
        episode = "No episode"
    return Decision(station, action, reason, episode, drivers)


def decide(fc: pd.DataFrame, thr: float) -> pd.DataFrame:
    rows = []
    for s, r in fc.iterrows():
        d = decide_row(s, r, thr)
        lat, lon, state, region = STATIONS[s]
        rows.append({"station": s, "state": state, "region": region, "lat": lat, "lon": lon,
                     "api_now": r["api_now"], "pred_24": r["pred_24"], "lo_24": r["lo_24"], "hi_24": r["hi_24"],
                     "pred_48": r["pred_48"], "lo_48": r["lo_48"], "hi_48": r["hi_48"],
                     "p_alert": r["p_alert"], "action": d.action, "reason": d.reason, "episode": d.episode,
                     "top_drivers": d.drivers[:3],
                     "category_24": api_category(r["pred_24"])})
    return pd.DataFrame(rows).set_index("station")


def explain_driver(group: str, value: float) -> str:
    sign = "pushes up" if value > 0 else "pulls down"
    return f"{group} {sign} tomorrow's forecast by {abs(value):.0f} API points"


def bulletin(dec: pd.DataFrame, issued: pd.Timestamp) -> str:
    """Plain-language bulletin grouped by state."""
    lines = [f"HazeCast MY bulletin - issued {issued:%d %b %Y %H:00} (forecast for the next 24-48 h)", ""]
    n = dec["action"].value_counts()
    lines.append(" | ".join(f"{a}: {int(n.get(a, 0))}" for a in ACTIONS))
    lines.append("")
    hot = dec[dec["action"].isin(["ALERT", "WATCH"])]
    if hot.empty:
        lines.append("No station is expected to reach the Unhealthy band (API > 100) in the next 24 hours.")
    for state, g in hot.groupby("state"):
        worst = g.sort_values("p_alert", ascending=False)
        names = ", ".join(f"{s} ({r['action']}, {r['p_alert']:.0%}, tomorrow ~{r['pred_24']:.0f} "
                          f"[{r['lo_24']:.0f}-{r['hi_24']:.0f}])" for s, r in worst.iterrows())
        ep = worst["episode"].mode().iloc[0]
        lines.append(f"- {state}: {names}. Pattern: {ep}.")
    abst = dec[dec["action"] == "INSUFFICIENT DATA"].index.tolist()
    if abst:
        lines.append("")
        lines.append(f"Abstained (insufficient recent data): {', '.join(abst)}.")
    lines.append("")
    lines.append("Experimental research forecast. Always follow official DOE / APIMS advisories.")
    return "\n".join(lines)
