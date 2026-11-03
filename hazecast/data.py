"""Load and clean the DOE APIMS hourly Air Pollutant Index archive."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stations import STATIONS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RENAME = {"Kulim Hi-Tech": "Kulim"}


def load_wide(data_dir: Path | str = DATA_DIR, start: str | None = "2013-01-01") -> pd.DataFrame:
    """Return an hourly-grid wide frame: index = timestamp, columns = stations.

    Missing hours are NaN. Readings <= 0 or > 1000 are treated as sensor errors.
    """
    data_dir = Path(data_dir)
    frames = []
    for f in sorted(data_dir.glob("apims-*.csv")):
        d = pd.read_csv(f)
        d = d.rename(columns=RENAME)
        d["Time"] = pd.to_datetime(d["Time"], format="ISO8601")
        frames.append(d)
    wide = pd.concat(frames, ignore_index=True).set_index("Time").sort_index()
    wide = wide[~wide.index.duplicated(keep="last")]
    wide = wide.apply(pd.to_numeric, errors="coerce").astype("float32")
    wide = wide.where((wide > 0) & (wide <= 1000))
    wide = wide[[c for c in wide.columns if c in STATIONS]]
    wide.index = wide.index.floor("h")
    wide = wide.groupby(level=0).mean()
    if start:
        wide = wide.loc[start:]
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    wide = wide.reindex(grid)
    wide.index.name = "time"
    return wide


def api_category(v: float) -> str:
    """DOE API bands."""
    if v is None or np.isnan(v):
        return "No data"
    if v <= 50:
        return "Good"
    if v <= 100:
        return "Moderate"
    if v <= 200:
        return "Unhealthy"
    if v <= 300:
        return "Very unhealthy"
    return "Hazardous"


CATEGORY_COLOURS = {
    "Good": "#2E7D32",
    "Moderate": "#F9A825",
    "Unhealthy": "#E65100",
    "Very unhealthy": "#B71C1C",
    "Hazardous": "#4A148C",
    "No data": "#9E9E9E",
}
