import numpy as np
import pandas as pd
import pytest

from hazecast import conformal
from hazecast.agent import decide_row
from hazecast.data import api_category
from hazecast.evaluate import lead_times, onset_events
from hazecast.features import FEATURES, build_targets, make_long
from hazecast.stations import STATIONS, nearest_neighbours


def toy_wide(hours=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=hours, freq="h")
    cols = ["Kuching", "Samarahan", "Sri Aman", "Klang", "Shah Alam"]
    data = 40 + rng.normal(0, 5, size=(hours, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols).astype("float32")


def test_features_do_not_leak_future():
    w = toy_wide()
    long_a = make_long(w)
    w2 = w.copy()
    w2.iloc[300:] = 999.0  # change the future only
    long_b = make_long(w2)
    t = w.index[299]
    a = long_a.loc[t, FEATURES]
    b = long_b.loc[t, FEATURES]
    pd.testing.assert_frame_equal(a, b)


def test_targets_look_forward():
    w = toy_wide()
    tg = build_targets(w)
    assert tg["y_24"].iloc[0, 0] == w.iloc[24, 0]
    w.iloc[10, 0] = 150
    tg = build_targets(w)
    assert tg["alert_24"].iloc[0, 0] == 1.0   # t=0 sees hour 10 inside (0, 24]
    assert tg["alert_24"].iloc[10, 0] == 0.0  # hour 10 itself is not in its own future window


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0, 150, 20000)
    std = rng.uniform(0, 10, 20000)
    y = pred + rng.normal(0, 1, 20000) * conformal.difficulty(pred, std) * 0.5
    cal = conformal.fit(pred[:10000], y[:10000], std[:10000], alpha=0.2)
    lo, hi = conformal.interval(pred[10000:], cal, std[10000:])
    rep = conformal.coverage_report(pred[10000:], y[10000:], lo, hi)
    assert 0.77 < rep["empirical_coverage"] < 0.83
    # normalised scores keep coverage roughly equal across levels
    assert abs(rep["by_predicted_band"][">100"]["coverage"] - 0.8) < 0.05


def test_conformal_width_grows_with_volatility_and_level():
    cal = {"q": 1.0}
    lo1, hi1 = conformal.interval(np.array([40.0]), cal, np.array([2.0]))
    lo2, hi2 = conformal.interval(np.array([150.0]), cal, np.array([30.0]))
    assert (hi2 - lo2)[0] > (hi1 - lo1)[0]
    lo3, _ = conformal.interval(np.array([1.0]), {"q": 5.0}, np.array([10.0]))
    assert lo3[0] == 0.0


def test_onset_and_lead_time():
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    w = pd.DataFrame({"Kuching": 50.0}, index=idx)
    w.iloc[120:130, 0] = 150.0
    ev = onset_events(w)
    assert ev == [(idx[120], "Kuching")]
    flags = pd.DataFrame({"Kuching": False}, index=idx)
    flags.iloc[110, 0] = True
    res = lead_times(ev, flags)
    assert res["detected"] == 1 and res["median_lead_h"] == 10.0


def test_agent_abstains_and_alerts():
    base = {"api_now": 60.0, "missing_24": 0.0, "p_alert": 0.05, "hi_24": 80.0,
            "drv::Nationwide haze": 0.0, "drv::Own station: current level": 0.0}
    assert decide_row("Klang", pd.Series({**base, "missing_24": 0.8}), 0.2).action == "INSUFFICIENT DATA"
    assert decide_row("Klang", pd.Series({**base, "p_alert": 0.4}), 0.2).action == "ALERT"
    assert decide_row("Klang", pd.Series({**base, "hi_24": 120.0}), 0.2).action == "WATCH"
    assert decide_row("Klang", pd.Series(base), 0.2).action == "CLEAR"
    d = decide_row("Klang", pd.Series({**base, "p_alert": 0.6, "drv::Nationwide haze": 30.0,
                                       "drv::Own station: current level": 8.0}), 0.2)
    assert d.episode.startswith("Widespread")


@pytest.mark.parametrize("v,cat", [(30, "Good"), (100, "Moderate"), (101, "Unhealthy"), (350, "Hazardous")])
def test_api_bands(v, cat):
    assert api_category(v) == cat


def test_station_metadata():
    assert len(STATIONS) == 67
    nn = nearest_neighbours(3)
    assert "Samarahan" in nn["Kuching"]
