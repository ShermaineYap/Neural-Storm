"""HazeCast MY - Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hazecast.agent import ACTION_COLOURS, ACTIONS, bulletin, decide, explain_driver
from hazecast.data import CATEGORY_COLOURS, api_category, load_wide
from hazecast.forecast import Forecaster, fold_for
from hazecast.stations import STATIONS

ROOT = Path(__file__).parent
st.set_page_config(page_title="HazeCast MY", page_icon="🌫️", layout="wide")

INK, SMOKE, OCHRE, MUTED = "#1D2B33", "#E8ECEE", "#B7791F", "#5B6B73"
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label {{ font-family: 'IBM Plex Sans', sans-serif; color: {INK}; }}
.stApp {{ background: {SMOKE}; }}
h1, h2, h3, .headline {{ font-family: 'Bricolage Grotesque', sans-serif !important; color: {INK}; letter-spacing: -0.01em; }}
.headline {{ font-size: clamp(1.9rem, 4.2vw, 3.3rem); line-height: 1.08; font-weight: 700; margin: 0.2rem 0 0.6rem; max-width: 22ch; }}
.subline {{ font-size: 1.05rem; color: {MUTED}; max-width: 70ch; margin-bottom: 1.2rem; }}
.pill {{ display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-weight: 600; font-size: 0.85rem; color: white; }}
.drv {{ margin: 0.15rem 0; }}
.note {{ font-size: 0.85rem; color: {MUTED}; }}
section[data-testid="stSidebar"] {{ background: #DCE2E5; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading 9 years of hourly DOE readings…")
def get_forecaster() -> Forecaster:
    return Forecaster(wide=load_wide(ROOT / "data"), art_dir=ROOT / "artifacts")


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return json.loads((ROOT / "artifacts" / "metrics.json").read_text())


@st.cache_data(show_spinner="Forecasting every station…")
def forecast_at(t: str) -> pd.DataFrame:
    fc = get_forecaster()
    raw = fc.forecast(pd.Timestamp(t))
    return raw


@st.cache_data(show_spinner="Building the station timeline…")
def station_series(station: str, t_from: str, t_to: str) -> pd.DataFrame:
    return get_forecaster().series(station, pd.Timestamp(t_from), pd.Timestamp(t_to))


fc = get_forecaster()
metrics = get_metrics()
wide = fc.wide
data_end = wide.index.max()

PRESETS = {
    "Sept 2019 haze: Klang Valley on the brink (16 Sep, 08:00)": "2019-09-16 08:00",
    "Sept 2019 haze: Sarawak onset (09 Sep, 08:00)": "2019-09-09 08:00",
    "Aug 2019 haze: first warning signs (01 Aug, 08:00)": "2019-08-01 08:00",
    "Quiet day: monsoon season (15 Jan 2021, 08:00)": "2021-01-15 08:00",
    f"Latest data in archive ({data_end:%d %b %Y %H:00})": str(data_end),
}

with st.sidebar:
    st.markdown("### Forecast issued at")
    choice = st.radio("Scenario", list(PRESETS) + ["Pick a date and hour"], index=0, label_visibility="collapsed")
    if choice == "Pick a date and hour":
        d = st.date_input("Date", value=pd.Timestamp("2019-09-15").date(),
                          min_value=pd.Timestamp("2014-01-15").date(), max_value=data_end.date())
        h = st.slider("Hour", 0, 23, 8)
        issue = pd.Timestamp(d) + pd.Timedelta(hours=h)
    else:
        issue = pd.Timestamp(PRESETS[choice])
    issue = min(issue, data_end)
    st.markdown(
        f"<p class='note'>Model used: walk-forward fold <b>{fold_for(issue)}</b>. "
        "Forecasts inside 2019–2022 come from a model that never saw that period.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='note'>Data: DOE APIMS hourly Air Pollutant Index, 67 stations, 2013 – May 2022 "
        "(data.gov.my, CC BY; 2018+ scraped by ynshung/api-malaysia).</p>",
        unsafe_allow_html=True,
    )

raw = forecast_at(str(issue))
dec = decide(raw, fc.thr)
n_alert = int((dec["action"] == "ALERT").sum())
n_watch = int((dec["action"] == "WATCH").sum())
n_valid = int((dec["action"] != "INSUFFICIENT DATA").sum())

if n_alert:
    head = f"Tomorrow, {n_alert} of {n_valid} stations may breathe unhealthy air."
elif n_watch:
    head = f"No alerts tomorrow, but {n_watch} station{'s' if n_watch > 1 else ''} worth watching."
else:
    head = "Clear air expected at every station tomorrow."
st.markdown(f"<div class='headline'>{head}</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='subline'>HazeCast MY forecasts the Air Pollutant Index 24 and 48 hours ahead for Malaysia's "
    f"DOE monitoring stations, with an 80% prediction interval and the reason behind each forecast. "
    f"Issued {issue:%A %d %B %Y, %H:00}.</div>",
    unsafe_allow_html=True,
)

tab_map, tab_station, tab_evidence, tab_about = st.tabs(
    ["Map and bulletin", "Station detail", "Model evidence", "How it works"])

# ---------------------------------------------------------------- map
with tab_map:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        horizon = st.radio("Show", ["Tomorrow (+24 h)", "Day after (+48 h)", "Agent decision"],
                           horizontal=True, label_visibility="collapsed")
        df = dec.reset_index()
        if horizon == "Agent decision":
            colours = df["action"].map(ACTION_COLOURS)
            label = df["action"]
        else:
            col = "pred_24" if horizon.startswith("Tomorrow") else "pred_48"
            cats = df[col].apply(api_category)
            cats = cats.where(df["action"] != "INSUFFICIENT DATA", "No data")
            colours = cats.map(CATEGORY_COLOURS)
            label = cats
        hover = (df["station"] + " (" + df["state"] + ")<br>Now: " + df["api_now"].round(0).astype("Int64").astype(str)
                 + "<br>+24 h: " + df["pred_24"].round(0).astype(int).astype(str)
                 + " [" + df["lo_24"].round(0).astype(int).astype(str) + "–" + df["hi_24"].round(0).astype(int).astype(str) + "]"
                 + "<br>P(Unhealthy in 24 h): " + (df["p_alert"] * 100).round(0).astype(int).astype(str) + "%"
                 + "<br>" + df["action"])
        fig = go.Figure(go.Scattermap(
            lat=df["lat"], lon=df["lon"], mode="markers",
            marker=dict(size=np.clip(9 + df["pred_24"] / 9, 10, 34), color=colours, opacity=0.9),
            text=hover, hoverinfo="text"))
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=4.0, lon=108.5), zoom=4.3),
                          margin=dict(l=0, r=0, t=0, b=0), height=470, paper_bgcolor=SMOKE)
        st.plotly_chart(fig, use_container_width=True)
        legend = (ACTION_COLOURS.items() if horizon == "Agent decision"
                  else [(k, v) for k, v in CATEGORY_COLOURS.items()])
        st.markdown(" ".join(f"<span class='pill' style='background:{v}'>{k}</span>" for k, v in legend),
                    unsafe_allow_html=True)
    with c2:
        st.markdown("#### HazeAgent bulletin")
        st.code(bulletin(dec, issue), language=None, wrap_lines=True)
        top = dec.sort_values("p_alert", ascending=False).head(8)
        st.markdown("#### Highest risk tomorrow")
        st.dataframe(
            top[["state", "api_now", "pred_24", "lo_24", "hi_24", "p_alert", "action"]].rename(columns={
                "state": "State", "api_now": "Now", "pred_24": "+24 h", "lo_24": "Low", "hi_24": "High",
                "p_alert": "P(Unhealthy)", "action": "Decision"}),
            column_config={"P(Unhealthy)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                           "Now": st.column_config.NumberColumn(format="%.0f"),
                           "+24 h": st.column_config.NumberColumn(format="%.0f"),
                           "Low": st.column_config.NumberColumn(format="%.0f"),
                           "High": st.column_config.NumberColumn(format="%.0f")},
            use_container_width=True)

# ---------------------------------------------------------------- station
with tab_station:
    default = dec["p_alert"].idxmax()
    names = sorted(STATIONS)
    station = st.selectbox("Station", names, index=names.index(default))
    r = dec.loc[station]
    a, b, c, d_ = st.columns(4)
    a.metric("API now", "–" if np.isnan(r["api_now"]) else f"{r['api_now']:.0f}", api_category(r["api_now"]))
    b.metric("+24 h forecast", f"{r['pred_24']:.0f}", f"80% interval {r['lo_24']:.0f}–{r['hi_24']:.0f}", delta_color="off")
    c.metric("+48 h forecast", f"{r['pred_48']:.0f}", f"80% interval {r['lo_48']:.0f}–{r['hi_48']:.0f}", delta_color="off")
    d_.metric("Chance of Unhealthy in 24 h", f"{r['p_alert']:.0%}", r["action"], delta_color="off")
    st.markdown(f"<span class='pill' style='background:{ACTION_COLOURS[r['action']]}'>{r['action']}</span> "
                f"{r['reason']} Episode type: <b>{r['episode']}</b>.", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        t_from = issue - pd.Timedelta(days=7)
        ser = station_series(station, str(t_from), str(issue))
        actual = wide[station].loc[t_from: issue + pd.Timedelta(hours=48)]
        past24 = ser[["pred_24", "lo_24", "hi_24", "p_alert"]].copy()
        past24.index = past24.index + pd.Timedelta(hours=24)
        fg = go.Figure()
        fg.add_trace(go.Scatter(x=past24.index, y=past24["hi_24"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fg.add_trace(go.Scatter(x=past24.index, y=past24["lo_24"], fill="tonexty", fillcolor="rgba(183,121,31,0.18)",
                                line=dict(width=0), name="80% interval (issued 24 h earlier)"))
        fg.add_trace(go.Scatter(x=past24.index, y=past24["pred_24"], line=dict(color=OCHRE, width=2, dash="dot"),
                                name="24-h-ahead forecast"))
        obs_hist = actual.loc[:issue]
        obs_future = actual.loc[issue:]
        fg.add_trace(go.Scatter(x=obs_hist.index, y=obs_hist, line=dict(color=INK, width=2), name="Observed API"))
        if len(obs_future) > 1:
            fg.add_trace(go.Scatter(x=obs_future.index, y=obs_future, line=dict(color=INK, width=1.5, dash="dash"),
                                    name="What actually happened next", opacity=0.55))
        for h in (24, 48):
            fg.add_trace(go.Scatter(
                x=[issue + pd.Timedelta(hours=h)], y=[r[f"pred_{h}"]], mode="markers",
                marker=dict(color=OCHRE, size=12, line=dict(color=INK, width=1)),
                error_y=dict(type="data", symmetric=False, array=[r[f"hi_{h}"] - r[f"pred_{h}"]],
                             arrayminus=[r[f"pred_{h}"] - r[f"lo_{h}"]], color=OCHRE, thickness=2),
                name=f"Forecast +{h} h", showlegend=False))
        fg.add_hline(y=100, line=dict(color=CATEGORY_COLOURS["Unhealthy"], width=1, dash="dash"),
                     annotation_text="Unhealthy (100)", annotation_position="top left")
        fg.add_vline(x=issue, line=dict(color=MUTED, width=1))
        fg.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=SMOKE, plot_bgcolor="white",
                         legend=dict(orientation="h", y=-0.15), yaxis_title="API", font=dict(family="IBM Plex Sans"))
        st.plotly_chart(fg, use_container_width=True)
        st.markdown("<p class='note'>Dotted line: what HazeCast predicted for each hour, issued 24 hours before it. "
                    "Grey vertical line: issue time. Dashed: the real readings after the forecast was made.</p>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### Why this forecast")
        drv = raw.loc[station]
        groups = {k[5:]: v for k, v in drv.items() if str(k).startswith("drv::")}
        groups = dict(sorted(groups.items(), key=lambda kv: abs(kv[1]), reverse=True))
        base = drv["base_value"]
        now_txt = "–" if np.isnan(drv["api_now"]) else f"{drv['api_now']:.0f}"
        st.markdown(f"<p class='note'>Tomorrow's forecast = API now ({now_txt}) + typical 24-h change "
                    f"({base:+.0f}) + the drivers below. Contributions are exact TreeSHAP values from the "
                    f"+24 h model.</p>", unsafe_allow_html=True)
        gb = go.Figure(go.Bar(
            x=list(groups.values())[::-1], y=list(groups.keys())[::-1], orientation="h",
            marker_color=[OCHRE if v > 0 else "#4F7A8A" for v in list(groups.values())[::-1]]))
        gb.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=SMOKE, plot_bgcolor="white",
                         xaxis_title="API points added to tomorrow's forecast", font=dict(family="IBM Plex Sans"))
        st.plotly_chart(gb, use_container_width=True)
        for g, v in list(groups.items())[:3]:
            st.markdown(f"<p class='drv'>• {explain_driver(g, v)}.</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------- evidence
with tab_evidence:
    st.markdown("### Walk-forward results")
    st.markdown(
        "Each test period is forecast by a model trained only on earlier years "
        "(2013–2018 → 2019, 2013–2019 → 2020, 2013–2020 → 2021–22). No random splits.")
    folds = metrics["folds"]
    rows = []
    for fname, f in folds.items():
        for model, m in f["regression_h24"].items():
            rows.append({"Test period": fname.split("_", 1)[1], "Model": model, "MAE": m["MAE"],
                         "RMSE": m["RMSE"], "MAE on haze hours (API>100)": m.get("MAE_haze", np.nan),
                         "Skill vs persistence": m["skill_vs_persistence"]})
    t24 = pd.DataFrame(rows)
    pivot = t24.pivot_table(index="Model", columns="Test period", values="MAE").round(2)
    pivot["Mean"] = pivot.mean(axis=1).round(2)
    st.markdown("**+24 h forecast error (MAE, API points; lower is better)**")
    st.dataframe(pivot.sort_values("Mean"), use_container_width=True)
    pivot_h = t24.pivot_table(index="Model", columns="Test period", values="MAE on haze hours (API>100)").round(1)
    st.markdown("**+24 h error on haze hours only (true API > 100)**")
    st.dataframe(pivot_h.sort_values(pivot_h.columns[0]), use_container_width=True)

    st.markdown("**Unhealthy-within-24-h alerts**")
    arows = []
    for fname, f in folds.items():
        for model, m in f["alert_24h"].items():
            arows.append({"Test period": fname.split("_", 1)[1], "Method": model,
                          "Precision": m["precision"], "Recall": m["recall"], "F1": m["F1"],
                          "PR-AUC": m["PR_AUC"], "Alerts issued": m["alerts"], "Positives": m["positives"]})
    st.dataframe(pd.DataFrame(arows).round(3), use_container_width=True, hide_index=True)

    st.markdown("**Haze onsets: first hour a station crosses 100 after two clear days**")
    lrows = []
    for fname, f in folds.items():
        for model, m in f["onset_lead_time"].items():
            lrows.append({"Test period": fname.split("_", 1)[1], "Method": model, "Onsets": m["events"],
                          "Warned in advance": m["detected"], "Detection rate": m["detection_rate"],
                          "Median warning (h)": m["median_lead_h"]})
    st.dataframe(pd.DataFrame(lrows).round(2), use_container_width=True, hide_index=True)

    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        st.markdown("**Prediction-interval coverage (target 80%)**")
        crow = []
        for fname, f in folds.items():
            for h in (24, 48):
                cf = f[f"conformal_h{h}"]
                crow.append({"Test period": fname.split("_", 1)[1], "Horizon": f"+{h} h",
                             "Coverage": cf["empirical_coverage"], "Mean width": cf["mean_width"],
                             "Coverage when forecast > 100": cf["by_predicted_band"].get(">100", {}).get("coverage", np.nan)})
        st.dataframe(pd.DataFrame(crow).round(3), use_container_width=True, hide_index=True)
        st.markdown("**Alert probability calibration (reliability)**")
        rel = folds[list(folds)[0]]["reliability"]
        rf = go.Figure()
        rf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=MUTED, dash="dash"), name="Perfect"))
        for fname, f in folds.items():
            rr = f["reliability"]
            rf.add_trace(go.Scatter(x=[p["p_mean"] for p in rr], y=[p["freq"] for p in rr], mode="lines+markers",
                                    name=fname.split("_", 1)[1]))
        rf.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=SMOKE, plot_bgcolor="white",
                         xaxis_title="Predicted probability", yaxis_title="Observed frequency")
        st.plotly_chart(rf, use_container_width=True)
    with cc2:
        st.markdown("**What the +24 h model relies on (mean |SHAP|, 2021–22 test)**")
        sh = metrics.get("shap_global_reg24", {})
        top = list(sh.items())[:12][::-1]
        sf = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h", marker_color=OCHRE))
        sf.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=SMOKE, plot_bgcolor="white",
                         xaxis_title="Mean |contribution| (API points)")
        st.plotly_chart(sf, use_container_width=True)
        lat = metrics.get("latency_all_stations_s", {})
        if lat:
            st.markdown(f"Forecasting all {lat['n_stations']} stations for one hour, including feature building, "
                        f"takes **{lat['median']*1000:.0f} ms** on a 2-core CPU.")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(Path(ROOT / "docs" / "how_it_works.md").read_text() if (ROOT / "docs" / "how_it_works.md").exists()
                else "See the README on GitHub.")

st.markdown("<p class='note'>Experimental research forecast built for Neural Storm 2026 (GP-01). "
            "Not an official DOE product; always follow APIMS and DOE advisories.</p>", unsafe_allow_html=True)
