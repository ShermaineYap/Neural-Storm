"""Train and evaluate HazeCast MY with walk-forward (expanding-window) validation.

Folds (test periods never seen in training):
  F1: train 2013-2018  -> test 2019        (the severe September 2019 haze)
  F2: train 2013-2019  -> test 2020
  F3: train 2013-2020  -> test 2021-01 .. 2022-05

Within each training window the final 15% of hours is held out as a
calibration set, used only for the conformal intervals and the matched-budget
alert threshold (models are never fitted or early-stopped on it). The alert threshold is
fixed in advance from a cost ratio (a missed Unhealthy day costs 4x a false
alarm -> p >= 1/(1+4) = 0.2); it is never tuned on test data.

Usage:  python train.py            # full run, writes artifacts/
        python train.py --quick    # small subsample for a smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hazecast import conformal
from hazecast.data import load_wide
from hazecast.evaluate import clf_metrics, lead_times, onset_events, reg_metrics, reliability
from hazecast.features import FEATURES, make_long

ART = Path("artifacts")
FOLDS = [
    ("F1_2019", "2019-01-01", "2020-01-01"),
    ("F2_2020", "2020-01-01", "2021-01-01"),
    ("F3_2021-22", "2021-01-01", "2022-06-01"),
]
COST_RATIO = 4.0
ALERT_THR = 1.0 / (1.0 + COST_RATIO)
RULE_THR = 80.0  # simple heuristic baseline: warn when API is already >= 80
# Tree counts are fixed in advance (no early stopping): the calibration window straddles
# the 2019 haze, and early-stopping on it stopped one fold after 12 trees. Keeping the
# calibration set out of model fitting also keeps the conformal intervals clean.
LGB_REG = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               n_jobs=2, verbose=-1)
LGB_CLF = dict(LGB_REG, objective="binary")
LINEAR_FEATS = [f for f in FEATURES if f != "station_code"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit_lgb_reg(Xtr, ytr, Xca, yca):
    m = lgb.LGBMRegressor(**LGB_REG)
    m.fit(Xtr, ytr, categorical_feature=["station_code"])
    return m


def fit_lgb_clf(Xtr, ytr, Xca, yca):
    m = lgb.LGBMClassifier(**LGB_CLF)
    m.fit(Xtr, ytr, categorical_feature=["station_code"])
    return m


def climatology(train: pd.DataFrame) -> pd.Series:
    t = train.index.get_level_values("time")
    key = pd.DataFrame({"station": train.index.get_level_values("station"),
                        "month": t.month, "hour": t.hour, "api": train["api"].to_numpy()})
    return key.groupby(["station", "month", "hour"])["api"].mean()


def clim_predict(clim: pd.Series, df: pd.DataFrame, horizon: int) -> np.ndarray:
    t = df.index.get_level_values("time") + pd.Timedelta(hours=horizon)
    key = pd.MultiIndex.from_arrays([df.index.get_level_values("station"), t.month, t.hour])
    return clim.reindex(key).to_numpy(dtype=float)


def main(quick: bool = False):
    ART.mkdir(exist_ok=True)
    (ART / "models").mkdir(exist_ok=True)
    t0 = time.time()
    wide = load_wide()
    long = make_long(wide)
    log("long table", long.shape, f"{time.time() - t0:.1f}s")
    times = long.index.get_level_values("time")

    results = {"folds": {}, "config": {
        "alert_threshold": ALERT_THR, "cost_ratio_miss_to_false_alarm": COST_RATIO,
        "rule_baseline_threshold": RULE_THR, "conformal_alpha": 0.2,
        "regression_target": "delta: y(t+h) - API(t); forecast = API(t) + model",
        "ensemble": "0.5 * LightGBM + 0.5 * Ridge (weights fixed a priori)",
        "features": FEATURES, "train_start": str(wide.index.min()),
        "data_end": str(wide.index.max()), "n_station_hours": int(len(long)),
        "n_stations": int(wide.shape[1]),
    }}
    conformal_all = {}

    for name, test_start, test_end in FOLDS:
        log("=== fold", name)
        tr_mask = times < test_start
        te_mask = (times >= test_start) & (times < test_end)
        train, test = long[tr_mask], long[te_mask]
        ttimes = train.index.get_level_values("time").unique().sort_values()
        cal_start = ttimes[int(len(ttimes) * 0.85)]
        fit_part = train[train.index.get_level_values("time") < cal_start]
        cal_part = train[train.index.get_level_values("time") >= cal_start]
        # thin the fit set to every 2nd issue hour (neighbouring hours are near-duplicates)
        hrs = fit_part.index.get_level_values("time").hour
        fit_part = fit_part[hrs % 2 == 0]
        if quick:
            fit_part = fit_part.sample(min(len(fit_part), 200_000), random_state=0)
            test = test.sample(min(len(test), 200_000), random_state=0).sort_index()
        log("fit", len(fit_part), "cal", len(cal_part), "test", len(test), "cal_start", cal_start)

        fold = {"train_end": test_start, "calibration_start": str(cal_start),
                "test": [test_start, test_end], "n_fit": int(len(fit_part)),
                "n_cal": int(len(cal_part)), "n_test": int(len(test))}
        clim = climatology(train)
        X_te = test[FEATURES]

        for h in (24, 48):
            yname = f"y_{h}"
            ft = fit_part[fit_part[yname].notna()]
            ca = cal_part[cal_part[yname].notna()]
            y_te = test[yname].to_numpy(dtype=float)
            preds = {
                "Persistence (API now)": test["api"].to_numpy(dtype=float),
                "24-h mean persistence": test["mean_24"].to_numpy(dtype=float),
                "Climatology (station x month x hour)": clim_predict(clim, test, h),
            }
            # learned models forecast the CHANGE from the current reading (y_{t+h} - API_t);
            # trees cannot extrapolate levels beyond the training range, but changes they can.
            d_ft = ft[yname] - ft["api"]
            d_ca = ca[yname] - ca["api"]
            api_te = test["api"].to_numpy(dtype=float)
            t1 = time.time()
            ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
            ridge.fit(ft[LINEAR_FEATS], d_ft)
            preds["Ridge regression"] = api_te + ridge.predict(test[LINEAR_FEATS])
            log(f"ridge h{h} {time.time() - t1:.1f}s")
            if h == 24:
                t1 = time.time()
                sub = ft.sample(min(len(ft), 300_000), random_state=0)
                mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                 max_iter=40, random_state=0))
                mlp.fit(sub[LINEAR_FEATS], sub[yname] - sub["api"])
                preds["MLP neural net (64-32)"] = api_te + mlp.predict(test[LINEAR_FEATS])
                log(f"mlp {time.time() - t1:.1f}s")
            t1 = time.time()
            m = fit_lgb_reg(ft[FEATURES], d_ft, ca[FEATURES], d_ca)
            log(f"lgb reg h{h} {time.time() - t1:.1f}s")
            preds["LightGBM"] = api_te + m.predict(X_te)
            m.booster_.save_model(str(ART / "models" / f"{name}_reg{h}.txt"))
            joblib.dump(ridge, ART / "models" / f"{name}_ridge{h}.joblib")
            # HazeCast = equal-weight ensemble of LightGBM (non-linear, strong on normal days)
            # and Ridge (extrapolates linearly into record haze levels). Weights fixed a priori.
            preds["HazeCast ensemble (LightGBM + Ridge)"] = 0.5 * (preds["LightGBM"] + preds["Ridge regression"])

            api_ca = ca["api"].to_numpy(dtype=float)
            cal_pred = api_ca + 0.5 * (m.predict(ca[FEATURES]) + ridge.predict(ca[LINEAR_FEATS]))
            cal = conformal.fit(cal_pred, ca[yname].to_numpy(dtype=float), ca["std_24"].to_numpy(), alpha=0.2)
            conformal_all[f"{name}_reg{h}"] = cal
            ens = preds["HazeCast ensemble (LightGBM + Ridge)"]
            lo, hi = conformal.interval(ens, cal, test["std_24"].to_numpy())
            pers_mae = reg_metrics(y_te, preds["Persistence (API now)"])["MAE"]
            fold[f"regression_h{h}"] = {
                k: dict(reg_metrics(y_te, v), skill_vs_persistence=1 - reg_metrics(y_te, v)["MAE"] / pers_mae)
                for k, v in preds.items()
            }
            fold[f"conformal_h{h}"] = dict(conformal.coverage_report(ens, y_te, lo, hi),
                                           target_coverage=0.8, q=cal["q"],
                                           n_cal_pred_over_100=cal["n_cal_pred_over_100"])

        # ---- alert classifier ----
        ft = fit_part[fit_part["alert_24"].notna()]
        ca = cal_part[cal_part["alert_24"].notna()]
        t1 = time.time()
        clf = fit_lgb_clf(ft[FEATURES], ft["alert_24"], ca[FEATURES], ca["alert_24"])
        log(f"lgb clf {time.time() - t1:.1f}s")
        clf.booster_.save_model(str(ART / "models" / f"{name}_alert24.txt"))
        y_te = test["alert_24"].to_numpy(dtype=float)
        p = clf.predict_proba(X_te)[:, 1]
        api_now = test["api"].to_numpy(dtype=float)
        rule_p = (api_now >= RULE_THR).astype(float)
        pers_p = (api_now > 100).astype(float)
        # threshold giving the SAME alert rate as the >=80 rule, chosen on the calibration set
        p_ca = clf.predict_proba(ca[FEATURES])[:, 1]
        rule_rate_ca = float(np.mean(ca["api"].to_numpy() >= RULE_THR))
        thr_budget = float(np.quantile(p_ca, 1 - rule_rate_ca)) if rule_rate_ca > 0 else ALERT_THR
        fold["matched_budget_threshold"] = {"rule_alert_rate_cal": rule_rate_ca, "threshold": thr_budget}
        fold["alert_24h"] = {
            "LightGBM classifier (p>=0.2)": clf_metrics(y_te, p, ALERT_THR),
            "LightGBM classifier (same alert budget as rule)": clf_metrics(y_te, p, thr_budget),
            "LightGBM classifier (p>=0.5)": clf_metrics(y_te, p, 0.5),
            "Rule: API now >= 80": clf_metrics(y_te, rule_p, 0.5),
            "Persistence: API now > 100": clf_metrics(y_te, pers_p, 0.5),
        }
        ok = ~np.isnan(y_te)
        fold["reliability"] = reliability(y_te[ok], p[ok])

        # lead time on haze onsets inside the test period (full test set, not subsample)
        te_wide = wide.loc[test_start:test_end]
        events = onset_events(te_wide)
        flags = {}
        for lab, vec in [("LightGBM classifier (p>=0.2)", p >= ALERT_THR),
                         ("LightGBM classifier (same alert budget as rule)", p >= thr_budget),
                         ("Rule: API now >= 80", rule_p > 0)]:
            s = pd.Series(vec, index=test.index).unstack("station")
            flags[lab] = s.reindex(index=te_wide.index, columns=te_wide.columns)
        fold["onset_lead_time"] = {k: lead_times(events, v) for k, v in flags.items()}

        results["folds"][name] = fold
        log(json.dumps({k: fold["regression_h24"][k]["MAE"] for k in fold["regression_h24"]}, indent=0))
        log(json.dumps(fold["alert_24h"]["LightGBM classifier (p>=0.2)"]))
        log(json.dumps(fold["onset_lead_time"]))

        if name == FOLDS[-1][0]:
            # global SHAP-style attribution via LightGBM's exact TreeSHAP contributions
            samp = test.sample(min(len(test), 20_000), random_state=0)
            from hazecast.forecast import Forecaster, ensemble_contributions
            with open(ART / "conformal.json", "w") as f:
                json.dump(conformal_all, f, indent=2)
            fc0 = Forecaster(wide=wide)
            contrib = ensemble_contributions(fc0, samp[FEATURES], name, 24)[0]
            imp = np.abs(contrib).mean(axis=0)
            results["shap_global_reg24"] = dict(sorted(zip(FEATURES, imp.astype(float).tolist()),
                                                       key=lambda kv: -kv[1]))
            hz = samp["y_24"].to_numpy() > 100
            if hz.sum() > 20:
                imp_h = np.abs(contrib[hz]).mean(axis=0)
                results["shap_haze_hours_reg24"] = dict(sorted(zip(FEATURES, imp_h.astype(float).tolist()),
                                                               key=lambda kv: -kv[1]))
            # latency: forecast every station for one issue hour, including feature build
            fc = fc0
            t_issue = wide.index[-1]
            fc.forecast(t_issue)
            runs = []
            for _ in range(5):
                t1 = time.perf_counter()
                fc.forecast(t_issue)
                runs.append(time.perf_counter() - t1)
            results["latency_all_stations_s"] = {"median": float(np.median(runs)), "n_stations": int(wide.shape[1])}

    # summary across folds
    def avg(path):
        vals = []
        for f in results["folds"].values():
            v = f
            for k in path:
                v = v[k]
            vals.append(v)
        return float(np.mean(vals))

    summary = {}
    for h in (24, 48):
        summary[f"MAE_h{h}"] = {k: avg([f"regression_h{h}", k, "MAE"])
                                for k in results["folds"][FOLDS[0][0]][f"regression_h{h}"]}
        summary[f"coverage_h{h}"] = avg([f"conformal_h{h}", "empirical_coverage"])
    results["summary"] = summary
    results["runtime_s"] = time.time() - t0
    with open(ART / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    with open(ART / "conformal.json", "w") as f:
        json.dump(conformal_all, f, indent=2)
    log("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
