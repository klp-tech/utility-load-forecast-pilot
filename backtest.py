#!/usr/bin/env python3
"""Backtest an off-the-shelf time-series foundation model (Chronos-2) against simple baselines.

For every day in each test window, stand at local midnight, forecast the next `horizon_hours`
using only what would have been known then, and score against what happened.

  naive_week         same hour, most recent week available at the origin
  regression_fx      per-hour-of-day least squares on weather (trained on actual weather from the
                     2 years before each window), driven by the ARCHIVED forecast
  chronos            Chronos-2, load history only, zero-shot (no training on your data)
  chronos_wx_fx      Chronos-2 + archived weather forecast as known-future covariates
  chronos_wx_actual  Chronos-2 + the weather that actually happened. UPPER BOUND ONLY: no live
                     system has this. It shows what weather-forecast error costs. Run at lag 0.
  chronos_feat_fx    Chronos-2 + weather + derived inputs (cooling/heating degree-hours,
                     weekend, holiday; common.add_features)
  regression_trend_fx  regression_fx + a linear time trend (load growth) + holiday flag
  avg_chronos_reg    equal-weight mean of chronos_feat_fx and regression_trend_fx. The two
                     tend to err in opposite directions on extreme days (a foundation model
                     under-reacts to heat beyond its context, a cubic regression over-reacts),
                     so their average can beat both

Data delay: each `lags_hours` value re-runs the backtest with the last load reading that many
hours before the origin. Weather in that gap is observed (it has happened); the future is the
forecast.

Output (output/): forecasts_lag<N>.csv (every hourly forecast), summary.csv
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from chronos import Chronos2Pipeline  # noqa: E402

from common import FEAT_EXTRA, ROOT, WX, add_features, load_config, p, read_hourly  # noqa: E402

FEAT = WX + FEAT_EXTRA


class Run:
    def __init__(self, cfg, df, pipe):
        b = cfg["backtest"]
        self.cfg, self.df, self.pipe = cfg, df, pipe
        self.H, self.ctx = b["horizon_hours"], b["context_hours"]
        self.lag = 0

    # ------------------------------------------------------------ inputs
    def fx_frame(self, o: pd.Timestamp) -> pd.DataFrame:
        """Weather for the gap hours (observed) and the horizon (forecast, lead K for day K)."""
        idx = pd.date_range(o - pd.Timedelta(hours=self.lag), periods=self.H + self.lag, freq="h")
        out = pd.DataFrame(index=idx, columns=WX, dtype=float)
        gap = idx[idx < o]
        out.loc[gap, WX] = self.df.loc[gap, WX].to_numpy()
        fut = idx[idx >= o]
        for k in range(1, 8):
            sl = fut[(k - 1) * 24: k * 24]
            if len(sl):
                for v in WX:
                    out.loc[sl, v] = self.df.loc[sl, f"{v}_d{k}"].to_numpy()
        return out

    def actual_frame(self, o):
        idx = pd.date_range(o - pd.Timedelta(hours=self.lag), periods=self.H + self.lag, freq="h")
        return self.df.loc[idx, WX]

    # ------------------------------------------------------------ models
    def naive_week(self, origins):
        out = {}
        for o in origins:
            idx = pd.date_range(o, periods=self.H, freq="h")
            last = o - pd.Timedelta(hours=self.lag + 1)
            hrs = ((idx - last) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
            weeks = np.maximum(np.ceil(hrs / 168).astype(int), 1)
            src = idx - pd.to_timedelta(weeks * 168, unit="h")
            out[o] = pd.DataFrame({"p50": self.df["load_mw"].reindex(src).values}, index=idx)
        return out

    def _design(self, w: pd.DataFrame, t0=None) -> np.ndarray:
        t = w["temp_f"].to_numpy(dtype=float)
        cols = [np.ones(len(w)), t, t**2, t**3, w["dewpoint_f"], w["cloud_pct"],
                w["solar_wm2"], (w.index.day_of_week >= 5).astype(float)]
        if t0 is not None:  # trend (years from the window start) + holiday flag
            cols += [((w.index - t0) / pd.Timedelta(days=365.25)).to_numpy(dtype=float),
                     add_features(w[WX], self.cfg)["holiday"].to_numpy()]
        return np.column_stack(cols).astype(float)

    def regression_fx(self, origins, win_start, trend=False):
        t0 = win_start if trend else None
        tr = self.df.loc[win_start - pd.Timedelta(days=730): win_start - pd.Timedelta(hours=1),
                         ["load_mw"] + WX].dropna()
        coef = {}
        for hr in range(24):
            x = tr[tr.index.hour == hr]
            coef[hr], *_ = np.linalg.lstsq(self._design(x, t0), x["load_mw"].to_numpy(), rcond=None)
        out = {}
        for o in origins:
            w = self.fx_frame(o).loc[o:]
            x = self._design(w, t0)
            out[o] = pd.DataFrame({"p50": [x[i] @ coef[t.hour] for i, t in enumerate(w.index)]},
                                  index=w.index)
        return out

    def fx_feat_frame(self, o):
        return add_features(self.fx_frame(o), self.cfg)

    def chronos(self, origins, future=None, covs=WX):
        covs = covs if future is not None else []
        hist, fut = [], []
        for o in origins:
            cut = o - pd.Timedelta(hours=self.lag)
            h = self.df.loc[cut - pd.Timedelta(hours=self.ctx): cut - pd.Timedelta(hours=1),
                            ["load_mw"] + covs].copy()
            h["item_id"] = f"{o:%Y-%m-%d}"
            hist.append(h)
            if future is not None:
                fut.append(future(o)[covs].assign(item_id=f"{o:%Y-%m-%d}"))
        pred = self.pipe.predict_df(
            pd.concat(hist).rename_axis("ts").reset_index(),
            future_df=pd.concat(fut).rename_axis("ts").reset_index() if fut else None,
            id_column="item_id", timestamp_column="ts", target="load_mw",
            prediction_length=self.H + self.lag, quantile_levels=[0.1, 0.5, 0.9], batch_size=100)
        by_id = dict(tuple(pred.groupby("item_id")))
        return {o: by_id[f"{o:%Y-%m-%d}"].set_index("ts")[["0.1", "0.5", "0.9"]]
                .set_axis(["p10", "p50", "p90"], axis=1).loc[o:] for o in origins}


def score(actual: pd.Series, f: pd.DataFrame) -> dict:
    a, q = actual.reindex(f.index).to_numpy(), f["p50"].to_numpy(dtype=float)
    a1, q1 = a[:24], q[:24]
    ia, iq = int(np.nanargmax(a1)), int(np.nanargmax(q1))
    r = {"day1_mape_pct": float(np.nanmean(np.abs(q1 - a1) / a1) * 100),
         "day1_wape_pct": float(np.nansum(np.abs(q1 - a1)) / np.nansum(a1) * 100),
         "day1_mae_mw": float(np.nanmean(np.abs(q1 - a1))),
         "week_wape_pct": float(np.nansum(np.abs(q - a)) / np.nansum(a) * 100),
         "peak_err_pct": float((q1[iq] - a1[ia]) / a1[ia] * 100),
         "peak_hour_within_1h": abs(ia - iq) <= 1, "band80_cover_pct": np.nan,
         "day1_pinball_mw": np.nan}
    if "p10" in f and f["p10"].notna().any():
        lo, hi = f["p10"].to_numpy()[:24], f["p90"].to_numpy()[:24]
        r["band80_cover_pct"] = float(np.mean((a1 >= lo) & (a1 <= hi)) * 100)
        # Mean pinball (quantile) loss over the 10/50/90% forecasts: scores the whole range,
        # not just the median. Lower is better; for a point forecast it equals MAE / 2.
        pin = [np.nanmean(np.maximum(qq * (a1 - f[c].to_numpy()[:24]),
                                     (qq - 1) * (a1 - f[c].to_numpy()[:24])))
               for qq, c in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90"))]
        r["day1_pinball_mw"] = float(np.mean(pin))
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    b = cfg["backtest"]
    t0 = time.time()
    df = read_hourly(p(cfg, cfg["data"]["load_csv"]))[["load_mw"]]
    df = df.join(read_hourly(p(cfg, cfg["data"]["weather_csv"]))[WX], how="left")
    df = df.join(read_hourly(p(cfg, cfg["data"]["forecast_csv"])), how="left")
    df["load_mw"] = df["load_mw"].interpolate(limit_area="inside")
    df = df.join(add_features(df[WX], cfg)[FEAT_EXTRA])  # history inputs: actual weather
    pipe = Chronos2Pipeline.from_pretrained(
        b["chronos_model"], device_map="cuda" if torch.cuda.is_available() else "cpu")
    run = Run(cfg, df, pipe)
    last_ok = min(df["load_mw"].last_valid_index(), df["temp_f_d7"].last_valid_index())
    outdir = ROOT / "output"
    outdir.mkdir(exist_ok=True)

    rows = []
    for lag in b["lags_hours"]:
        run.lag = lag
        fc_rows = []
        for w in b["windows"]:
            ws, we = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
            origins = [o for o in pd.date_range(ws, we, freq="D")
                       if o + pd.Timedelta(hours=run.H - 1) <= last_ok]
            if not origins:
                print(f"skip {w['name']}: no data to score it")
                continue
            print(f"lag {lag} h / {w['name']}: {len(origins)} origins", flush=True)
            builders = {
                "naive_week": lambda: run.naive_week(origins),
                "regression_fx": lambda: run.regression_fx(origins, ws),
                "chronos": lambda: run.chronos(origins),
                "chronos_wx_fx": lambda: run.chronos(origins, run.fx_frame),
                "chronos_wx_actual": lambda: run.chronos(origins, run.actual_frame),
                "chronos_feat_fx": lambda: run.chronos(origins, run.fx_feat_frame, FEAT),
                "regression_trend_fx": lambda: run.regression_fx(origins, ws, trend=True),
            }
            built = {}
            for m in b["models"]:
                if m == "chronos_wx_actual" and lag != 0:
                    continue
                if m == "avg_chronos_reg":
                    for need in ("chronos_feat_fx", "regression_trend_fx"):
                        if need not in built:
                            built[need] = builders[need]()
                    fc = {o: ((built["chronos_feat_fx"][o][["p50"]] +
                               built["regression_trend_fx"][o][["p50"]]) / 2) for o in origins}
                else:
                    fc = built[m] if m in built else builders[m]()
                    built[m] = fc
                for o in origins:
                    f = fc[o]
                    rows.append({"lag_h": lag, "window": w["name"], "model": m, "origin": o.date(),
                                 **score(df["load_mw"], f)})
                    fc_rows.append(pd.DataFrame({
                        "window": w["name"], "model": m, "origin": o, "ts": f.index,
                        "h": np.arange(len(f)),
                        "actual": df["load_mw"].reindex(f.index).to_numpy(),
                        "p10": f["p10"].to_numpy() if "p10" in f else np.nan,
                        "p50": f["p50"].to_numpy(),
                        "p90": f["p90"].to_numpy() if "p90" in f else np.nan}))
        pd.concat(fc_rows).round(4).to_csv(outdir / f"forecasts_lag{lag}.csv", index=False)

    daily = pd.DataFrame(rows)
    summary = (daily.groupby(["lag_h", "window", "model"], sort=False)
               .agg(days=("origin", "count"), day1_wape_pct=("day1_wape_pct", "mean"),
                    day1_mae_mw=("day1_mae_mw", "mean"), week_wape_pct=("week_wape_pct", "mean"),
                    peak_abs_err_pct=("peak_err_pct", lambda s: s.abs().mean()),
                    peak_hour_hit_pct=("peak_hour_within_1h", lambda s: s.mean() * 100),
                    band80_cover_pct=("band80_cover_pct", "mean"),
                    day1_pinball_mw=("day1_pinball_mw", "mean"))
               .round(3).reset_index())
    # Skill vs the naive baseline (a scaled error, in the spirit of MASE): 1 - WAPE/WAPE_naive.
    # Positive = better than "same hour last week"; 0.5 = half the error.
    nv = summary[summary.model == "naive_week"].set_index(["lag_h", "window"])["day1_wape_pct"]
    summary["skill_vs_naive"] = (1 - summary["day1_wape_pct"] /
                                 summary.set_index(["lag_h", "window"]).index.map(nv)).round(3)
    summary.to_csv(outdir / "summary.csv", index=False)
    daily.to_csv(outdir / "daily_scores.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))
    print(f"\nwrote {outdir} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
