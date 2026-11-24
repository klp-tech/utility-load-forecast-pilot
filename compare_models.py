#!/usr/bin/env python3
"""Compare open-weight forecasting foundation models on your load, two ways each.

  <model>_feat   the model is given weather + degree-hours + calendar as known inputs
                 (only models that accept them: Chronos-2, t0-gamma, TiRex-2, Timestral)
  <model>_resid  a weather + trend regression (backtest.py regression_trend_fx) is the
                 baseline; the model forecasts what is left over, and the baseline is added back.
                 Any model can run this way, with or without covariate support.
plus
  regression_trend_fx   the baseline on its own
  avg_feat              equal-weight mean of every *_feat model
  avg_feat_reg          0.5 x avg_feat + 0.5 x regression_trend_fx

Same honest set-up as backtest.py: forecasts from every local midnight in each test window,
archived weather forecasts (not actual weather), and a data delay (`compare.lag_hours`).
Weights are fixed and equal, chosen before scoring, so nothing is tuned on the test windows.

Output (output/compare/): forecasts.csv, summary.csv (WAPE by forecast day, daily-peak error
on all days and on the top 10% of days, range coverage)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import foundation_models as fm
from backtest import FEAT, Run
from common import FEAT_EXTRA, ROOT, WX, add_features, load_config, p, read_hourly


def build_inputs(run: Run, origins, target: pd.Series, ctx_h: int):
    hl = run.H + run.lag
    ctx, cov = [], []
    for o in origins:
        cut = o - pd.Timedelta(hours=run.lag)
        hidx = pd.date_range(cut - pd.Timedelta(hours=ctx_h), periods=ctx_h, freq="h")
        ctx.append(target.reindex(hidx).interpolate(limit_direction="both").to_numpy(np.float32))
        fut = run.fx_feat_frame(o)[FEAT]  # gap (observed) + horizon (archived forecast)
        c = pd.concat([run.df.loc[hidx, FEAT], fut]).astype(float).ffill().bfill()
        cov.append(c.to_numpy(np.float32).T)
    return np.stack(ctx), np.stack(cov), hl


def to_frames(run: Run, origins, p10, p50, p90):
    out = {}
    for i, o in enumerate(origins):
        idx = pd.date_range(o - pd.Timedelta(hours=run.lag), periods=p50.shape[1], freq="h")
        out[o] = pd.DataFrame({"p10": p10[i], "p50": p50[i], "p90": p90[i]}, index=idx).loc[o:].iloc[:run.H]
    return out


def baseline(run: Run, win_start):
    """Weather + trend regression: (history fit on actual weather, future on forecast)."""
    tr = run.df.loc[win_start - pd.Timedelta(days=730): win_start - pd.Timedelta(hours=1),
                    ["load_mw"] + WX].dropna()
    coef = {}
    for hr in range(24):
        x = tr[tr.index.hour == hr]
        coef[hr], *_ = np.linalg.lstsq(run._design(x, win_start), x["load_mw"].to_numpy(), rcond=None)

    def predict(w: pd.DataFrame) -> pd.Series:
        x = run._design(w, win_start)
        hrs = w.index.hour.to_numpy()
        y = np.empty(len(w))
        for hr in range(24):
            y[hrs == hr] = x[hrs == hr] @ coef[hr]
        return pd.Series(y, index=w.index)
    return predict


def summarize(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.dropna(subset=["actual", "p50"]).copy()
    fc["day_ahead"] = fc["h"] // 24 + 1
    fc["ae"] = (fc["p50"] - fc["actual"]).abs()
    g = fc.groupby(["model", "day_ahead"])
    wape = (g["ae"].sum() / g["actual"].sum() * 100).unstack()
    wape.columns = [f"wape_day{c}" for c in wape.columns]
    fc["date"] = fc["ts"].dt.normalize()
    pk = fc.groupby(["model", "origin", "date", "day_ahead"]).agg(a=("actual", "max"), p=("p50", "max")).reset_index()
    pk["err"] = (pk["p"] - pk["a"]) / pk["a"] * 100
    top = pk["a"] >= pk.groupby("date")["a"].first().quantile(0.9)
    d1 = pk["day_ahead"] == 1
    band = fc[(fc["day_ahead"] == 1) & fc["p10"].notna()]
    out = wape.join(pd.DataFrame({
        "day1_peak_abs_err_pct": pk[d1].groupby("model")["err"].apply(lambda s: s.abs().mean()),
        "top10pct_days_peak_abs_err_pct": pk[d1 & top].groupby("model")["err"].apply(lambda s: s.abs().mean()),
        "top10pct_days_peak_bias_pct": pk[d1 & top].groupby("model")["err"].mean(),
        "day1_band80_cover_pct": band.groupby("model").apply(
            lambda b: ((b.actual >= b.p10) & (b.actual <= b.p90)).mean() * 100, include_groups=False),
    }))
    return out.round(2).sort_values("wape_day1").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    b, c = cfg["backtest"], cfg.get("compare", {})
    t0 = time.time()
    df = read_hourly(p(cfg, cfg["data"]["load_csv"]))[["load_mw"]]
    df = df.join(read_hourly(p(cfg, cfg["data"]["weather_csv"]))[WX], how="left")
    df = df.join(read_hourly(p(cfg, cfg["data"]["forecast_csv"])), how="left")
    df["load_mw"] = df["load_mw"].interpolate(limit_area="inside")
    df = df.join(add_features(df[WX], cfg)[FEAT_EXTRA])
    run = Run(cfg, df, None)
    run.lag = c.get("lag_hours", 24)
    ctx_h = c.get("context_hours", b["context_hours"])
    last_ok = min(df["load_mw"].last_valid_index(), df["temp_f_d7"].last_valid_index())

    prepared = []
    for w in b["windows"]:
        ws, we = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        origins = [o for o in pd.date_range(ws, we, freq="D") if o + pd.Timedelta(hours=run.H - 1) <= last_ok]
        if not origins:
            continue
        base = baseline(run, ws)
        resid = df["load_mw"] - base(df[WX].ffill().bfill())
        ctx_l, cov, hl = build_inputs(run, origins, df["load_mw"], ctx_h)
        ctx_r, _, _ = build_inputs(run, origins, resid, ctx_h)
        fut_base = {o: base(run.fx_frame(o).loc[o:]) for o in origins}
        prepared.append((w["name"], origins, ctx_l, ctx_r, cov, hl, fut_base))
        print(f"{w['name']}: {len(origins)} origins")

    rows = []

    def keep(name, window, frames):
        for o, f in frames.items():
            rows.append(pd.DataFrame({"window": window, "model": name, "origin": o, "ts": f.index,
                                      "h": np.arange(len(f)), "actual": df["load_mw"].reindex(f.index).to_numpy(),
                                      "p10": f.get("p10", np.nan), "p50": f["p50"], "p90": f.get("p90", np.nan)}))

    for window, origins, _, _, _, _, fut_base in prepared:
        keep("regression_trend_fx", window, {o: fut_base[o].iloc[:run.H].to_frame("p50") for o in origins})

    for name in c.get("models", ["chronos2"]):
        tm = time.time()
        model = fm.REGISTRY[name](c["kumo_dir"]) if name == "kumo" else fm.REGISTRY[name]()
        for window, origins, ctx_l, ctx_r, cov, hl, fut_base in prepared:
            if model.takes_covariates:
                keep(f"{name}_feat", window, to_frames(run, origins, *model(ctx_l, cov, hl)))
            r = to_frames(run, origins, *model(ctx_r, None, hl))
            keep(f"{name}_resid", window, {o: r[o].add(fut_base[o].reindex(r[o].index), axis=0) for o in origins})
        print(f"{name}: {time.time() - tm:.0f} s")
        del model

    fc = pd.concat(rows, ignore_index=True)
    piv = fc.pivot_table(index=["window", "origin", "ts", "h"], columns="model", values="p50", aggfunc="first")
    act = fc.groupby(["window", "origin", "ts", "h"])["actual"].first()
    feat = [m for m in piv.columns if m.endswith("_feat")]
    if feat:
        for name, s in (("avg_feat", piv[feat].mean(axis=1, skipna=False)),
                        ("avg_feat_reg", (piv[feat].mean(axis=1, skipna=False) + piv["regression_trend_fx"]) / 2)):
            x = s.rename("p50").to_frame().assign(actual=act, model=name).reset_index()
            fc = pd.concat([fc, x], ignore_index=True)
    out = ROOT / "output" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    fc.round(4).to_csv(out / "forecasts.csv", index=False)
    s = summarize(fc)
    s.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(s.to_string(index=False))
    print(f"wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
