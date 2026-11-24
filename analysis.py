#!/usr/bin/env python3
"""Tables and figures from the saved backtest forecasts (output/forecasts_lag<N>.csv).

Writes output/report/: tables.xlsx, tables.md and fig1-4 PNGs. Every figure has a matching
table, so values never depend on reading colour.

Errors are WAPE (total absolute miss / total load), not plain MAPE: MAPE explodes on the odd
near-zero hour in real meter data (outages, collection gaps) and then misleads.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT, load_config  # noqa: E402

OUT = ROOT / "output" / "report"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"chronos_wx_fx": "#2a78d6", "chronos": "#eb6834", "regression_fx": "#1baf7a",
         "naive_week": "#eda100", "chronos_wx_actual": "#e87ba4",
         "chronos_feat_fx": "#4a3aa7", "regression_trend_fx": "#0d7d61", "avg_chronos_reg": "#0b0b0b"}
LABEL = {"chronos_wx_fx": "Chronos-2 + forecast weather", "chronos": "Chronos-2, load only",
         "regression_fx": "Weather regression", "naive_week": "Same hour last week",
         "chronos_wx_actual": "Chronos-2 + actual weather (bound)",
         "chronos_feat_fx": "Chronos-2 + weather + degree-hours", "regression_trend_fx":
         "Weather regression + trend", "avg_chronos_reg": "Average: Chronos-2 features + regression"}
ORDER = list(LABEL)[::-1]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def load(lag: int) -> pd.DataFrame | None:
    path = ROOT / "output" / f"forecasts_lag{lag}.csv"
    if not path.exists():
        return None
    f = pd.read_csv(path, parse_dates=["origin", "ts"])
    f["day"] = f["h"] // 24 + 1
    f["ae"] = (f["p50"] - f["actual"]).abs()
    return f


def wape(g) -> float:
    return g["ae"].sum() / g["actual"].sum() * 100


def peaks(f) -> pd.DataFrame:
    d1 = f[f["day"] == 1]
    g = d1.groupby(["window", "model", "origin"])
    out = g.agg(actual_peak=("actual", "max"), p50_peak=("p50", "max"), p90_peak=("p90", "max"))
    out["actual_hr"] = g["actual"].apply(lambda s: int(np.nanargmax(s.to_numpy())))
    out["p50_hr"] = g["p50"].apply(lambda s: int(np.nanargmax(s.to_numpy())))
    return out.reset_index()


def tables(fs: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    f0 = fs[0]
    pk = peaks(f0).assign(err=lambda x: (x.p50_peak - x.actual_peak) / x.actual_peak * 100,
                          hit=lambda x: (x.p50_hr - x.actual_hr).abs() <= 1)
    d1 = f0[f0["day"] == 1]
    t1 = pd.concat([
        d1.groupby(["window", "model"]).apply(wape, include_groups=False).rename("day1_wape_pct"),
        d1.groupby(["window", "model"])["ae"].mean().rename("day1_mae_mw"),
        f0.groupby(["window", "model"]).apply(wape, include_groups=False).rename("week_wape_pct"),
        pk.groupby(["window", "model"]).agg(peak_abs_err_pct=("err", lambda s: s.abs().mean()),
                                            peak_hour_hit_pct=("hit", lambda s: s.mean() * 100),
                                            days=("origin", "count"))], axis=1).reset_index()
    t2 = (f0.groupby(["model", "day"]).apply(wape, include_groups=False).unstack("day")
          .add_prefix("day").reindex([m for m in ORDER if m in set(f0.model)]).reset_index())
    t3 = pd.DataFrame([{"lag_h": lag, "model": m, "day1_wape_pct": wape(g),
                        "day1_mae_mw": g["ae"].mean()}
                       for lag, f in fs.items() for m, g in f[f["day"] == 1].groupby("model")
                       if m != "chronos_wx_actual"]).sort_values(["model", "lag_h"])
    thr = pk[pk.model == "naive_week"].groupby("window")["actual_peak"].quantile(0.9)
    top = pk[pk.actual_peak >= pk.window.map(thr)]
    t4 = top.groupby("model").apply(lambda g: pd.Series({
        "days": len(g), "p50_peak_bias_pct": g.err.mean(),
        "p90_peak_bias_pct": ((g.p90_peak - g.actual_peak) / g.actual_peak * 100).mean(),
        "actual_within_p90_pct": (g.actual_peak <= g.p90_peak).mean() * 100}),
        include_groups=False).reset_index()
    # Monthly-peak watch: alert each morning if the day's forecast peak >= the month's highest
    # actual daily peak so far (test days only). Did it catch the month's peak day, and at what
    # cost in alert days?
    base = pk[pk.model == "naive_week"].assign(month=lambda x: x.origin.dt.to_period("M"))
    rows = []
    for m in [x for x in ORDER if x in set(pk.model)]:
        sub = pk[pk.model == m].set_index("origin")
        for q in ("p50", "p90"):
            if sub[f"{q}_peak"].isna().all():
                continue
            caught = alerts = n = 0
            for _, mb in base.groupby(["window", "month"]):
                mb = mb.sort_values("origin")
                mtd = mb["actual_peak"].cummax().shift(1).fillna(0).to_numpy()
                al = sub[f"{q}_peak"].reindex(mb["origin"]).to_numpy() >= mtd
                caught += bool(al[int(mb["actual_peak"].to_numpy().argmax())])
                alerts += int(al.sum())
                n += 1
            rows.append({"model": m, "signal": q, "months": n, "peaks_caught": caught,
                         "alert_days_per_month": round(alerts / n, 1)})
    return {"T1_accuracy": t1, "T2_by_day": t2, "T3_data_delay": t3, "T4_high_days": t4,
            "T5_peak_watch": pd.DataFrame(rows)}


def _label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    ax.plot([x], [y], "o", ms=5, color=color, mec="white", mew=1.5)


def figures(fs, t, site):
    f0 = fs[0]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for _, r in t["T2_by_day"].iterrows():
        y = r[[c for c in r.index if c.startswith("day")]].to_numpy(dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, "--" if r.model == "chronos_wx_actual" else "-", lw=2, color=COLOR[r.model])
        _label(ax, x[-1], y[-1], LABEL[r.model], COLOR[r.model])
    ax.set_xlim(0.8, len(y) + 2.8)
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("WAPE (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_title(f"Error by forecast day ({site})", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_error_by_day.png", dpi=200)
    plt.close(fig)

    if len(fs) > 1:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for m, g in t["T3_data_delay"].groupby("model"):
            ax.plot(g.lag_h, g.day1_wape_pct, lw=2, color=COLOR[m])
            _label(ax, g.lag_h.iloc[-1], g.day1_wape_pct.iloc[-1], LABEL[m], COLOR[m])
        ax.set_xticks(sorted(fs))
        ax.set_xlim(-3, max(fs) + 40)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Data delay (hours)")
        ax.set_ylabel("Day-1 WAPE (%)")
        ax.grid(axis="y", color=GRID)
        ax.set_title("How much stale data costs", loc="left")
        fig.tight_layout()
        fig.savefig(OUT / "fig2_data_delay.png", dpi=200)
        plt.close(fig)

    pk = peaks(f0)
    g = pk[pk.model == "chronos_wx_fx"]
    if len(g):
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        lo, hi = g.actual_peak.min() * 0.95, g.actual_peak.max() * 1.05
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls=":")
        ax.scatter(g.actual_peak, g.p50_peak, s=22, color=COLOR["chronos_wx_fx"], edgecolor="white")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Actual daily peak (MW)")
        ax.set_ylabel("Forecast daily peak (MW)")
        ax.grid(color=GRID)
        ax.set_title("Chronos-2 + forecast weather: daily peaks", loc="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig3_daily_peaks.png", dpi=200)
        plt.close(fig)

        days = pk[pk.model == "naive_week"].nlargest(2, "actual_peak").sort_values("origin").origin
        fig, axes = plt.subplots(1, len(days), figsize=(7.6, 3.4), sharey=True)
        for ax, o in zip(np.atleast_1d(axes), days):
            d = f0[(f0.origin == o) & (f0.day == 1)]
            a = d[d.model == "chronos_wx_fx"]
            ax.fill_between(a.h, a.p10, a.p90, color=COLOR["chronos_wx_fx"], alpha=0.15, lw=0)
            for m in ["naive_week", "regression_fx", "chronos_wx_fx"]:
                s = d[d.model == m]
                ax.plot(s.h, s.p50, lw=1.8, color=COLOR[m], label=LABEL[m])
            ax.plot(a.h, a.actual, lw=2.4, color=INK, label="Actual")
            ax.set_title(o.strftime("%a %Y-%m-%d"), loc="left")
            ax.set_xlabel("Hour")
            ax.grid(axis="y", color=GRID)
        np.atleast_1d(axes)[0].set_ylabel("Load (MW)")
        np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / "fig4_highest_days.png", dpi=200)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    OUT.mkdir(parents=True, exist_ok=True)
    fs = {lag: f for lag in cfg["backtest"]["lags_hours"] if (f := load(lag)) is not None}
    if 0 not in fs:
        raise SystemExit("output/forecasts_lag0.csv missing: run backtest.py first")
    t = {k: v.round(2) for k, v in tables(fs).items()}
    figures(fs, t, cfg["site"]["name"])
    with pd.ExcelWriter(OUT / "tables.xlsx") as xw:
        for k, v in t.items():
            v.to_excel(xw, sheet_name=k, index=False)
    (OUT / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v.to_markdown(index=False)}"
                                              for k, v in t.items()), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for k, v in t.items():
            print(f"\n== {k}\n{v.to_string(index=False)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
