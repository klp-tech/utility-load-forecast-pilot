#!/usr/bin/env python3
"""Make a SYNTHETIC hourly load series so the pipeline runs with no utility data.

Built from the site's real public weather (run weather.py first), so it has realistic weather
response, daily and weekly shape, holidays, rooftop-solar midday dips, slow growth and
autocorrelated noise. It is a stand-in to show how the tools run. It is not a model of any
real utility, and results on it say nothing about accuracy on real data.

Output: data.load_csv -> ts, load_mw
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import load_config, p, read_hourly, us_holidays

# Typical normalised hourly shapes (hour 0..23): evening-peaking mixed residential/commercial.
WEEKDAY = np.array([.62, .58, .56, .55, .57, .65, .80, .92, .95, .95, .96, .97,
                    .97, .96, .96, .98, 1.05, 1.15, 1.18, 1.12, 1.02, .90, .78, .68])
WEEKEND = np.array([.66, .61, .58, .56, .56, .58, .64, .74, .85, .92, .95, .96,
                    .95, .94, .94, .96, 1.02, 1.12, 1.16, 1.10, 1.01, .91, .80, .71])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    d = cfg["demo"]
    wx = read_hourly(p(cfg, cfg["data"]["weather_csv"])).loc[d["start"]:]
    rng = np.random.default_rng(d["seed"])
    idx = wx.index

    t = wx["temp_f"].rolling(4, min_periods=1).mean().to_numpy()  # building thermal lag
    heat = 0.008 * np.clip(55 - t, 0, None)
    cool = 0.0037 * np.clip(t - 65, 0, None) ** 1.5
    weekend = idx.day_of_week >= 5
    hol = set(us_holidays(range(idx.year.min(), idx.year.max() + 1)))
    holiday = np.isin(idx.normalize(), list(hol))
    shape = np.where(weekend | holiday, WEEKEND[idx.hour], WEEKDAY[idx.hour])
    shape = shape / WEEKDAY.mean()
    years = (idx - idx[0]) / pd.Timedelta(days=365.25)
    growth = (1 + d["growth_per_year"]) ** years
    pv = 0.04 * d["mean_mw"] * wx["solar_wm2"].fillna(0).to_numpy() / 1000 * growth

    noise = np.zeros(len(idx))
    e = rng.normal(0, 0.02, len(idx))
    for i in range(1, len(idx)):  # AR(1): errors persist for hours, like real load
        noise[i] = 0.9 * noise[i - 1] + e[i]

    base = d["mean_mw"] / 1.17  # weather terms lift the mean; keep it near mean_mw
    load = base * shape * (1 + heat + cool) * np.where(holiday, 0.92, 1.0) * growth * (1 + noise) - pv
    out = pd.DataFrame({"load_mw": np.round(load, 4)}, index=idx)
    out.index.name = "ts"
    path = p(cfg, cfg["data"]["load_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    print(f"SYNTHETIC load: {len(out):,} h, mean {out.load_mw.mean():.1f} MW, "
          f"peak {out.load_mw.max():.1f} MW at {out.load_mw.idxmax()} -> {path}")


if __name__ == "__main__":
    main()
