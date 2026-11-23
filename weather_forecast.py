#!/usr/bin/env python3
"""ARCHIVED day-ahead weather forecasts for the site (Open-Meteo Previous Runs API).

For every hour, `<var>_previous_dayK` is the value forecast by the model run about K x 24 h
earlier. The backtest uses lead K for day K of its 7-day horizon, so each forecast only sees
weather information published before its origin. This is what a live system would have had.
Feeding models the weather that actually happened (reanalysis) instead is a common, optimistic
mistake.

Output: data.forecast_csv -> ts, then <var>_d<K> for K = 1..7
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd
import requests

from common import OPEN_METEO_VARS, load_config, p, utc_to_local_hourly

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LEADS = range(1, 8)


def fetch(lat, lon, start: date, end: date) -> pd.DataFrame:
    hourly = [f"{v}_previous_day{k}" for v in OPEN_METEO_VARS for k in LEADS]
    r = requests.get(URL, timeout=180, params={
        "latitude": lat, "longitude": lon, "hourly": ",".join(hourly),
        "temperature_unit": "fahrenheit", "timezone": "GMT",
        "start_date": start.isoformat(), "end_date": end.isoformat()})
    r.raise_for_status()
    ren = {f"{v}_previous_day{k}": f"{s}_d{k}" for v, s in OPEN_METEO_VARS.items() for k in LEADS}
    return pd.DataFrame(r.json()["hourly"]).rename(columns=ren)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    s = cfg["site"]
    start, end = date.fromisoformat(cfg["weather"]["forecast_start"]), date.today() - timedelta(days=1)
    parts, a = [], start
    while a <= end:  # quarter-sized requests: 28 series x ~2,200 h
        b = min(a + timedelta(days=91), end)
        print(f"fetching {a} .. {b}")
        parts.append(fetch(s["latitude"], s["longitude"], a, b))
        a = b + timedelta(days=1)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("time")
    cols = [c for c in df.columns if c != "time"]
    h = utc_to_local_hourly(df, "time", cols, s["timezone"])
    out = p(cfg, cfg["data"]["forecast_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    h.round(2).to_csv(out)
    print(f"{len(h):,} hours, {h.index.min()} .. {h.index.max()} -> {out}")


if __name__ == "__main__":
    main()
