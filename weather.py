#!/usr/bin/env python3
"""Hourly ACTUAL weather (Open-Meteo historical archive, ERA5 reanalysis) for the site.

Free, no API key. The archive lags real time by about 5 days. Requested in UTC and converted
to naive local hour-beginning time.

Output: data.weather_csv -> ts, temp_f, dewpoint_f, cloud_pct, solar_wm2
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd
import requests

from common import OPEN_METEO_VARS, WX, load_config, p, utc_to_local_hourly

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat, lon, start: date, end: date) -> pd.DataFrame:
    r = requests.get(URL, timeout=120, params={
        "latitude": lat, "longitude": lon, "start_date": start.isoformat(),
        "end_date": end.isoformat(), "hourly": ",".join(OPEN_METEO_VARS),
        "temperature_unit": "fahrenheit", "timezone": "GMT"})
    r.raise_for_status()
    return pd.DataFrame(r.json()["hourly"]).rename(columns=OPEN_METEO_VARS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    cfg = load_config(ap.parse_args().config)
    s = cfg["site"]
    start = date.fromisoformat(cfg["weather"]["archive_start"])
    end = date.today() - timedelta(days=1)
    parts = []
    for y in range(start.year, end.year + 1):  # one request per calendar year
        a, b = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
        print(f"fetching {a} .. {b}")
        parts.append(fetch(s["latitude"], s["longitude"], a, b))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("time")
    h = utc_to_local_hourly(df, "time", WX, s["timezone"])
    h = h.loc[: h["temp_f"].last_valid_index()]  # trailing days arrive as nulls
    out = p(cfg, cfg["data"]["weather_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    h.round(2).to_csv(out)
    print(f"{len(h):,} hours, {h.index.min()} .. {h.index.max()} -> {out}")


if __name__ == "__main__":
    main()
