"""Shared config loading and small helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
WX = ["temp_f", "dewpoint_f", "cloud_pct", "solar_wm2"]
OPEN_METEO_VARS = {"temperature_2m": "temp_f", "dew_point_2m": "dewpoint_f",
                   "cloud_cover": "cloud_pct", "shortwave_radiation": "solar_wm2"}


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml if present, else config.example.yaml (the synthetic demo)."""
    if path is None:
        path = ROOT / "config.yaml"
        if not path.exists():
            path = ROOT / "config.example.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def p(cfg: dict, rel: str) -> Path:
    """Resolve a config path relative to the repo root."""
    q = Path(rel)
    return q if q.is_absolute() else ROOT / q


def read_hourly(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["ts"], index_col="ts")


def us_holidays(years) -> dict[pd.Timestamp, str]:
    """US federal holidays that move load (observed-date shifts ignored)."""
    out = {}
    for y in years:
        def nth(month, weekday, n):
            d = pd.Timestamp(y, month, 1)
            return d + pd.Timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
        may31 = pd.Timestamp(y, 5, 31)
        out.update({pd.Timestamp(y, 1, 1): "New Year's Day", nth(1, 0, 3): "MLK Day",
                    nth(2, 0, 3): "Presidents Day",
                    may31 - pd.Timedelta(days=may31.weekday()): "Memorial Day",
                    pd.Timestamp(y, 7, 4): "Independence Day", nth(9, 0, 1): "Labor Day",
                    nth(11, 3, 4): "Thanksgiving", pd.Timestamp(y, 12, 25): "Christmas"})
    return out


def utc_to_local_hourly(df: pd.DataFrame, time_col: str, cols: list[str], tz: str) -> pd.DataFrame:
    """UTC timestamps -> naive local hour-beginning, regular hourly index.

    DST: the repeated fall-back hour is averaged; the missing spring-forward hour is
    interpolated. Load data gets the same treatment in prep.py, so the series line up.
    """
    t = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.floor("h")
    h = df.assign(ts=t).groupby("ts")[cols].mean()
    h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="h"))
    h = h.interpolate(limit=3, limit_area="inside")
    h.index.name = "ts"
    return h
