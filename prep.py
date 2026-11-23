#!/usr/bin/env python3
"""Turn a raw meter export into the clean hourly series the backtest expects.

Handles the usual problems:
- Sub-hourly cadence (5, 15 or 30 min), averaged to hourly demand.
- Interval-start or interval-end timestamps.
- Energy per interval (kWh) or demand (kW/MW).
- Several meters summed, e.g. one per substation.
- All-zero reads (outages, lost reads), dropped rather than averaged in.
- DST: the repeated fall-back hour is averaged and the missing spring hour interpolated.

Example (two substation meters, 15-min kW demand, interval-end labels, local time):
    python prep.py --input raw.csv --ts-col dt --value-cols sub1_kw,sub2_kw --units kW --label end

Output: data.load_csv -> ts (local, hour-beginning), load_mw
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import load_config, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--input", required=True, help="raw CSV")
    ap.add_argument("--ts-col", required=True)
    ap.add_argument("--value-cols", required=True, help="comma list; summed")
    ap.add_argument("--units", choices=["kW", "MW", "kWh", "MWh"], default="kW",
                    help="kWh/MWh = energy per interval (converted to demand)")
    ap.add_argument("--label", choices=["start", "end"], default="start",
                    help="whether timestamps mark the start or end of each interval")
    ap.add_argument("--utc", action="store_true", help="timestamps are UTC, not local")
    ap.add_argument("--min-coverage", type=float, default=0.75,
                    help="fraction of an hour's intervals needed to keep the hour")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cols = [c.strip() for c in args.value_cols.split(",")]

    raw = pd.read_csv(args.input, usecols=[args.ts_col] + cols)
    ts = pd.to_datetime(raw[args.ts_col], utc=args.utc)
    if args.utc:
        ts = ts.dt.tz_convert(cfg["site"]["timezone"]).dt.tz_localize(None)
    raw = raw.assign(ts=ts).drop(columns=[args.ts_col]).sort_values("ts")
    step = raw["ts"].diff().dt.total_seconds().mode().iloc[0]
    per_hour = 3600 / step
    print(f"{len(raw):,} rows, cadence {step / 60:.0f} min")

    v = raw[cols].apply(pd.to_numeric, errors="coerce")
    zero = (v == 0).all(axis=1)
    v[zero] = float("nan")
    print(f"dropped {int(zero.sum())} all-zero intervals")
    if args.units in ("kWh", "MWh"):
        v = v * (3600 / step)  # energy per interval -> average demand
    scale = 1000 if args.units in ("kW", "kWh") else 1
    if args.label == "end":
        raw["ts"] = raw["ts"] - pd.Timedelta(seconds=step)

    df = pd.DataFrame({"ts": raw["ts"].dt.floor("h"), "mw": v.sum(axis=1, min_count=len(cols)) / scale})
    g = df.groupby("ts")["mw"].agg(["mean", "count"])
    g.loc[g["count"] < args.min_coverage * per_hour, "mean"] = float("nan")
    h = g["mean"].reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
    n_gap = int(h.isna().sum())
    h = h.interpolate(limit=3, limit_area="inside")
    out = h.rename("load_mw").to_frame()
    out.index.name = "ts"
    path = p(cfg, cfg["data"]["load_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(path)
    print(f"{len(out):,} hours; {n_gap} needed filling, {int(out.load_mw.isna().sum())} still "
          f"missing (gaps > 3 h); peak {out.load_mw.max():.2f} MW at {out.load_mw.idxmax()}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
