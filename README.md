# Utility Load Forecast Pilot

> **Pilot / working copy.** Code and methods are still being developed, a write-up of results on
> real utility data is in preparation, and this is not production software. Issues and
> suggestions are welcome.

An honest test bench for one question a small utility would ask: **can a free, off-the-shelf AI
forecasting model, run on hardware you already own, forecast your system load better than the
simple methods you'd otherwise use, and where does it fail?**

The model is Amazon's open-weight time-series foundation model
[Chronos-2](https://huggingface.co/amazon/chronos-2) (Apache-2.0). It is used **zero-shot**: no
training on your data. Everything runs locally, and no utility data leaves your machine.

## What makes the test honest

Many published load-forecast results quietly benefit from information a real forecaster would not
have. This bench avoids the two most common cases:

- **Archived weather forecasts, not actual weather.** Each forecast gets the weather forecast that
  was published before it was made (Open-Meteo's
  [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api): lead K for day K).
  Feeding a model the weather that actually happened flatters it. That run is kept only as a
  labelled upper bound.
- **Realistic data delay.** Meter data rarely arrives instantly. The backtest re-runs with the last
  load reading 0, 24 and 48 hours before each forecast, so you can see what stale data costs.

It also compares against baselines a utility would actually use: "same hour last week" and a
per-hour weather regression.

## Quick start (synthetic demo, no utility data needed)

```
python -m venv .venv
.venv\Scripts\activate                # Linux/macOS: source .venv/bin/activate
pip install torch                     # pick the right build: https://pytorch.org/get-started/locally/
pip install -r requirements.txt

python weather.py            # actual weather history for the demo site
python make_demo_data.py     # SYNTHETIC hourly load driven by that weather
python weather_forecast.py   # archived day-ahead weather forecasts
python backtest.py           # all models x all data delays  -> output/
python analysis.py           # tables and figures            -> output/report/
```

The synthetic load is a stand-in to show how the pipeline runs. It has a realistic weather
response, daily and weekly shape, holidays and rooftop-solar dips, but **its results say nothing
about accuracy on a real system**.

A full demo run takes about 10 minutes on an 8 GB GPU. It also runs on a CPU, more slowly.

## Running it on your own data

1. Copy `config.example.yaml` to `config.yaml`. Set your site's latitude, longitude and timezone,
   and set test windows that fall inside your data.
2. Turn your meter export into hourly load:
   ```
   python prep.py --input raw.csv --ts-col dt --value-cols sub1_kw,sub2_kw --units kW --label end
   ```
   It handles 5-, 15- and 30-minute data, interval-start or interval-end labels, energy or demand
   readings, several meters summed together, all-zero outage reads, and DST.
3. Run `weather.py`, `weather_forecast.py`, `backtest.py` and `analysis.py` as above.

`data/`, `output/` and `config.yaml` are git-ignored, so your data stays local.

About a year of hourly load is enough; the model uses up to about 11 months of context. Archived
weather forecasts are available from early 2024 for most weather models, so test windows should
start after that.

## Models compared

| Name | What it is |
|---|---|
| `naive_week` | Same hour in the most recent available week |
| `regression_fx` | Least squares for each hour of the day on temperature (cubic), dew point, cloud, irradiance and weekend, trained on the two years before each test window, driven by the forecast weather |
| `chronos` | Chronos-2 on load history only |
| `chronos_wx_fx` | Chronos-2 with the weather forecast as known-future inputs |
| `chronos_wx_actual` | Chronos-2 with the actual weather. **An upper bound only, not achievable live** |

## What it reports

- **WAPE:** total absolute error divided by total load. It replaces plain MAPE, which explodes on
  the odd near-zero hour in real meter data.
- **MAE (MW):** the typical miss per hour.
- **Skill vs naive:** 1 − WAPE / WAPE of `naive_week`. A scaled error in the spirit of MASE.
- **Pinball loss:** scores the model's 10/50/90% forecast range, not just its median.
- **Range coverage:** how often the actual falls inside the 10–90% range. About 80% is calibrated.
- **Daily-peak error and peak-hour hits** (within ±1 h).
- **Peak watch:** each morning, flag the day if the forecast peak (median or 90th percentile)
  reaches the month's highest so far, then count how many monthly peak days were caught and how
  many alerts that took.

## Known limitations

- **No holiday input.** Holidays are not yet passed to the models.
- **Single-point weather.** One weather point represents the whole service area.
- **One foundation model.** Others (TimesFM, Moirai, TiRex) are not yet wired in.
- **Simple baselines.** A tuned gradient-boosting model would be a stronger one.
- **Forecasts run in calendar days from local midnight.**

## Related work

- [OpenSTEF](https://github.com/OpenSTEF/openstef): an open-source short-term forecasting
  framework used by grid operators, with a Chronos-2 feeder example.
- [Chronos](https://github.com/amazon-science/chronos-forecasting): the model, and
  [fev](https://github.com/autogluon/fev), its evaluation harness.
- Recent load evaluations of time-series foundation models:
  - [Pendyala et al. 2026](https://arxiv.org/abs/2609.06656) (ISO-NE zones)
  - [Kaljevic et al.](https://arxiv.org/abs/2610.07232) (weather-forecast uncertainty)
  - [EnergyBench](https://arxiv.org/abs/2611.07784) (information frozen at a real deadline)
  - [Kaas et al.](https://arxiv.org/abs/2607.01966) (low-voltage feeders)
- Hewamalage, Ackermann & Bergmeir (2023),
  [Forecast evaluation for data scientists](https://doi.org/10.1007/s10618-022-00894-5): the
  pitfalls this bench tries to avoid.

## Data sources and licences

- **This repository:** MIT (see `LICENSE`).
- **Chronos-2:** Apache-2.0, downloaded from Hugging Face on first run.
- **Weather:** [Open-Meteo](https://open-meteo.com/), data licensed **CC BY 4.0**, and
  attribution is required in anything you publish. Open-Meteo's free API is for non-commercial use;
  check [their terms](https://open-meteo.com/en/terms) for yours. Underlying weather models carry
  their own terms. ERA5: Hersbach et al. (2020), *Q. J. R. Meteorol. Soc.* 146:1999-2049.
- **No utility data is included.** The demo data is generated locally.

## References

- Ansari et al. (2025), *Chronos-2: From Univariate to Universal Forecasting*,
  [arXiv:2510.15821](https://arxiv.org/abs/2510.15821).
- Ansari et al. (2024), *Chronos: Learning the Language of Time Series*, TMLR,
  [arXiv:2403.07815](https://arxiv.org/abs/2403.07815).

---
Developed with assistance from Anthropic Claude.
