"""One adapter per open-weight time-series foundation model, all with the same interface.

    model(ctx, cov, horizon) -> (p10, p50, p90), each [B, horizon]

ctx  [B, T]           load history (or a residual), float32, ending at the data cut
cov  [B, F, T + h]    known inputs over history AND horizon (weather, degree-hours, calendar),
                      or None. Only models with `takes_covariates = True` use it.

Every package is optional: install only the models you want (see MODELS.md for licences).
Licences differ. Check MODELS.md before any production use.
"""
from __future__ import annotations

import os
import sys

# TiRex-2 torch.compile()s a layer; on Windows there is no Triton, so run eagerly.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

Q = (0.1, 0.5, 0.9)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Chronos2:
    """Amazon Chronos-2 (Apache-2.0). pip install chronos-forecasting"""
    name, takes_covariates = "chronos2", True

    def __init__(self, ckpt="amazon/chronos-2"):
        from chronos import Chronos2Pipeline
        self.pipe = Chronos2Pipeline.from_pretrained(ckpt, device_map=DEVICE)

    def __call__(self, ctx, cov, h):
        T = ctx.shape[1]
        start = pd.Timestamp("2000-01-01")
        hist, fut = [], []
        for i in range(len(ctx)):
            idx = pd.date_range(start, periods=T + h, freq="h")
            d = pd.DataFrame({"y": ctx[i]}, index=idx[:T])
            if cov is not None:
                for j in range(cov.shape[1]):
                    d[f"c{j}"] = cov[i, j, :T]
                f = pd.DataFrame({f"c{j}": cov[i, j, T:] for j in range(cov.shape[1])}, index=idx[T:])
                fut.append(f.assign(item_id=str(i)))
            hist.append(d.assign(item_id=str(i)))
        pred = self.pipe.predict_df(
            pd.concat(hist).rename_axis("ts").reset_index(),
            future_df=pd.concat(fut).rename_axis("ts").reset_index() if fut else None,
            id_column="item_id", timestamp_column="ts", target="y", prediction_length=h,
            quantile_levels=list(Q), batch_size=50)
        g = dict(tuple(pred.groupby("item_id")))
        out = [np.stack([g[str(i)][str(q)].to_numpy() for i in range(len(ctx))]) for q in Q]
        return tuple(out)


class TimesFM25:
    """Google TimesFM 2.5 (Apache-2.0 weights). pip install "timesfm[torch]".
    Not TimesFM 3.0: its weights are non-commercial / non-production. No covariates here
    (TimesFM's covariate mode needs JAX)."""
    name, takes_covariates = "timesfm25", False

    def __init__(self, ckpt="google/timesfm-2.5-200m-pytorch", context=8192):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(ckpt)
        self.m.compile(timesfm.ForecastConfig(
            max_context=context, max_horizon=1024, normalize_inputs=True, per_core_batch_size=16,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=False, fix_quantile_crossing=True))

    def __call__(self, ctx, cov, h):
        _, qs = self.m.forecast(horizon=h, inputs=[c for c in ctx])
        return qs[:, :h, 1], qs[:, :h, 5], qs[:, :h, 9]  # deciles: index 1=p10, 5=p50, 9=p90


class T0Gamma:
    """The Forecasting Company t0-gamma (Apache-2.0). pip install "tfc-t0[torch]>=0.6.1" """
    name, takes_covariates = "t0", True

    def __init__(self, ckpt="theforecastingcompany/t0-gamma"):
        from t0 import T0Forecaster
        self.m = T0Forecaster.from_pretrained(ckpt).eval().to(DEVICE)

    def __call__(self, ctx, cov, h, bs=8):
        outs = []
        for i in range(0, len(ctx), bs):
            with torch.no_grad():
                f = self.m.predict(ctx[i:i + bs], horizon=h, quantile_levels=Q,
                                   future_covariates=None if cov is None else cov[i:i + bs])
            q = f.quantiles
            q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
            outs.append(q.reshape(q.shape[0], h, len(Q)))
        q = np.concatenate(outs)
        return q[..., 0], q[..., 1], q[..., 2]


class TiRex2:
    """NX-AI TiRex-2 (Apache-2.0). pip install tirex-2"""
    name, takes_covariates = "tirex2", True

    def __init__(self, ckpt="NX-AI/TiRex-2"):
        import tirex2
        self.t = tirex2
        self.m = tirex2.load_model(ckpt, device=DEVICE)

    def __call__(self, ctx, cov, h, bs=16):
        p = ([], [], [])
        for i in range(0, len(ctx), bs):
            ts = [self.t.TimeseriesType(
                target=torch.from_numpy(ctx[j]).unsqueeze(0), past_covariates=None,
                future_covariates=None if cov is None else torch.from_numpy(cov[j]))
                for j in range(i, min(i + bs, len(ctx)))]
            for a in self.m.forecast(ts, prediction_length=h, output_type="numpy"):
                a = np.asarray(a)  # (1, 9 deciles, h)
                p[0].append(a[0, 0]); p[1].append(a[0, 4]); p[2].append(a[0, 8])
        return tuple(np.stack(x) for x in p)


class Timestral:
    """Mistral Timestral-209M. Modified MIT: no rights for organizations above USD 20M monthly
    revenue. pip install git+https://github.com/mistralai/timestral.git"""
    name, takes_covariates = "timestral", True

    def __init__(self, ckpt="mistralai/Timestral-209M"):
        from timestral import TimestralForecaster
        self.m = TimestralForecaster.from_pretrained(ckpt, device=DEVICE)

    def __call__(self, ctx, cov, h, bs=8):
        p = {q: [] for q in Q}
        for i in range(0, len(ctx), bs):
            f = self.m.forecast(ctx[i:i + bs], prediction_length=h,
                                known_covariates=None if cov is None else cov[i:i + bs],
                                clip_nonnegative=False)
            for q in Q:
                a = f.quantile(q)
                a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
                p[q].append(a.reshape(a.shape[0], h))
        return tuple(np.concatenate(p[q]) for q in Q)


class KumoForecast:
    """NVIDIA Kumo-Forecast (Apache-2.0). Not on PyPI: clone github.com/NVIDIA/Kumo-TS and set
    compare.kumo_dir to its Kumo-Forecast folder. MOMENT backbone: 512 h context, point
    forecast, no known-future covariates. Its checkpoint loads with pickle semantics."""
    name, takes_covariates = "kumo", False

    def __init__(self, kumo_dir):
        if not kumo_dir:
            raise SystemExit("set compare.kumo_dir to a clone of NVIDIA/Kumo-TS/Kumo-Forecast")
        sys.path.append(str(kumo_dir))
        from sdk.forecasting import ForecastingConfig, perform_forecasting
        self.cfg_cls, self.run = ForecastingConfig, perform_forecasting

    def __call__(self, ctx, cov, h):
        pts = []
        for row in ctx:
            d = pd.DataFrame({"timestamp": pd.date_range("2000-01-01", periods=512, freq="h"),
                              "target": row[-512:]})
            cfg = self.cfg_cls(seq_len=512, forecast_horizon=h, model_horizon=72, device=DEVICE,
                               num_workers=0)
            pts.append(self.run(d, config=cfg)["target_forecast"].to_numpy(dtype=float)[:h])
        p = np.stack(pts)
        nan = np.full_like(p, np.nan)  # point forecast: no range
        return nan, p, nan


REGISTRY = {"chronos2": Chronos2, "timesfm25": TimesFM25, "t0": T0Gamma, "tirex2": TiRex2,
            "timestral": Timestral, "kumo": KumoForecast}
