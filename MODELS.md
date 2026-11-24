# Foundation models: what `compare_models.py` can run, and their licences

Licences were read from each model card or licence file on **2026-11-23**. They change: check
the current card before you rely on a model in production. "Production" here means using the
forecast in your utility's operations, which several research licences exclude.

| Key | Model | Maker | Weights licence | Takes weather / calendar inputs | Install |
|---|---|---|---|---|---|
| `chronos2` | [Chronos-2](https://huggingface.co/amazon/chronos-2) (120M) | Amazon | Apache-2.0 | Yes | `pip install chronos-forecasting` |
| `timesfm25` | [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (200M) | Google | Apache-2.0 | Not in this adapter (its covariate mode needs JAX) | `pip install "timesfm[torch]"` |
| `t0` | [t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma) | The Forecasting Company | Apache-2.0 | Yes | `pip install "tfc-t0[torch]>=0.6.1"` (Python 3.11+) |
| `tirex2` | [TiRex-2](https://huggingface.co/NX-AI/TiRex-2) | NX-AI | Apache-2.0 | Yes | `pip install tirex-2` (Python 3.11+) |
| `timestral` | [Timestral-209M](https://huggingface.co/mistralai/Timestral-209M) | Mistral AI | **Modified MIT:** no rights for an organization whose consolidated monthly revenue exceeds USD 20M; attribution notice required | Yes | `pip install git+https://github.com/mistralai/timestral.git` |
| `kumo` | [Kumo-Forecast](https://huggingface.co/nvidia/Kumo-Forecast) | NVIDIA | Apache-2.0 | No (past only); 512 h context; point forecast | Not on PyPI: clone [NVIDIA/Kumo-TS](https://github.com/NVIDIA/Kumo-TS), set `compare.kumo_dir` |

## Not wired in, and why

| Model | Reason |
|---|---|
| TimesFM 3.0 (Google) | Its pretrained weights are under `timesfm-non-commercial-license-v1.0`: non-commercial, **non-production** use only (PyPI notice, 2026-11). TimesFM 2.5 weights remain Apache-2.0 |
| Moirai 2.0 (Salesforce) | CC BY-NC 4.0: non-commercial |
| TiRex (v1, NX-AI) | Licence listed as "other"; TiRex-2 is Apache-2.0 and is used instead |
| TimeH2O (H2O.ai) | Licence listed as "other"; not reviewed |

## Practical notes

- **All of these run zero-shot.** Nothing is trained on your data, and nothing leaves your
  machine except the one-time weight download from Hugging Face.
- **Hardware:** all six ran on an 8 GB laptop-class GPU (NVIDIA RTX A1000). Five of them forecast
  a full test window in seconds; Kumo-Forecast calls once per forecast and is slower.
- **Windows:** TiRex-2 compiles one layer with `torch.compile`, which needs Triton, which has no
  Windows build. `foundation_models.py` sets `TORCHDYNAMO_DISABLE=1` so it runs eagerly.
- **Kumo-Forecast's checkpoint loads with pickle semantics** (its own code says so). Load it only
  from NVIDIA's repository.
- **Install into one environment** only what you plan to compare. Each adds its own dependencies;
  none of the six changed an existing package in our environment, but check with
  `pip install --dry-run` first.
