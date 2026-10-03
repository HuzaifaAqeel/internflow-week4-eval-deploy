# Week 4 — Performance Evaluation & Deployment

**Varolline AI Engineer internship · Week 4** — *Optimize latency, implement
model monitoring, and deploy containerized AI service.*

Evaluates the trained Titanic model, benchmarks inference latency, adds
prediction logging + input-drift monitoring, and packages the Week-3 FastAPI
service as a Docker container. Fully self-contained: `app/` + `artifacts/`
are vendored in.

## Repository layout

```
week4-eval-deploy/
├── app/                        # FastAPI service (from Week 3) + preprocessing
│   ├── main.py                 # /health, /predict, /predict/batch, LRU cache
│   └── preprocessing.py        # cleaning/feature-engineering transformers
├── artifacts/
│   ├── model.joblib            # winning end-to-end pipeline
│   ├── preprocessor.joblib     # fitted preprocessing pipeline
│   ├── model_card.json
│   └── training_stats.json     # reference distributions for drift checks
├── data/raw/titanic.csv        # vendored raw data
├── src/
│   ├── evaluate.py             # test-set + per-segment metrics -> EVAL_REPORT.md
│   ├── benchmark.py            # latency p50/p95/p99, throughput -> BENCHMARK.md
│   └── monitor.py              # JSONL prediction logger + PSI/z-score drift check
├── reports/
│   ├── EVAL_REPORT.md
│   ├── BENCHMARK.md
│   ├── drift_report.md
│   └── drift_report.json
├── logs/predictions.jsonl      # sample prediction log (git-ignored in practice)
├── Dockerfile                  # python:3.12-slim image
├── docker-compose.yml
├── DEPLOY.md                   # build / run / deploy / scaling
├── requirements.txt
├── LICENSE (MIT)
└── README.md
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/evaluate.py    # test metrics + per-segment analysis
python src/benchmark.py   # latency benchmark over 1000 calls
python src/monitor.py     # log sample predictions + drift check
```

## Results

**Test metrics** (n = 134): accuracy **0.791**, F1 **0.720**,
precision 0.735, recall 0.706, ROC-AUC 0.822.
Per-segment: Pclass=2 is near-perfect (F1=0.96); Pclass=3 is hardest
(F1=0.56). Full breakdown in [`reports/EVAL_REPORT.md`](reports/EVAL_REPORT.md).

**Latency** (1000 single-row calls): p50 **12.0 ms**, p95 **18.0 ms**,
p99 **32.9 ms**, throughput **76.2 pred/s**.
See [`reports/BENCHMARK.md`](reports/BENCHMARK.md).

**Monitoring** (`src/monitor.py`):

- `log_prediction()` appends every inference (timestamp, input, prediction,
  probability) to `logs/predictions.jsonl`.
- `check_drift()` compares live feature distributions to the training
  reference using PSI (numerics) and z-score of the mean shift. On a
  simulated drifted feed it correctly flagged **Age, Fare, FarePerPerson**
  as `DRIFT` while clean features stayed `ok`.

## Deploy

```bash
docker build -t titanic-api:1.0.0 .
docker run -d -p 8000:8000 titanic-api:1.0.0
# or: docker compose up -d --build
```

Full instructions, registry push, and scaling notes in
[`DEPLOY.md`](DEPLOY.md).
