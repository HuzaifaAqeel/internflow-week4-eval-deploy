# DEPLOY.md — Building, running & deploying the containerised service

**Varolline AI Engineer internship · Week 4** — *Optimize latency, implement
model monitoring, and deploy containerized AI service.*

## Prerequisites

- Docker Engine 24+ and Docker Compose v2
- ~500 MB free disk (Python 3.12-slim base + ML stack)

## 1. Build the image

```bash
docker build -t internflow-week4-titanic-api:1.0.0 .
```

What gets baked in:

| Layer | Contents |
| --- | --- |
| `python:3.12-slim` | Base runtime |
| `requirements.txt` | fastapi, uvicorn, scikit-learn, pandas, joblib, cachetools, pydantic |
| `app/` | FastAPI service + preprocessing module |
| `artifacts/` | `model.joblib`, `preprocessor.joblib`, `model_card.json`, `training_stats.json` |

No network access is needed at runtime — the model and data are vendored.

## 2. Run

Single container:

```bash
docker run -d --name titanic-api -p 8000:8000 internflow-week4-titanic-api:1.0.0
curl http://localhost:8000/health
```

Or with Compose (includes a `/health` healthcheck and restart policy):

```bash
docker compose up -d --build
docker compose logs -f
```

## 3. Verify

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":1,"Name":"Test, Mrs. R","Sex":"female","Age":29,
       "SibSp":0,"Parch":0,"Ticket":"PC 1","Fare":100.0,"Embarked":"S"}'
# -> {"survived": true, "probability": 0.95..., "cached": false}
```

## 4. Push & deploy (example: any registry + VM)

```bash
docker tag internflow-week4-titanic-api:1.0.0 <registry>/titanic-api:1.0.0
docker push <registry>/titanic-api:1.0.0
# on the target host:
docker pull <registry>/titanic-api:1.0.0
docker run -d -p 8000:8000 --restart unless-stopped <registry>/titanic-api:1.0.0
```

## Scaling notes

- **Stateless service** — the LRU cache is per-process/in-memory, so the
  container scales horizontally behind any load balancer (nginx, ALB, …).
  Cache-hit rate drops as you add replicas; for a shared cache, front it
  with Redis (swap `cachetools.LRUCache` for a Redis-backed dict).
- **Workers**: uvicorn runs a single worker by default. For CPU-bound sklearn
  inference add workers: `uvicorn app.main:app --workers 4` (or run under
  gunicorn). Each worker holds its own model copy (~10 MB) — cheap.
- **Throughput**: measured ~76 single-row predictions/s per process
  (p95 ≈ 18 ms); batch requests amortise the per-call pandas overhead.
- **Monitoring in production**: mount a volume at `/srv/logs` so
  `logs/predictions.jsonl` survives restarts, and run `python src/monitor.py`
  on a schedule (cron) against recent live traffic for drift alerts.
- **Model updates**: rebuild the image with fresh `artifacts/` and roll the
  deployment (blue/green or rolling) — the `/health` endpoint includes the
  model version for verification.

## Validation status

> Docker is not installed in the build sandbox, so `docker build` could not
> be executed here. The Dockerfile was validated by hand: base image
> `python:3.12-slim` exists, all `COPY` sources exist in the repo, the
> `uvicorn app.main:app` entrypoint was exercised via `TestClient`, and
> `requirements.txt` installs cleanly in a fresh venv.
