"""Week 3 — RESTful Inference Service (Varolline AI Engineer internship).

FastAPI service exposing the trained Titanic survival model:

  GET  /health          service + model status
  POST /predict         single-passenger prediction (pydantic-validated)
  POST /predict/batch   batch prediction

Features: LRU response caching keyed on the request payload, an
``X-Process-Time`` header on every response, and graceful error handling.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from cachetools import LRUCache
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import preprocessing  # noqa: F401,E402  (needed so joblib can unpickle the pipeline)

MODEL_NAME = "titanic-survival"
MODEL_VERSION = "1.0.0"
CACHE = LRUCache(maxsize=1024)
_model = None


class Passenger(BaseModel):
    Pclass: int = Field(ge=1, le=3, description="Ticket class (1/2/3)")
    Name: str = Field(min_length=1)
    Sex: str = Field(pattern="^(male|female)$")
    Age: Optional[float] = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0, le=20, default=0)
    Parch: int = Field(ge=0, le=20, default=0)
    Ticket: str = Field(min_length=1)
    Fare: float = Field(ge=0)
    Cabin: Optional[str] = None
    Embarked: Optional[str] = Field(default=None, pattern="^(C|Q|S)$")


def load_model():
    global _model
    if _model is None:
        path = ROOT / "artifacts" / "model.joblib"
        if not path.exists():
            raise RuntimeError(
                f"Model artifact not found at {path}. "
                "Run `python scripts/build_artifacts.py` first."
            )
        _model = joblib.load(path)
    return _model


def cache_key(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def predict_rows(rows: List[dict]) -> List[dict]:
    model = load_model()
    frame = pd.DataFrame(rows)
    try:
        proba = model.predict_proba(frame)[:, 1]
        labels = model.predict(frame)
    except Exception as exc:  # graceful: never leak a raw traceback
        raise HTTPException(status_code=500,
                            detail=f"Inference failed: {exc}") from exc
    return [
        {"survived": bool(lbl), "probability": round(float(p), 4)}
        for lbl, p in zip(labels, proba)
    ]


def cached_predict(payload: dict) -> dict:
    key = cache_key(payload)
    if key in CACHE:
        result = dict(CACHE[key])
        result["cached"] = True
        return result
    result = predict_rows([payload])[0]
    result["cached"] = False
    CACHE[key] = {k: v for k, v in result.items() if k != "cached"}
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # fail fast if artifacts are missing/corrupt
    yield
    CACHE.clear()


app = FastAPI(
    title="Titanic Survival Inference API",
    description="Week 3 — Varolline AI Engineer internship",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - start:.4f}s"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": MODEL_VERSION}


@app.post("/predict")
def predict(passenger: Passenger):
    try:
        return cached_predict(passenger.model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Inference failed: {exc}") from exc


@app.post("/predict/batch")
def predict_batch(passengers: List[Passenger]):
    if not passengers:
        raise HTTPException(status_code=422, detail="Batch must not be empty.")
    if len(passengers) > 1000:
        raise HTTPException(status_code=422,
                            detail="Batch too large (max 1000).")
    payloads = [p.model_dump() for p in passengers]
    results, hits = [], 0
    uncached_idx, uncached_rows = [], []
    for i, payload in enumerate(payloads):
        key = cache_key(payload)
        if key in CACHE:
            r = dict(CACHE[key])
            r["cached"] = True
            results.append(r)
            hits += 1
        else:
            uncached_idx.append(i)
            uncached_rows.append(payload)
            results.append(None)
    if uncached_rows:
        fresh = predict_rows(uncached_rows)
        for i, payload, r in zip(uncached_idx, uncached_rows, fresh):
            r["cached"] = False
            CACHE[cache_key(payload)] = {k: v for k, v in r.items()
                                         if k != "cached"}
            results[i] = r
    return {"predictions": results, "cache_hits": hits}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.detail})
