"""Week 4 — Inference latency benchmark.

Times 1,000 single-row ``predict_proba`` calls through the production
pipeline and reports p50/p95/p99 latency plus throughput.

Usage:
    python src/benchmark.py
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import preprocessing  # noqa: F401,E402  (for unpickling the model)

N_CALLS = 1000


def main() -> None:
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    df = pd.read_csv(ROOT / "data" / "raw" / "titanic.csv").drop(columns=["Survived"])

    # warm up (first call includes lazy init costs)
    model.predict_proba(df.iloc[[0]])

    latencies = []
    for i in range(N_CALLS):
        row = df.iloc[[i % len(df)]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat = np.array(latencies)
    total_s = lat.sum() / 1000.0
    stats = {
        "calls": N_CALLS,
        "p50_ms": round(float(np.percentile(lat, 50)), 3),
        "p95_ms": round(float(np.percentile(lat, 95)), 3),
        "p99_ms": round(float(np.percentile(lat, 99)), 3),
        "mean_ms": round(float(lat.mean()), 3),
        "throughput_per_s": round(N_CALLS / total_s, 1),
    }
    print("Latency benchmark (single-row predict_proba, ms):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    lines = [
        "# Latency Benchmark (Week 4)",
        "",
        f"**{N_CALLS}** single-row `predict_proba` calls through the full "
        "production pipeline (preprocessing + LogisticRegression), after warm-up.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| p50 latency | {stats['p50_ms']} ms |",
        f"| p95 latency | {stats['p95_ms']} ms |",
        f"| p99 latency | {stats['p99_ms']} ms |",
        f"| mean latency | {stats['mean_ms']} ms |",
        f"| Throughput | {stats['throughput_per_s']} predictions/s |",
        "",
        "## Notes",
        "",
        "- Latency is dominated by the pandas/sklearn preprocessing "
        "overhead per call, not the linear model itself.",
        "- The Week-3 API's LRU response cache makes repeat payloads "
        "~free; batch requests amortise the overhead further.",
        "- For higher throughput: batch incoming requests, or vectorise "
        "predictions across rows instead of one row per call.",
    ]
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "BENCHMARK.md").write_text("\n".join(lines) + "\n")
    print("Wrote reports/BENCHMARK.md")


if __name__ == "__main__":
    main()
