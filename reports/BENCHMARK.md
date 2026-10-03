# Latency Benchmark (Week 4)

**1000** single-row `predict_proba` calls through the full production pipeline (preprocessing + LogisticRegression), after warm-up.

| Metric | Value |
| --- | --- |
| p50 latency | 12.031 ms |
| p95 latency | 18.037 ms |
| p99 latency | 32.939 ms |
| mean latency | 13.127 ms |
| Throughput | 76.2 predictions/s |

## Notes

- Latency is dominated by the pandas/sklearn preprocessing overhead per call, not the linear model itself.
- The Week-3 API's LRU response cache makes repeat payloads ~free; batch requests amortise the overhead further.
- For higher throughput: batch incoming requests, or vectorise predictions across rows instead of one row per call.
