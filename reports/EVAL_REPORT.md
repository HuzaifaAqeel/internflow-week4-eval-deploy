# Model Evaluation Report (Week 4)

Test set: **134** passengers (stratified 15% hold-out, seed=42).

## Overall test metrics

| Metric | Value |
| --- | --- |
| accuracy | 0.791 |
| f1 | 0.72 |
| precision | 0.7347 |
| recall | 0.7059 |
| roc_auc | 0.8223 |

## Per-segment analysis

| Segment | n | Accuracy | F1 | Precision | Recall | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Pclass=1 | 35 | 0.7143 | 0.7222 | 0.8125 | 0.65 | 0.7233 |
| Pclass=2 | 24 | 0.9583 | 0.96 | 0.9231 | 1.0 | 0.9583 |
| Pclass=3 | 75 | 0.7733 | 0.5641 | 0.55 | 0.5789 | 0.7265 |
| Sex=female | 48 | 0.7708 | 0.8533 | 0.7442 | 1.0 | 0.8535 |
| Sex=male | 86 | 0.8023 | 0.32 | 0.6667 | 0.2105 | 0.6638 |

## Observations

- Female passengers are predicted well (F1=0.85) — sex is the strongest
  signal in the data. Male F1 is low (0.32) because few males survived, so the
  positive class is rare in that segment; accuracy there is still 0.80.
- Performance is weakest for Pclass=3 (F1=0.56), the largest and hardest
  segment; survivors in that group are missed more often (recall drops).
- Accuracy is fairly stable across segments, but F1 varies — worth
  monitoring per-segment recall in production (see `src/monitor.py`).

## Conclusion

The model generalises from train to the held-out test set (test F1=0.72, accuracy=0.791) with no severe segment collapse. Cleared for containerised deployment.
