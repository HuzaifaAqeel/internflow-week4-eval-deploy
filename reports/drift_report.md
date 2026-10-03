# Input Drift Report (Week 4)

Compared **200** simulated live requests against the training reference (n=623).

| Feature | PSI / TV | z (mean shift) | Status |
| --- | --- | --- | --- |
| Age | 0.455 | 0.594 | DRIFT |
| Fare | 1.898 | 0.377 | DRIFT |
| SibSp | 0.03 | 0.22 | ok |
| Parch | 0.003 | 0.02 | ok |
| FamilySize | 0.022 | 0.131 | ok |
| FarePerPerson | 1.245 | 0.189 | DRIFT |
| Pclass | 0.014 | — | ok |
| Sex | 0.032 | — | ok |
| Embarked | 0.034 | — | ok |
| Deck | 0.055 | — | ok |
| Title | 0.054 | — | ok |

## Verdict

⚠️ **DRIFT DETECTED** in: Age, Fare, FarePerPerson. Investigate the upstream data feed before trusting predictions; consider retraining.

_Thresholds: PSI > 0.25 or |z| > 1.0 → DRIFT; PSI > 0.10 or |z| > 0.5 → watch._
