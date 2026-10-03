"""Week 4 — Model monitoring: prediction logging + input-drift detection.

* ``log_prediction(payload, result)`` appends one JSON line per inference to
  ``logs/predictions.jsonl`` (timestamp, input, prediction, probability).
* ``check_drift()`` compares a simulated live sample against the training
  feature distribution using Population Stability Index (PSI) for numerics
  and z-score of the mean shift; writes ``reports/drift_report.md``.

Usage:
    python src/monitor.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import preprocessing  # noqa: F401,E402  (for unpickling)

TARGET = "Survived"
SEED = 42
LOG_FILE = ROOT / "logs" / "predictions.jsonl"
STATS_FILE = ROOT / "artifacts" / "training_stats.json"

NUMERIC_MONITOR = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "FarePerPerson"]
CATEGORICAL_MONITOR = ["Pclass", "Sex", "Embarked", "Deck", "Title"]


# ---------------------------------------------------------------- logging ---
def log_prediction(payload: dict, result: dict) -> None:
    """Append one inference record as JSONL."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "survived": result.get("survived"),
        "probability": result.get("probability"),
    }
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ----------------------------------------------------------------- drift ---
def _engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> engineered features (same transform the model uses)."""
    prep = joblib.load(ROOT / "artifacts" / "preprocessor.joblib")
    return prep.named_steps["engineer"].transform(df)


def build_training_stats() -> dict:
    """Reference distributions from the 70% train split (deterministic)."""
    df = pd.read_csv(ROOT / "data" / "raw" / "titanic.csv")
    train_df, _ = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[TARGET])
    eng = _engineered(train_df.drop(columns=[TARGET]))
    stats = {"n": len(eng), "numeric": {}, "categorical": {}}
    for col in NUMERIC_MONITOR:
        stats["numeric"][col] = {
            "mean": float(eng[col].mean()), "std": float(eng[col].std()),
            "p10": float(eng[col].quantile(0.10)),
            "p50": float(eng[col].quantile(0.50)),
            "p90": float(eng[col].quantile(0.90)),
        }
    for col in CATEGORICAL_MONITOR:
        stats["categorical"][col] = (
            eng[col].value_counts(normalize=True).to_dict())
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    return stats


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(qs).to_numpy())
    if len(edges) < 3:  # degenerate — fall back to a single bin
        return 0.0
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / e_counts.sum()
    a_pct = a_counts / a_counts.sum()
    # smooth zeros
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def check_drift() -> dict:
    """Compare a simulated live sample to training stats; write a report."""
    if not STATS_FILE.exists():
        build_training_stats()
    stats = json.loads(STATS_FILE.read_text())

    df = pd.read_csv(ROOT / "data" / "raw" / "titanic.csv")
    _, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[TARGET])
    # simulate drift: live passengers skew younger and pay higher fares
    live_df = temp_df.sample(n=200, random_state=SEED).copy()
    live_df["Age"] = (live_df["Age"] - 8).clip(lower=1)
    live_df["Fare"] = live_df["Fare"] * 1.5

    eng_live = _engineered(live_df.drop(columns=[TARGET]))
    df_train = pd.read_csv(ROOT / "data" / "raw" / "titanic.csv")
    train_df, _ = train_test_split(
        df_train, test_size=0.30, random_state=SEED,
        stratify=df_train[TARGET])
    eng_train = _engineered(train_df.drop(columns=[TARGET]))

    findings = []
    for col in NUMERIC_MONITOR:
        psi = _psi(eng_train[col], eng_live[col])
        ref = stats["numeric"][col]
        z = abs(eng_live[col].mean() - ref["mean"]) / max(ref["std"], 1e-9)
        status = ("DRIFT" if psi > 0.25 or z > 1.0
                  else "watch" if psi > 0.10 or z > 0.5 else "ok")
        findings.append({"feature": col, "psi": round(psi, 3),
                         "z_mean_shift": round(float(z), 3), "status": status})
    for col in CATEGORICAL_MONITOR:
        ref = pd.Series(stats["categorical"][col])  # JSON keys are strings
        live = eng_live[col].astype(str).value_counts(normalize=True)
        tv = float((live.subtract(ref, fill_value=0).abs().sum()) / 2)  # total variation
        status = "DRIFT" if tv > 0.25 else "watch" if tv > 0.10 else "ok"
        findings.append({"feature": col, "total_variation": round(tv, 3),
                         "status": status})

    drifted = [f["feature"] for f in findings if f["status"] == "DRIFT"]
    lines = [
        "# Input Drift Report (Week 4)",
        "",
        f"Compared **200** simulated live requests against the training "
        f"reference (n={stats['n']}).",
        "",
        "| Feature | PSI / TV | z (mean shift) | Status |",
        "| --- | --- | --- | --- |",
    ]
    for f in findings:
        metric = f.get("psi", f.get("total_variation"))
        z = f.get("z_mean_shift", "—")
        lines.append(f"| {f['feature']} | {metric} | {z} | {f['status']} |")
    lines += [
        "",
        "## Verdict",
        "",
        ("⚠️ **DRIFT DETECTED** in: " + ", ".join(drifted) + ". "
         "Investigate the upstream data feed before trusting predictions; "
         "consider retraining." if drifted
         else "✅ No significant drift — input distribution matches training."),
        "",
        "_Thresholds: PSI > 0.25 or |z| > 1.0 → DRIFT; PSI > 0.10 or |z| > 0.5 "
        "→ watch._",
    ]
    rep = ROOT / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "drift_report.md").write_text("\n".join(lines) + "\n")
    (rep / "drift_report.json").write_text(json.dumps(findings, indent=2))
    print("Drift findings:")
    for f in findings:
        print(f"  {f['feature']}: {f['status']}")
    print(f"Wrote {rep / 'drift_report.md'}")
    return {"findings": findings, "drifted": drifted}


def demo_logging() -> None:
    """Log a few sample predictions so logs/predictions.jsonl exists."""
    model = joblib.load(ROOT / "artifacts" / "model.joblib")
    samples = [
        {"Pclass": 1, "Name": "Demo, Mrs. A", "Sex": "female", "Age": 30,
         "SibSp": 0, "Parch": 0, "Ticket": "D1", "Fare": 120.0,
         "Cabin": "C1", "Embarked": "S"},
        {"Pclass": 3, "Name": "Demo, Mr. B", "Sex": "male", "Age": 25,
         "SibSp": 0, "Parch": 0, "Ticket": "D2", "Fare": 8.0,
         "Cabin": None, "Embarked": "Q"},
    ]
    frame = pd.DataFrame(samples)
    proba = model.predict_proba(frame)[:, 1]
    labels = model.predict(frame)
    for payload, lbl, p in zip(samples, labels, proba):
        log_prediction(payload, {"survived": bool(lbl),
                                  "probability": round(float(p), 4)})
    print(f"Logged {len(samples)} predictions to {LOG_FILE}")


def main() -> None:
    demo_logging()
    check_drift()


if __name__ == "__main__":
    main()
