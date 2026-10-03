"""Titanic survival preprocessing pipeline (shared logic, vendored per repo).

Covers the Week-1 data-cleaning spec:
  * Missing values: Age imputed by Pclass/Sex group median,
    Embarked imputed by mode, Cabin -> Deck feature.
  * Outliers: Fare clipped to the 99th percentile learned on the fit data.
  * Feature engineering: Title (from Name), FamilySize, IsAlone,
    FarePerPerson; PassengerId / Name / Ticket / Cabin dropped.
  * Encoding: one-hot for categoricals, standard scaling for numerics.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Survived"
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare",
                    "FamilySize", "IsAlone", "FarePerPerson"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Deck", "Title"]

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Capt": "Rare",
    "Sir": "Rare", "Lady": "Rare", "Don": "Rare", "Dona": "Rare",
    "Countess": "Rare", "Jonkheer": "Rare",
}


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    raw = match.group(1).strip() if match else "Unknown"
    return TITLE_MAP.get(raw, "Rare")


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleaning + feature engineering for the raw Titanic CSV.

    fit() learns: Age medians per (Pclass, Sex) group, the Embarked mode,
    and the Fare 99th-percentile clip bound. transform() applies them.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        medians = X.groupby(["Pclass", "Sex"])["Age"].median()
        self.age_medians_ = medians.to_dict()
        self.age_global_median_ = float(X["Age"].median())
        self.embarked_mode_ = X["Embarked"].mode(dropna=True).iloc[0]
        self.fare_clip_ = float(X["Fare"].quantile(0.99))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        # --- missing values ---
        for (pclass, sex), med in self.age_medians_.items():
            mask = X["Age"].isna() & (X["Pclass"] == pclass) & (X["Sex"] == sex)
            X.loc[mask, "Age"] = med
        X["Age"] = X["Age"].fillna(self.age_global_median_)  # unseen group fallback
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)
        # --- engineered features ---
        X["Title"] = X["Name"].apply(extract_title)
        X["Deck"] = X["Cabin"].fillna("U").astype(str).str[0]
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["FarePerPerson"] = X["Fare"] / X["FamilySize"]
        # --- outliers: clip extreme fares ---
        X["Fare"] = X["Fare"].clip(upper=self.fare_clip_)
        # --- drop identifiers / high-cardinality text ---
        return X.drop(columns=[c for c in DROP_COLS if c in X.columns])


def build_preprocessor() -> Pipeline:
    """Full preprocessing pipeline: engineering -> impute -> encode/scale."""
    engineer = TitanicFeatureEngineer()
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    encoder_step = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([("engineer", engineer), ("encode", encoder_step)])


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL].astype(int)
