import pandas as pd
import numpy as np
import json
import pickle

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

RAW_DATA_PATH      = Path("data/raw/creditcard.csv")
PROCESSED_DIR      = Path("data/processed")
EDA_SUMMARY_PATH   = PROCESSED_DIR / "eda_summary.json"
PIPELINE_PATH      = PROCESSED_DIR / "feature_pipeline.pkl"


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw credit card data."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features and clean up columns.
    - Extract Hour from Time
    - Drop raw Time column (Hour captures the useful signal)
    """
    print("Engineering features...")
    df = df.copy()

    # Extract hour of day from Time (seconds since first transaction)
    df['Hour'] = (df['Time'] / 3600).astype(int) % 24

    # Drop raw Time — Hour captures the cyclical signal we care about
    df = df.drop(columns=['Time'])

    print(f"Features after engineering: {df.shape[1]} columns")
    return df


def build_pipeline() -> Pipeline:
    """
    Build a sklearn Pipeline that scales Amount and Hour.
    RobustScaler is used instead of StandardScaler because it's
    resistant to outliers — important given the extreme Amount values.
    """
    pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])
    return pipeline


def apply_pipeline(df: pd.DataFrame, pipeline: Pipeline, fit: bool) -> pd.DataFrame:
    """
    Apply the feature pipeline to the dataframe.
    - fit=True during training (fit + transform)
    - fit=False during inference (transform only)
    Only Amount and Hour are scaled; V1-V28 are already PCA-transformed.
    """
    df = df.copy()
    cols_to_scale = ['Amount', 'Hour']

    if fit:
        df[cols_to_scale] = pipeline.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = pipeline.transform(df[cols_to_scale])

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split into train/test sets.
    Stratify on Class to preserve the fraud ratio in both splits.
    """
    print(f"Splitting data ({int((1-test_size)*100)}/{int(test_size*100)} train/test)...")
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
    print(f"Test fraud rate:  {y_test.mean()*100:.4f}%")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to the training
    set only — never to the test set, as that would leak information.
    SMOTE generates synthetic fraud examples to balance the classes.
    """
    print("Applying SMOTE to training set...")
    print(f"Before SMOTE — Legitimate: {(y_train==0).sum():,} | Fraud: {(y_train==1).sum():,}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE  — Legitimate: {(y_resampled==0).sum():,} | Fraud: {(y_resampled==1).sum():,}")
    return X_resampled, y_resampled


def save_splits(X_train, X_test, y_train, y_test):
    """Save train/test splits to data/processed/ as CSVs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    train['Class'] = y_train.values
    test = X_test.copy()
    test['Class'] = y_test.values

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Saved train.csv ({len(train):,} rows) and test.csv ({len(test):,} rows)")


def save_pipeline(pipeline: Pipeline):
    """Serialize the fitted pipeline so it can be reused at inference time."""
    with open(PIPELINE_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {PIPELINE_PATH}")


def run():
    """
    Full feature engineering run:
    load → engineer → pipeline → split → SMOTE → save
    """
    print("=" * 50)
    print("Starting feature engineering pipeline")
    print("=" * 50)

    # 1. Load
    df = load_data()

    # 2. Engineer features
    df = engineer_features(df)

    # 3. Build and apply pipeline (fit on full data before split,
    #    then we refit on train only during training step)
    pipeline = build_pipeline()

    # 4. Split first — then fit pipeline on train only to prevent leakage
    X_train, X_test, y_train, y_test = split_data(df)

    # 5. Fit pipeline on train, transform both splits
    X_train_scaled = apply_pipeline(
        X_train.copy(), pipeline, fit=True
    )
    X_test_scaled = apply_pipeline(
        X_test.copy(), pipeline, fit=False
    )

    # 6. Apply SMOTE to training set only
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)

    # 7. Save everything
    save_splits(
        pd.DataFrame(X_train_resampled, columns=X_train.columns),
        X_test_scaled,
        pd.Series(y_train_resampled, name='Class'),
        y_test
    )
    save_pipeline(pipeline)

    print("=" * 50)
    print("Feature engineering complete")
    print("=" * 50)


if __name__ == "__main__":
    run()