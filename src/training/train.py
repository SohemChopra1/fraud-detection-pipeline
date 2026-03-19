import pandas as pd
import numpy as np
import json
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)



PROCESSED_DIR  = Path("data/processed")
TRAIN_PATH     = PROCESSED_DIR / "train.csv"
TEST_PATH      = PROCESSED_DIR / "test.csv"
MODELS_DIR     = Path("models")
MLFLOW_DIR     = Path("mlruns")


def load_splits():
    """Load train and test splits from data/processed/."""
    print("Loading train/test splits...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=['Class'])
    y_train = train['Class']
    X_test  = test.drop(columns=['Class'])
    y_test  = test['Class']

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_models() -> dict:
    """
    Define the three candidate models with sensible defaults.
    - LogisticRegression: fast baseline
    - RandomForest: strong ensemble, handles nonlinearity
    - XGBoost: typically best on tabular data, handles imbalance natively
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=577,  
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    }


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model and return a metrics dictionary.
    We focus on F1, precision, recall, and ROC-AUC rather than accuracy
    because of the severe class imbalance.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4)
    }

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")
    print(f"F1 Score:  {metrics['f1']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"ROC-AUC:   {metrics['roc_auc']}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> str:
    """Plot and save a confusion matrix, return the file path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()

    path = str(MODELS_DIR / f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_best_model(model, model_name: str, metrics: dict):
    """Save the best model and its metrics to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    meta = {"model_name": model_name, "metrics": metrics}
    with open(MODELS_DIR / "best_model_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest model ({model_name}) saved to {model_path}")


def run():
    """
    Full training run:
    load → train each model → evaluate → track with MLflow → save best
    """
    print("=" * 50)
    print("Starting model training")
    print("=" * 50)


    MODELS_DIR.mkdir(parents=True, exist_ok=True)


    X_train, X_test, y_train, y_test = load_splits()


    mlflow.set_tracking_uri(str(MLFLOW_DIR))
    mlflow.set_experiment("fraud_detection")

    models      = get_models()
    results     = {}
    best_f1     = 0
    best_model  = None
    best_name   = None
    best_metrics= None

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        with mlflow.start_run(run_name=model_name):


            mlflow.log_params(model.get_params())


            model.fit(X_train, y_train)


            metrics = evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics


            mlflow.log_metrics(metrics)


            cm_path = plot_confusion_matrix(model, X_test, y_test, model_name)
            mlflow.log_artifact(cm_path)


            mlflow.sklearn.log_model(model, model_name)


            if metrics['f1'] > best_f1:
                best_f1      = metrics['f1']
                best_model   = model
                best_name    = model_name
                best_metrics = metrics


    save_best_model(best_model, best_name, best_metrics)


    print("\n" + "=" * 50)
    print("Model Comparison Summary")
    print("=" * 50)
    comparison = pd.DataFrame(results).T.sort_values('f1', ascending=False)
    print(comparison.to_string())
    print(f"\n Best model: {best_name} (F1: {best_f1})")


    comparison.to_csv(MODELS_DIR / "model_comparison.csv")
    print("Comparison saved to models/model_comparison.csv")

    print("\n" + "=" * 50)
    print("Training complete")
    print("=" * 50)


if __name__ == "__main__":
    run()