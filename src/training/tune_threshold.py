import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")


def load_best_model():
    with open(MODELS_DIR / "best_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "best_model_meta.json") as f:
        meta = json.load(f)
    return model, meta


def run():
    print("Loading model and test data...")
    model, meta = load_best_model()

    test  = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = test.drop(columns=['Class'])
    y_test = test['Class']


    y_proba = model.predict_proba(X_test)[:, 1]


    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        results.append({
            "threshold": round(thresh, 2),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred), 4)
        })

    df_results = pd.DataFrame(results)


    best_row = df_results.loc[df_results['f1'].idxmax()]
    print(f"\nBest threshold: {best_row['threshold']}")
    print(f"F1:        {best_row['f1']}")
    print(f"Precision: {best_row['precision']}")
    print(f"Recall:    {best_row['recall']}")


    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_results['threshold'], df_results['f1'],
            label='F1', color='steelblue', linewidth=2)
    ax.plot(df_results['threshold'], df_results['precision'],
            label='Precision', color='green', linewidth=2)
    ax.plot(df_results['threshold'], df_results['recall'],
            label='Recall', color='crimson', linewidth=2)
    ax.axvline(best_row['threshold'], color='black',
               linestyle='--', label=f"Best threshold ({best_row['threshold']})")
    ax.set_title('Precision / Recall / F1 vs Classification Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "threshold_tuning.png"), dpi=150)
    plt.show()

    meta['best_threshold'] = float(best_row['threshold'])
    meta['tuned_metrics']  = {
        "f1":        best_row['f1'],
        "precision": best_row['precision'],
        "recall":    best_row['recall']
    }
    with open(MODELS_DIR / "best_model_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest threshold saved to models/best_model_meta.json")


if __name__ == "__main__":
    run()