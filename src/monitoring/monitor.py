import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from datetime import datetime


PROCESSED_DIR  = Path("data/processed")
MODELS_DIR     = Path("models")
MONITORING_DIR = Path("data/monitoring")
PIPELINE_PATH  = PROCESSED_DIR / "feature_pipeline.pkl"
MODEL_PATH     = MODELS_DIR / "best_model.pkl"
META_PATH      = MODELS_DIR / "best_model_meta.json"


#drift detection
def load_reference_data() -> pd.DataFrame:
    """
    Training set = reference distribution. 
    """
    df = pd.read_csv(PROCESSED_DIR / "train.csv")
    # SMOTE syntehtic samples excluded from reference distribution 
    return df


def load_recent_data() -> pd.DataFrame:
    """
    Simulate recent incoming data distribution (test set).
    """
    df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return df


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.05
) -> dict:
    """
    Run a Kolmogorov-Smirnov test on each feature to detect data drift with threshold p-value of 0.05.
    Returns a dict with drift results and summary.
    """
    print("\n" + "=" * 50)
    print("Running Data Drift Detection (KS Test)")
    print("=" * 50)

    feature_cols = [c for c in reference.columns if c != 'Class']
    results = {}
    drifted_features = []

    for col in feature_cols:
        ref_vals  = reference[col].dropna()
        curr_vals = current[col].dropna()

        # subsample reference incase of SMOTE inflation
        if len(ref_vals) > len(curr_vals) * 3:
            ref_vals = ref_vals.sample(n=len(curr_vals), random_state=42)

        ks_stat, p_value = stats.ks_2samp(ref_vals, curr_vals)
        drifted = p_value < threshold

        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value":      round(float(p_value), 4),
            "drift_detected": drifted
        }

        if drifted:
            drifted_features.append(col)

    drift_rate = len(drifted_features) / len(feature_cols) * 100

    print(f"Features tested:   {len(feature_cols)}")
    print(f"Features drifted:  {len(drifted_features)}")
    print(f"Drift rate:        {drift_rate:.1f}%")

    if drifted_features:
        print(f"Drifted features:  {', '.join(drifted_features)}")
    else:
        print("No significant drift detected ✓")

    return {
        "timestamp":        datetime.now().isoformat(),
        "features_tested":  len(feature_cols),
        "features_drifted": len(drifted_features),
        "drift_rate_pct":   round(drift_rate, 2),
        "drifted_features": drifted_features,
        "per_feature":      results
    }



def detect_model_drift(
    model,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = 0.05
) -> dict:
    """
    Detect model drift by comparing the distribution of predicted
    fraud probabilities on reference vs current data.

    If the model starts outputting very different probability distributions
    it likely means the underlying data has changed and retraining is needed.
    """
    print("\n" + "=" * 50)
    print("Running Model Drift Detection")
    print("=" * 50)

    feature_cols = [c for c in reference.columns if c != 'Class']


    ref_sample = reference[feature_cols].sample(
        n=min(len(current), len(reference)), random_state=42
    )
    curr_features = current[feature_cols]

    # predicted prob
    ref_probas  = model.predict_proba(ref_sample)[:, 1]
    curr_probas = model.predict_proba(curr_features)[:, 1]

    # KS test on prob distributions
    ks_stat, p_value = stats.ks_2samp(ref_probas, curr_probas)
    drift_detected = p_value < threshold

    print(f"Reference mean probability:  {ref_probas.mean():.4f}")
    print(f"Current mean probability:    {curr_probas.mean():.4f}")
    print(f"KS Statistic:                {ks_stat:.4f}")
    print(f"P-value:                     {p_value:.4f}")
    print(f"Model drift detected:        {'YES ' if drift_detected else 'NO ✓'}")

    return {
        "timestamp":            datetime.now().isoformat(),
        "ks_statistic":         round(float(ks_stat), 4),
        "p_value":              round(float(p_value), 4),
        "drift_detected":       drift_detected,
        "ref_mean_probability": round(float(ref_probas.mean()), 4),
        "curr_mean_probability":round(float(curr_probas.mean()), 4),
        "ref_fraud_rate":       round(float((ref_probas >= 0.5).mean()), 4),
        "curr_fraud_rate":      round(float((curr_probas >= 0.5).mean()), 4)
    }


#Data visualisation 
def plot_drift_report(
    data_drift: dict,
    model_drift: dict
):
    """
    Generate a drift report with three charts:
    1. Top drift features by KS statistic
    2. Distribution of p-values across all features
    3. Predicted probability distributions 
    """
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # KS statistic chart 
    per_feature = data_drift['per_feature']
    df_features = pd.DataFrame(per_feature).T
    df_features['ks_statistic'] = df_features['ks_statistic'].astype(float)
    top_features = df_features.nlargest(10, 'ks_statistic')

    colors = ['crimson' if row['drift_detected'] else 'steelblue'
              for _, row in top_features.iterrows()]
    axes[0].barh(top_features.index, top_features['ks_statistic'],
                 color=colors, alpha=0.8)
    axes[0].set_title('Top 10 Features by KS Statistic\n(red = drift detected)')
    axes[0].set_xlabel('KS Statistic')
    axes[0].axvline(0.1, color='black', linestyle='--', alpha=0.5, label='0.1 reference')

    # p-value chart
    p_values = [v['p_value'] for v in per_feature.values()]
    axes[1].hist(p_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(0.05, color='crimson', linestyle='--', label='p=0.05 threshold')
    axes[1].set_title('P-value Distribution Across Features')
    axes[1].set_xlabel('P-value')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    #prob distribution chart
    axes[2].set_title('Model Drift — Predicted Probability Distribution')
    axes[2].set_xlabel('Fraud Probability')
    axes[2].set_ylabel('Density')
    axes[2].text(
        0.5, 0.5,
        f"KS Stat: {model_drift['ks_statistic']}\n"
        f"P-value: {model_drift['p_value']}\n"
        f"Drift: {'YES ' if model_drift['drift_detected'] else 'NO ✓'}",
        transform=axes[2].transAxes,
        ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    path = MONITORING_DIR / "drift_report.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"\nDrift report saved to {path}")

#warning 2
def save_report(data_drift: dict, model_drift: dict):
    """Save the full drift report as JSON for downstream alerting."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(),
        "data_drift":   data_drift,
        "model_drift":  model_drift,
        "action_required": bool(
            data_drift['drift_rate_pct'] > 20 or
            model_drift['drift_detected']
        )
    }

    # conversion for JSON 
    def convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    path = MONITORING_DIR / "drift_report.json"
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=convert)

    print(f"Full report saved to {path}")

    if report['action_required']:
        print("\n  ACTION REQUIRED: Significant drift detected.")
        print("   Consider retraining the model on recent data.")
    else:
        print("\n✓  No action required. Model is stable.")

    return report


#main
def run():
    print("=" * 50)
    print("Starting Monitoring Pipeline")
    print("=" * 50)


    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)


    reference = load_reference_data()
    current   = load_recent_data()

    print(f"Reference size: {len(reference):,} rows")
    print(f"Current size:   {len(current):,} rows")


    data_drift  = detect_data_drift(reference, current)
    model_drift = detect_model_drift(model, reference, current)

    # chart
    plot_drift_report(data_drift, model_drift)
    report = save_report(data_drift, model_drift)

    print("\n" + "=" * 50)
    print("Monitoring complete")
    print("=" * 50)

    return report


if __name__ == "__main__":
    run()