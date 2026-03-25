import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.serving.schemas import TransactionRequest, PredictionResponse, HealthResponse

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR     = Path("models")
PROCESSED_DIR  = Path("data/processed")
MODEL_PATH     = MODELS_DIR / "best_model.pkl"
META_PATH      = MODELS_DIR / "best_model_meta.json"
PIPELINE_PATH  = PROCESSED_DIR / "feature_pipeline.pkl"

# ── Global model state ───────────────────────────────────────────────────────
model_state = {}


def load_artifacts():
    """Load model, pipeline, and metadata into memory at startup."""
    print("Loading model artifacts...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        model_state['model'] = pickle.load(f)

    with open(PIPELINE_PATH, 'rb') as f:
        model_state['pipeline'] = pickle.load(f)

    with open(META_PATH) as f:
        meta = json.load(f)
        model_state['meta'] = meta
        model_state['threshold'] = meta.get('best_threshold', 0.5)

    print(f"Model loaded: {meta['model_name']}")
    print(f"Threshold:    {model_state['threshold']}")
    print(f"F1 Score:     {meta['tuned_metrics']['f1']}")


def get_risk_level(probability: float) -> str:
    """Convert a fraud probability to a human-readable risk level."""
    if probability < 0.4:
        return "LOW"
    elif probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def preprocess_transaction(transaction: TransactionRequest) -> pd.DataFrame:
    """
    Convert a raw transaction request into a feature-engineered DataFrame
    ready for model inference. Mirrors the exact steps in build_features.py.
    """
    # Convert to DataFrame
    data = transaction.model_dump()
    df = pd.DataFrame([data])

    # Engineer Hour feature (mirrors build_features.py)
    df['Hour'] = (df['Time'] / 3600).astype(int) % 24
    df = df.drop(columns=['Time'])

    # Scale Amount and Hour — must pass only these two columns to pipeline
    df[['Amount', 'Hour']] = model_state['pipeline'].transform(df[['Amount', 'Hour']])

    # Reorder columns to exactly match training order
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
    df = df[feature_cols]

    return df


# ── App lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on startup, clean up on shutdown."""
    load_artifacts()
    yield
    model_state.clear()
    print("Model artifacts cleared")


# ── App instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring API",
    version="1.0.0",
    lifespan=lifespan
)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check that the API is running and the model is loaded."""
    if not model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_name=model_state['meta']['model_name'],
        threshold=model_state['threshold'],
        f1_score=model_state['meta']['tuned_metrics']['f1']
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    """
    Score a single transaction and return a fraud prediction.
    - Preprocesses the transaction through the same feature pipeline used in training
    - Applies the tuned classification threshold
    - Returns fraud probability and risk level
    """
    if not model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess
        df = preprocess_transaction(transaction)

        # Get fraud probability
        fraud_probability = float(
            model_state['model'].predict_proba(df)[:, 1][0]
        )

        # Apply tuned threshold
        threshold = model_state['threshold']
        is_fraud = fraud_probability >= threshold

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_probability, 4),
            threshold_used=threshold,
            risk_level=get_risk_level(fraud_probability)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))