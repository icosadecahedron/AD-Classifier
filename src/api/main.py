"""
FastAPI serving layer for the trained AD classifier.

Run locally with:
    uvicorn src.api.main:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"gene_expression": {"GENE_00001": 8.2, "APOE": 9.1, ...}}'

Requires: fastapi, uvicorn, shap (see requirements.txt). This module is not
runnable in the current offline dev sandbox (no network to install those
packages) but is the real, intended serving code -- install requirements.txt
in any normal (networked) environment or the Docker image built from this
repo to run it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    import shap
    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    _HAS_SHAP = False

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "selected_genes.json"

app = FastAPI(
    title="Alzheimer's Gene Expression Classifier API",
    description="Predicts AD vs. control status from gene expression features, with SHAP explanations.",
    version="1.0.0",
)

_model = None
_feature_columns = None
_explainer = None


class PredictionRequest(BaseModel):
    gene_expression: Dict[str, float] = Field(
        ..., description="Mapping of gene name -> expression value. Missing genes default to 0."
    )


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_ad: float
    top_contributing_genes: Dict[str, float] | None = None


@app.on_event("startup")
def load_model():
    global _model, _feature_columns, _explainer
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.pipeline` first."
        )
    _model = joblib.load(MODEL_PATH)
    _feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    if _HAS_SHAP:
        try:
            _explainer = shap.Explainer(_model)
        except Exception:
            _explainer = None  # some model/SHAP combos need a background dataset; degrade gracefully


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    row = {gene: request.gene_expression.get(gene, 0.0) for gene in _feature_columns}
    X = pd.DataFrame([row])[_feature_columns]

    pred = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0][1] if hasattr(_model, "predict_proba") else float(pred)

    top_genes = None
    if _explainer is not None:
        try:
            shap_values = _explainer(X)
            contributions = dict(zip(_feature_columns, shap_values.values[0]))
            top_genes = dict(
                sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            )
        except Exception:
            top_genes = None

    return PredictionResponse(
        prediction=pred,
        prediction_label="Alzheimer's" if pred == 1 else "Control",
        probability_ad=float(proba),
        top_contributing_genes=top_genes,
    )


@app.get("/features")
def list_expected_features():
    """Returns the exact gene list the model expects as input."""
    if FEATURE_LIST_PATH.exists():
        with open(FEATURE_LIST_PATH) as f:
            return {"expected_genes": json.load(f)}
    return {"expected_genes": _feature_columns or []}
