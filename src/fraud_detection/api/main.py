"""FastAPI service for real-time fraud scoring.

Run:  uvicorn fraud_detection.api.main:app --reload     (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request

from .. import __version__
from ..config import DATA_DIR, MODELS_DIR
from . import db
from .schemas import BatchRequest, BatchResponse, PredictionResponse, Transaction
from .service import FraudModel

log = logging.getLogger("fraud.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(os.getenv("FRAUD_MODEL_DIR", MODELS_DIR))
    app.state.db_path = Path(os.getenv("FRAUD_DB_PATH", DATA_DIR / "predictions.db"))
    db.init_db(app.state.db_path)
    try:
        app.state.model = FraudModel(models_dir)
        log.info("Loaded model '%s' from %s", app.state.model.name, models_dir)
    except FileNotFoundError:
        app.state.model = None
        log.error("No trained model in %s - run `python -m fraud_detection.train` first.", models_dir)
    yield


app = FastAPI(
    title="Real-Time Credit Card Fraud Detection API",
    version=__version__,
    description="Score card transactions in milliseconds, with optional SHAP explanations.",
    lifespan=lifespan,
)


def _model(request: Request) -> FraudModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, "Model not loaded. Train one with `python -m fraud_detection.train`.")
    return model


def _audit(db_path: Path, preds: list[dict], records: list[dict]) -> None:
    rows = [
        {**{k: p[k] for k in ("transaction_id", "fraud_probability", "risk_level", "threshold", "model", "latency_ms")},
         "is_fraud": int(p["is_fraud"]), "amount": r.get("Amount")}
        for p, r in zip(preds, records)
    ]
    db.log_predictions(db_path, rows)


@app.get("/health", tags=["ops"])
def health(request: Request) -> dict:
    model = request.app.state.model
    return {"status": "ok" if model else "degraded", "model_loaded": model is not None, "version": __version__}


@app.get("/model-info", tags=["ops"])
def model_info(request: Request) -> dict:
    m = _model(request)
    return {
        "serving_model": m.name,
        "threshold": m.threshold,
        "trained_at": m.meta["trained_at"],
        "data_source": m.meta["data_source"],
        "test_metrics": m.meta["metrics"][m.name],
        "features": m.meta["feature_cols"],
    }


@app.get("/stats", tags=["ops"])
def stats(request: Request) -> dict:
    return db.stats(request.app.state.db_path)


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(
    tx: Transaction,  # type: ignore[valid-type]
    request: Request,
    background: BackgroundTasks,
    explain: bool = Query(False, description="Include top SHAP factors (adds a few ms)"),
) -> dict:
    model = _model(request)
    record = tx.model_dump()
    pred = model.predict([record], explain_top_k=5 if explain else 0)[0]
    background.add_task(_audit, request.app.state.db_path, [pred], [record])
    return pred


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(body: BatchRequest, request: Request, background: BackgroundTasks) -> dict:
    model = _model(request)
    records = [t.model_dump() for t in body.transactions]
    preds = model.predict(records)
    background.add_task(_audit, request.app.state.db_path, preds, records)
    return {"count": len(preds), "flagged": sum(p["is_fraud"] for p in preds), "predictions": preds}
