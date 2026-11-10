"""Pydantic request/response models for the fraud API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..config import V_COLS

_EXAMPLE = {"transaction_id": "txn-0001", "Time": 52_000.0, "Amount": 129.99, **{c: 0.0 for c in V_COLS}}


class _TransactionBase(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE})
    transaction_id: str | None = Field(None, description="Your own identifier, echoed back in the response")


# Dynamically declare Time, Amount, V1..V28 so the schema stays in sync with config.V_COLS.
Transaction = create_model(  # type: ignore[call-overload]
    "Transaction",
    __base__=_TransactionBase,
    Time=(float, Field(..., ge=0, description="Seconds elapsed since the first transaction in the dataset")),
    Amount=(float, Field(..., ge=0, description="Transaction amount")),
    **{c: (float, Field(..., description="Anonymised PCA component")) for c in V_COLS},
)


class Factor(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    transaction_id: str | None
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    model: str
    latency_ms: float
    top_factors: list[Factor] | None = None


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)  # type: ignore[valid-type]


class BatchResponse(BaseModel):
    count: int
    flagged: int
    predictions: list[PredictionResponse]
