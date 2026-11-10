import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.main import app
from fraud_detection.config import RAW_COLS, TARGET


@pytest.fixture()
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("FRAUD_DB_PATH", str(tmp_path / "pred.db"))
    with TestClient(app) as c:
        yield c


def _payload(row, tid="t1"):
    return {"transaction_id": tid, **{c: float(row[c]) for c in RAW_COLS}}


def test_training_beats_baseline(model_dir):
    import json

    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["metrics"][meta["serving_model"]]["pr_auc"] > 0.5  # base rate is only ~2%
    for name in ("random_forest", "xgboost", "isolation_forest"):
        assert (model_dir / f"{name}.joblib").exists()


def test_health_and_info(client):
    assert client.get("/health").json()["model_loaded"] is True
    assert "serving_model" in client.get("/model-info").json()


def test_predict_flags_fraud_more_than_legit(client, small_df):
    fraud = small_df[small_df[TARGET] == 1].head(15)
    legit = small_df[small_df[TARGET] == 0].head(15)
    pf = [client.post("/predict", json=_payload(r)).json()["fraud_probability"] for _, r in fraud.iterrows()]
    pl = [client.post("/predict", json=_payload(r)).json()["fraud_probability"] for _, r in legit.iterrows()]
    assert sum(pf) / len(pf) > sum(pl) / len(pl)


def test_predict_with_explanations(client, small_df):
    body = _payload(small_df[small_df[TARGET] == 1].iloc[0])
    r = client.post("/predict?explain=true", json=body).json()
    assert len(r["top_factors"]) == 5
    assert {"feature", "shap_value", "direction"} <= set(r["top_factors"][0])


def test_batch_and_stats(client, small_df):
    rows = [_payload(r, f"b{i}") for i, (_, r) in enumerate(small_df.head(10).iterrows())]
    r = client.post("/predict/batch", json={"transactions": rows}).json()
    assert r["count"] == 10
    assert client.get("/stats").json()["total_scored"] >= 10


def test_validation_errors(client, small_df):
    bad = _payload(small_df.iloc[0])
    bad["Amount"] = -5
    assert client.post("/predict", json=bad).status_code == 422
    del bad["V1"]
    assert client.post("/predict", json=bad).status_code == 422


def test_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("FRAUD_DB_PATH", str(tmp_path / "p.db"))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.get("/model-info").status_code == 503
