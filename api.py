"""
API de détection de fraude explicable.

Lancer :   uvicorn api:app --reload
Docs :     http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import fraud_core as fc

STATE: dict = {}

# Protection optionnelle : si la variable d'environnement API_KEY est définie,
# les routes (hors /health) exigent l'en-tête  X-API-Key: <clé>.
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: Optional[str] = Security(_key_header)) -> None:
    expected = os.environ.get("API_KEY")
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(401, "Clé API absente ou invalide (en-tête X-API-Key)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["art"] = fc.load_artifacts()          # chargé une seule fois au démarrage
    yield
    STATE.clear()


app = FastAPI(
    title="API de détection de fraude e-commerce",
    description="Score de risque d'une transaction, décision d'alerte et explication (SHAP, codes motifs).",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schémas
# --------------------------------------------------------------------------- #
class Transaction(BaseModel):
    transaction_date: datetime = Field(description="Date et heure de la transaction")
    transaction_amount: float = Field(gt=0, description="Montant de la transaction")
    payment_method: str
    product_category: str
    device_used: str
    quantity: int = Field(ge=1)
    customer_age: Optional[int] = Field(default=None, ge=0, le=120, description="Absent ou 0 : valeur typique utilisée")
    account_age_days: int = Field(ge=0, description="Ancienneté du compte en jours")
    addr_match: bool = Field(default=True, description="Adresse de livraison identique à l'adresse de facturation")

    model_config = {"json_schema_extra": {"examples": [{
        "transaction_date": "2024-03-20T03:14:00", "transaction_amount": 830.5,
        "payment_method": "credit card", "product_category": "electronics",
        "device_used": "mobile", "quantity": 2, "customer_age": 29,
        "account_age_days": 9, "addr_match": False}]}}


class Prediction(BaseModel):
    score: float = Field(description="Score de risque entre 0 et 1 (classement, pas une probabilité calibrée)")
    risk_level: str = Field(description="Faible / Moyen / Élevé")
    alert: bool
    threshold: float


class Contribution(BaseModel):
    label: str
    value: str
    contribution: float = Field(description="Effet sur le score (log-odds) : > 0 augmente le risque")


class Explanation(Prediction):
    base_value: float = Field(description="Niveau de risque moyen (log-odds) avant prise en compte de la transaction")
    reason_codes: list[str] = Field(description="Les 3 à 5 facteurs qui poussent le plus vers l'alerte")
    contributions: list[Contribution]


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _frame(items: list[Transaction]) -> pd.DataFrame:
    df = pd.DataFrame([t.model_dump() for t in items])
    bad = fc.unknown_categories(df, STATE["art"])
    if bad:
        allowed = {c: STATE["art"]["cat_levels"][c] for c in bad}
        raise HTTPException(422, detail={"message": "Modalité inconnue", "inconnues": bad, "autorisées": allowed})
    return df


def _thr(threshold: Optional[float]) -> float:
    return STATE["art"]["threshold"] if threshold is None else threshold


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    art = STATE["art"]
    return {"status": "ok", "model": art["model_name"], "threshold": art["threshold"]}


@app.get("/model", dependencies=[Depends(require_key)])
def model_info():
    art = STATE["art"]
    row = art["comparison"].loc[art["model_name"]]
    return {
        "model": art["model_name"],
        "threshold": art["threshold"],
        "analysis_cost_per_alert": art["c_alert"],
        "test_metrics": {k: float(row[k]) for k in ["PR-AUC test", "ROC-AUC test", "Préc@k test", "Rappel@k test"]},
        "categories": art["cat_levels"],
    }


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_key)])
def predict(tx: Transaction, threshold: Optional[float] = Query(None, ge=0, le=1, description="Seuil d'alerte (défaut : seuil à coût minimal)")):
    art = STATE["art"]
    thr = _thr(threshold)
    s = float(fc.score(fc.make_features(_frame([tx]), art), art)[0])
    return Prediction(score=s, risk_level=fc.risk_level(s, thr), alert=s >= thr, threshold=thr)


@app.post("/explain", response_model=Explanation, dependencies=[Depends(require_key)])
def explain(tx: Transaction, threshold: Optional[float] = Query(None, ge=0, le=1),
            n_reasons: int = Query(4, ge=3, le=5)):
    art = STATE["art"]
    thr = _thr(threshold)
    X = fc.make_features(_frame([tx]), art)
    s = float(fc.score(X, art)[0])
    e = fc.explain_rows(X, art, n_reasons)[0]
    return Explanation(
        score=s, risk_level=fc.risk_level(s, thr), alert=s >= thr, threshold=thr,
        base_value=e["base_value"], reason_codes=e["reason_codes"],
        contributions=[Contribution(label=c["label"], value=c["value"], contribution=c["contribution"])
                       for c in e["contributions"]],
    )


@app.post("/predict/batch", response_model=list[Prediction], dependencies=[Depends(require_key)])
def predict_batch(txs: list[Transaction], threshold: Optional[float] = Query(None, ge=0, le=1)):
    if not 0 < len(txs) <= 10_000:
        raise HTTPException(422, "Envoyer entre 1 et 10 000 transactions")
    art = STATE["art"]
    thr = _thr(threshold)
    scores = fc.score(fc.make_features(_frame(txs), art), art)
    return [Prediction(score=float(s), risk_level=fc.risk_level(float(s), thr), alert=bool(s >= thr), threshold=thr)
            for s in scores]