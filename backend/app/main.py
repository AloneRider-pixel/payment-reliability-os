from datetime import date
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.app.scoring import score_buyer, predict_invoice_risk
from backend.app.models import BuyerHistory, InvoiceInput

app = FastAPI(
    title="Payment Reliability OS",
    version="0.1.0",
    description="Explainable B2B payment-behavior intelligence.",
)

class BuyerScoreResponse(BaseModel):
    score: float
    grade: str
    late_probability: float
    expected_delay_days: float
    reasons: list[str]

class InvoiceRiskResponse(BaseModel):
    late_probability: float
    expected_delay_days: float
    expected_payment_date: date
    cash_at_risk: float
    reasons: list[str]

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/score/buyer", response_model=BuyerScoreResponse)
def score_buyer_endpoint(history: BuyerHistory) -> BuyerScoreResponse:
    result = score_buyer(history)
    return BuyerScoreResponse(**result)

@app.post("/risk/invoice", response_model=InvoiceRiskResponse)
def risk_invoice_endpoint(payload: InvoiceInput) -> InvoiceRiskResponse:
    if payload.amount < 0:
        raise HTTPException(status_code=422, detail="amount must be non-negative")
    result = predict_invoice_risk(payload)
    return InvoiceRiskResponse(**result)
