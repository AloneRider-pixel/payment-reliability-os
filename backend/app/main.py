from datetime import date, datetime, timezone
from decimal import Decimal
import json

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import SessionLocal, init_db
from backend.app.entities import Business, Customer, Invoice, PredictionEvaluation, RiskPrediction
from backend.app.import_service import import_invoices, import_payments
from backend.app.models import BuyerHistory, InvoiceInput
from backend.app.repository import buyer_history, invoice_input
from backend.app.evaluator import evaluate_prediction, evaluation_summary
from backend.app.scoring import predict_invoice_risk, score_buyer

init_db()

app = FastAPI(title="Payment Reliability OS", version="0.3.0", description="Explainable B2B payment-behavior intelligence.")

class BuyerScoreResponse(BaseModel):
    score: float
    grade: str
    late_probability: float
    expected_delay_days: float
    reasons: list[str]
    evidence_count: int
    confidence: str

class InvoiceRiskResponse(BaseModel):
    late_probability: float
    expected_delay_days: float
    expected_payment_date: date
    cash_at_risk: float
    reasons: list[str]

class BusinessCreate(BaseModel):
    id: str
    name: str

def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.post("/businesses")
def create_business(payload: BusinessCreate, db: Session = Depends(get_db)):
    business_id = payload.id.strip()
    business_name = payload.name.strip()
    if not business_id or not business_name:
        raise HTTPException(status_code=422, detail="id and name are required")
    existing = db.get(Business, business_id)
    if existing:
        return {"id": existing.id, "name": existing.name, "created": False}
    business = Business(id=business_id, name=business_name)
    db.add(business)
    db.commit()
    return {"id": business.id, "name": business.name, "created": True}

@app.post("/imports/invoices")
async def upload_invoices(business_id: str = Query(min_length=1), file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="MVP ingestion currently accepts CSV files")
    try:
        result = import_invoices(db, business_id.strip(), await file.read())
        return {"business_id": business_id, **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/imports/payments")
async def upload_payments(business_id: str = Query(min_length=1), file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="MVP ingestion currently accepts CSV files")
    try:
        result = import_payments(db, business_id.strip(), await file.read())
        return {"business_id": business_id, **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/customers")
def list_customers(business_id: str = Query(min_length=1), db: Session = Depends(get_db)):
    customers = db.scalars(select(Customer).where(Customer.business_id == business_id).order_by(Customer.name)).all()
    return [{"id": c.id, "name": c.name, "external_key": c.external_key} for c in customers]

@app.get("/customers/{customer_id}/score", response_model=BuyerScoreResponse)
def customer_score(customer_id: int, db: Session = Depends(get_db)):
    try:
        result = score_buyer(buyer_history(db, customer_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return BuyerScoreResponse(**result)

@app.post("/score/buyer", response_model=BuyerScoreResponse)
def score_buyer_endpoint(history: BuyerHistory):
    return BuyerScoreResponse(**score_buyer(history))

@app.post("/risk/invoice", response_model=InvoiceRiskResponse)
def risk_invoice_endpoint(payload: InvoiceInput):
    return InvoiceRiskResponse(**predict_invoice_risk(payload))

@app.post("/risk/analyze/{invoice_id}", response_model=InvoiceRiskResponse)
def analyze_invoice(invoice_id: int, db: Session = Depends(get_db)):
    try:
        payload = invoice_input(db, invoice_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = predict_invoice_risk(payload)
    prediction = RiskPrediction(invoice_id=invoice_id, predicted_at=datetime.now(timezone.utc), late_probability=result["late_probability"], expected_delay_days=result["expected_delay_days"], expected_payment_date=result["expected_payment_date"], cash_at_risk=Decimal(str(result["cash_at_risk"])), model_version="baseline-v0.1", reasons=json.dumps(result["reasons"]))
    db.add(prediction)
    db.commit()
    return InvoiceRiskResponse(**result)

@app.get("/predictions")
def list_predictions(business_id: str = Query(min_length=1), limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)):
    rows = db.execute(select(RiskPrediction, Invoice, Customer).join(Invoice, RiskPrediction.invoice_id == Invoice.id).join(Customer, Invoice.customer_id == Customer.id).where(Invoice.business_id == business_id).order_by(RiskPrediction.predicted_at.desc()).limit(limit)).all()
    return [{"invoice_id": invoice.id, "invoice_number": invoice.invoice_number, "customer_id": customer.id, "customer": customer.name, "predicted_at": prediction.predicted_at, "late_probability": float(prediction.late_probability), "expected_delay_days": float(prediction.expected_delay_days), "expected_payment_date": prediction.expected_payment_date, "cash_at_risk": float(prediction.cash_at_risk), "model_version": prediction.model_version, "reasons": json.loads(prediction.reasons)} for prediction, invoice, customer in rows]

@app.post("/predictions/{prediction_id}/evaluate")
def evaluate_prediction_endpoint(prediction_id: int, db: Session = Depends(get_db)):
    try:
        return evaluate_prediction(db, prediction_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.get("/evaluations/summary")
def evaluation_summary_endpoint(business_id: str = Query(min_length=1), db: Session = Depends(get_db)):
    return evaluation_summary(db, business_id)

@app.get("/evaluations")
def list_evaluations(
    business_id: str = Query(min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(RiskPrediction, PredictionEvaluation, Invoice, Customer)
        .join(PredictionEvaluation, PredictionEvaluation.prediction_id == RiskPrediction.id)
        .join(Invoice, RiskPrediction.invoice_id == Invoice.id)
        .join(Customer, Invoice.customer_id == Customer.id)
        .where(Invoice.business_id == business_id)
        .order_by(PredictionEvaluation.evaluated_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "prediction_id": prediction.id,
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "customer": customer.name,
            "predicted_payment_date": prediction.expected_payment_date,
            "actual_payment_date": evaluation.actual_payment_date,
            "payment_date_error_days": evaluation.payment_date_error_days,
            "predicted_late_probability": float(prediction.late_probability),
            "actual_late": evaluation.actual_late,
            "actual_delay_days": evaluation.actual_delay_days,
            "model_version": prediction.model_version,
        }
        for prediction, evaluation, invoice, customer in rows
    ]

@app.get("/dashboard")
def dashboard(business_id: str = Query(min_length=1), db: Session = Depends(get_db)):
    invoices = db.scalars(select(Invoice).where(Invoice.business_id == business_id)).all()
    outstanding = Decimal("0")
    for invoice in invoices:
        paid = sum((p.amount for p in invoice.payments), Decimal("0"))
        outstanding += max(invoice.amount - paid, Decimal("0"))
    predictions = db.execute(select(RiskPrediction).join(Invoice, RiskPrediction.invoice_id == Invoice.id).where(Invoice.business_id == business_id).order_by(RiskPrediction.predicted_at.desc())).scalars().all()
    latest_by_invoice = {}
    for prediction in predictions:
        latest_by_invoice.setdefault(prediction.invoice_id, prediction)
    cash_at_risk = sum((p.cash_at_risk for p in latest_by_invoice.values()), Decimal("0"))
    return {"business_id": business_id, "invoice_count": len(invoices), "outstanding_receivables": float(outstanding), "cash_at_risk": float(cash_at_risk), "predicted_invoice_count": len(latest_by_invoice), "high_risk_invoice_count": sum(float(p.late_probability) >= 0.70 for p in latest_by_invoice.values())}