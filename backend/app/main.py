from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app import db
from backend.app.evaluator import backtest_business, evaluate_prediction, evaluation_summary
from backend.app.entities import (
    Business,
    Customer,
    Invoice,
    PredictionEvaluation,
    RiskPrediction,
)
from backend.app.import_service import import_invoices, import_payments
from backend.app.models import BuyerHistory, InvoiceInput
from backend.app.repository import buyer_history, invoice_input
from backend.app.ml import (
    active_model_status,
    assess_model_drift,
    latest_model_drift,
    predict_invoice_risk_with_active_model,
    retrain_if_needed,
    rollback_active_model,
    train_business_model,
)
from backend.app.scoring import MODEL_VERSION, predict_invoice_risk, score_buyer

db.init_db()

app = FastAPI(
    title="Payment Reliability OS",
    version="0.6.0",
    description="Explainable B2B payment-behavior intelligence.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScoreFactorResponse(BaseModel):
    name: str
    value: float
    weight: float
    contribution: float
    direction: str


class BuyerScoreResponse(BaseModel):
    score: float
    grade: str
    late_probability: float
    expected_delay_days: float
    reasons: list[str]
    evidence_count: int
    confidence: str
    model_version: str
    features: dict[str, float | int]
    factors: list[ScoreFactorResponse]


class InvoiceRiskResponse(BaseModel):
    late_probability: float
    expected_delay_days: float
    expected_payment_date: date
    cash_at_risk: float
    reasons: list[str]
    model_version: str


class BusinessCreate(BaseModel):
    id: str
    name: str


def get_db():
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
async def upload_invoices(
    business_id: str = Query(min_length=1),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename or file.filename.lower().rsplit(".", 1)[-1] not in {"csv", "xlsx"}:
        raise HTTPException(status_code=415, detail="Upload a CSV or XLSX file")
    try:
        result = import_invoices(
            db, business_id.strip(), await file.read(), file.filename
        )
        return {"business_id": business_id.strip(), **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/imports/payments")
async def upload_payments(
    business_id: str = Query(min_length=1),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename or file.filename.lower().rsplit(".", 1)[-1] not in {"csv", "xlsx"}:
        raise HTTPException(status_code=415, detail="Upload a CSV or XLSX file")
    try:
        result = import_payments(
            db, business_id.strip(), await file.read(), file.filename
        )
        return {"business_id": business_id.strip(), **result}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/customers")
def list_customers(
    business_id: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    customers = db.scalars(
        select(Customer)
        .where(Customer.business_id == business_id)
        .order_by(Customer.name)
    ).all()
    return [
        {
            "id": customer.id,
            "name": customer.name,
            "external_key": customer.external_key,
        }
        for customer in customers
    ]


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

    result = predict_invoice_risk_with_active_model(db, invoice_id)
    prediction = RiskPrediction(
        invoice_id=invoice_id,
        predicted_at=_utc_now_naive(),
        late_probability=result["late_probability"],
        expected_delay_days=result["expected_delay_days"],
        expected_payment_date=result["expected_payment_date"],
        cash_at_risk=Decimal(str(result["cash_at_risk"])),
        model_version=result["model_version"],
        reasons=json.dumps(result["reasons"]),
    )
    db.add(prediction)
    db.commit()
    return InvoiceRiskResponse(**result)


@app.post("/risk/analyze-all")
def analyze_all(
    business_id: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    now = _utc_now_naive()
    invoices = db.scalars(
        select(Invoice)
        .where(Invoice.business_id == business_id)
        .order_by(Invoice.id)
    ).all()

    created = 0
    skipped_settled = 0
    skipped_recent = 0

    for invoice in invoices:
        paid = sum((payment.amount for payment in invoice.payments), Decimal("0"))
        if paid >= invoice.amount:
            skipped_settled += 1
            continue

        latest = db.scalar(
            select(RiskPrediction)
            .where(RiskPrediction.invoice_id == invoice.id)
            .order_by(RiskPrediction.predicted_at.desc())
        )
        if latest and latest.predicted_at >= now - timedelta(hours=24):
            skipped_recent += 1
            continue

        result = predict_invoice_risk_with_active_model(db, invoice.id)
        db.add(
            RiskPrediction(
                invoice_id=invoice.id,
                predicted_at=now,
                late_probability=result["late_probability"],
                expected_delay_days=result["expected_delay_days"],
                expected_payment_date=result["expected_payment_date"],
                cash_at_risk=Decimal(str(result["cash_at_risk"])),
                model_version=result["model_version"],
                reasons=json.dumps(result["reasons"]),
            )
        )
        created += 1

    db.commit()
    return {
        "business_id": business_id,
        "invoices_seen": len(invoices),
        "predictions_created": created,
        "settled_skipped": skipped_settled,
        "recent_predictions_skipped": skipped_recent,
    }


@app.get("/predictions")
def list_predictions(
    business_id: str = Query(min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(RiskPrediction, Invoice, Customer)
        .join(Invoice, RiskPrediction.invoice_id == Invoice.id)
        .join(Customer, Invoice.customer_id == Customer.id)
        .where(Invoice.business_id == business_id)
        .order_by(RiskPrediction.predicted_at.desc())
        .limit(limit)
    ).all()

    return [
        {
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "customer_id": customer.id,
            "customer": customer.name,
            "predicted_at": prediction.predicted_at,
            "late_probability": float(prediction.late_probability),
            "expected_delay_days": float(prediction.expected_delay_days),
            "expected_payment_date": prediction.expected_payment_date,
            "cash_at_risk": float(prediction.cash_at_risk),
            "model_version": prediction.model_version,
            "reasons": json.loads(prediction.reasons),
        }
        for prediction, invoice, customer in rows
    ]


@app.post("/predictions/{prediction_id}/evaluate")
def evaluate_prediction_endpoint(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    try:
        return evaluate_prediction(db, prediction_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/models/train")
def train_model(
    business_id: str = Query(min_length=1),
    min_history: int = Query(default=3, ge=0, le=100),
    test_fraction: float = Query(default=0.30, ge=0.20, le=0.50),
    db: Session = Depends(get_db),
):
    try:
        return train_business_model(
            db,
            business_id,
            min_history=min_history,
            test_fraction=test_fraction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/models/status")
def model_status(business_id: str = Query(min_length=1), db: Session = Depends(get_db)):
    return active_model_status(db, business_id)


@app.get("/models/drift")
def model_drift(
    business_id: str = Query(min_length=1),
    recent_window: int = Query(default=30, ge=1, le=200),
    min_history: int = Query(default=3, ge=0, le=100),
    min_samples: int = Query(default=12, ge=1, le=200),
    db: Session = Depends(get_db),
):
    try:
        return assess_model_drift(
            db,
            business_id,
            recent_window=recent_window,
            min_history=min_history,
            min_samples=min_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/models/drift/latest")
def latest_model_drift_endpoint(
    business_id: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    return latest_model_drift(db, business_id)


@app.post("/models/retrain-if-needed")
def retrain_model_if_needed(
    business_id: str = Query(min_length=1),
    recent_window: int = Query(default=30, ge=1, le=200),
    min_history: int = Query(default=3, ge=0, le=100),
    min_samples: int = Query(default=12, ge=1, le=200),
    db: Session = Depends(get_db),
):
    try:
        return retrain_if_needed(
            db,
            business_id,
            recent_window=recent_window,
            min_history=min_history,
            min_samples=min_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/evaluations/backtest")
def historical_backtest(
    business_id: str = Query(min_length=1),
    min_history: int = Query(default=3, ge=0, le=100),
    db: Session = Depends(get_db),
):
    try:
        return backtest_business(db, business_id, min_history=min_history)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/models/rollback")
def rollback_model(
    business_id: str = Query(min_length=1),
    target_version: str | None = Query(default=None, min_length=1),
    db: Session = Depends(get_db),
):
    try:
        return rollback_active_model(db, business_id, target_version=target_version)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/evaluations/summary")
def evaluation_summary_endpoint(
    business_id: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    return evaluation_summary(db, business_id)


@app.get("/evaluations")
def list_evaluations(
    business_id: str = Query(min_length=1),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(RiskPrediction, PredictionEvaluation, Invoice, Customer)
        .join(
            PredictionEvaluation,
            PredictionEvaluation.prediction_id == RiskPrediction.id,
        )
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


@app.get("/risk/invoices")
def risk_invoices(
    business_id: str = Query(min_length=1),
    threshold: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(RiskPrediction, Invoice, Customer)
        .join(Invoice, RiskPrediction.invoice_id == Invoice.id)
        .join(Customer, Invoice.customer_id == Customer.id)
        .where(Invoice.business_id == business_id)
        .order_by(RiskPrediction.predicted_at.desc())
        .limit(limit * 5)
    ).all()

    latest = {}
    for prediction, invoice, customer in rows:
        if float(prediction.late_probability) < threshold:
            continue
        previous = latest.get(invoice.id)
        if previous is None or prediction.predicted_at > previous[0].predicted_at:
            latest[invoice.id] = (prediction, invoice, customer)

    selected = sorted(
        latest.values(),
        key=lambda row: (
            float(row[0].late_probability),
            float(row[0].cash_at_risk),
        ),
        reverse=True,
    )[:limit]

    return [
        {
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "customer": customer.name,
            "amount": float(invoice.amount),
            "due_date": invoice.due_date,
            "late_probability": float(prediction.late_probability),
            "expected_delay_days": float(prediction.expected_delay_days),
            "expected_payment_date": prediction.expected_payment_date,
            "cash_at_risk": float(prediction.cash_at_risk),
            "risk_band": (
                "HIGH"
                if float(prediction.late_probability) >= 0.70
                else "MEDIUM"
                if float(prediction.late_probability) >= 0.40
                else "LOW"
            ),
            "reasons": json.loads(prediction.reasons),
        }
        for prediction, invoice, customer in selected
    ]


@app.get("/risk/customers")
def risk_customers(
    business_id: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    customers = db.scalars(
        select(Customer)
        .where(Customer.business_id == business_id)
        .order_by(Customer.name)
    ).all()

    result = []
    for customer in customers:
        history = buyer_history(db, customer.id)
        result.append(
            {
                "customer_id": customer.id,
                "customer": customer.name,
                **score_buyer(history),
                "outstanding_amount": history.current_outstanding_amount,
            }
        )

    return sorted(result, key=lambda item: item["score"])


@app.get("/dashboard")
def dashboard(
    business_id: str = Query(min_length=1),
    db: Session = Depends(get_db),
):
    invoices = db.scalars(
        select(Invoice).where(Invoice.business_id == business_id)
    ).all()

    outstanding = Decimal("0")
    for invoice in invoices:
        paid = sum((payment.amount for payment in invoice.payments), Decimal("0"))
        outstanding += max(invoice.amount - paid, Decimal("0"))

    predictions = db.execute(
        select(RiskPrediction)
        .join(Invoice, RiskPrediction.invoice_id == Invoice.id)
        .where(Invoice.business_id == business_id)
        .order_by(RiskPrediction.predicted_at.desc())
    ).scalars().all()

    latest_by_invoice = {}
    for prediction in predictions:
        latest_by_invoice.setdefault(prediction.invoice_id, prediction)

    cash_at_risk = sum(
        (prediction.cash_at_risk for prediction in latest_by_invoice.values()),
        Decimal("0"),
    )

    return {
        "business_id": business_id,
        "invoice_count": len(invoices),
        "outstanding_receivables": float(outstanding),
        "cash_at_risk": float(cash_at_risk),
        "predicted_invoice_count": len(latest_by_invoice),
        "high_risk_invoice_count": sum(
            float(prediction.late_probability) >= 0.70
            for prediction in latest_by_invoice.values()
        ),
    }
