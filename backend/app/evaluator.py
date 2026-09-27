from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.entities import Invoice, PredictionEvaluation, RiskPrediction

def evaluate_prediction(session: Session, prediction_id: int) -> dict:
    prediction = session.scalar(
        select(RiskPrediction)
        .where(RiskPrediction.id == prediction_id)
    )
    if not prediction:
        raise ValueError("prediction not found")

    existing = session.scalar(
        select(PredictionEvaluation)
        .where(PredictionEvaluation.prediction_id == prediction_id)
    )
    if existing:
        return _evaluation_dict(existing)

    invoice = session.scalar(
        select(Invoice)
        .options(selectinload(Invoice.payments))
        .where(Invoice.id == prediction.invoice_id)
    )
    if not invoice:
        raise ValueError("invoice not found")

    from decimal import Decimal
    paid_total = sum((p.amount for p in invoice.payments), Decimal("0"))
    if paid_total < invoice.amount:
        raise ValueError("invoice is not fully settled")

    actual_payment_date = max(p.payment_date for p in invoice.payments)
    actual_delay_days = (actual_payment_date - invoice.due_date).days
    actual_late = actual_delay_days > 0
    payment_date_error_days = (actual_payment_date - prediction.expected_payment_date).days
    probability_brier_error = (float(prediction.late_probability) - float(actual_late)) ** 2

    evaluation = PredictionEvaluation(
        prediction_id=prediction_id,
        actual_payment_date=actual_payment_date,
        actual_delay_days=actual_delay_days,
        actual_late=actual_late,
        payment_date_error_days=payment_date_error_days,
        probability_brier_error=probability_brier_error,
        evaluated_at=datetime.now(timezone.utc),
    )
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)
    return _evaluation_dict(evaluation)

def _evaluation_dict(row: PredictionEvaluation) -> dict:
    return {
        "prediction_id": row.prediction_id,
        "actual_payment_date": row.actual_payment_date,
        "actual_delay_days": row.actual_delay_days,
        "actual_late": row.actual_late,
        "payment_date_error_days": row.payment_date_error_days,
        "probability_brier_error": float(row.probability_brier_error),
        "evaluated_at": row.evaluated_at,
    }

def evaluation_summary(session: Session, business_id: str) -> dict:
    rows = session.execute(
        select(PredictionEvaluation)
        .join(RiskPrediction, PredictionEvaluation.prediction_id == RiskPrediction.id)
        .join(Invoice, RiskPrediction.invoice_id == Invoice.id)
        .where(Invoice.business_id == business_id)
    ).scalars().all()

    if not rows:
        return {
            "business_id": business_id,
            "evaluated_predictions": 0,
            "date_mae_days": None,
            "mean_brier_error": None,
            "late_classification_accuracy": None,
        }

    date_mae = sum(abs(row.payment_date_error_days) for row in rows) / len(rows)
    brier = sum(float(row.probability_brier_error) for row in rows) / len(rows)
    correct = sum(
        (float(session.get(RiskPrediction, row.prediction_id).late_probability) >= 0.5) == row.actual_late
        for row in rows
    )
    return {
        "business_id": business_id,
        "evaluated_predictions": len(rows),
        "date_mae_days": round(date_mae, 2),
        "mean_brier_error": round(brier, 6),
        "late_classification_accuracy": round(correct / len(rows), 4),
    }