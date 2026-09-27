from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.entities import Invoice, PredictionEvaluation, RiskPrediction
from backend.app.models import BuyerHistory, InvoiceInput
from backend.app.scoring import MODEL_VERSION, predict_invoice_risk


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

    paid_total = sum((payment.amount for payment in invoice.payments), Decimal("0"))
    if paid_total < invoice.amount:
        raise ValueError("invoice is not fully settled")

    actual_payment_date = max(payment.payment_date for payment in invoice.payments)
    actual_delay_days = (actual_payment_date - invoice.due_date).days
    actual_late = actual_delay_days > 0
    payment_date_error_days = (
        actual_payment_date - prediction.expected_payment_date
    ).days
    probability_brier_error = (
        float(prediction.late_probability) - float(actual_late)
    ) ** 2

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
        (
            float(session.get(RiskPrediction, row.prediction_id).late_probability)
            >= 0.5
        )
        == row.actual_late
        for row in rows
    )
    return {
        "business_id": business_id,
        "evaluated_predictions": len(rows),
        "date_mae_days": round(date_mae, 2),
        "mean_brier_error": round(brier, 6),
        "late_classification_accuracy": round(correct / len(rows), 4),
    }


def _history_before_invoice(
    customer_invoices: list[Invoice],
    target: Invoice,
) -> BuyerHistory:
    cutoff = target.invoice_date
    prior = [
        invoice
        for invoice in customer_invoices
        if (
            invoice.invoice_date < cutoff
            or (invoice.invoice_date == cutoff and invoice.id < target.id)
        )
    ]

    settled = []
    outstanding = 0.0

    for invoice in prior:
        paid_by_cutoff = [
            payment
            for payment in invoice.payments
            if payment.payment_date <= cutoff
        ]
        paid_total = sum(float(payment.amount) for payment in paid_by_cutoff)
        outstanding += max(float(invoice.amount) - paid_total, 0.0)

        if paid_total + 0.005 < float(invoice.amount):
            continue

        final_payment = max(payment.payment_date for payment in paid_by_cutoff)
        settled.append(
            {
                "amount": float(invoice.amount),
                "delay": float((final_payment - invoice.due_date).days),
            }
        )

    target_paid_by_cutoff = sum(
        float(payment.amount)
        for payment in target.payments
        if payment.payment_date <= cutoff
    )
    outstanding += max(float(target.amount) - target_paid_by_cutoff, 0.0)

    delays = [item["delay"] for item in settled]
    settled_amounts = [item["amount"] for item in settled]

    return BuyerHistory(
        payment_delays_days=delays,
        invoice_count=len(delays),
        late_invoice_count=sum(delay > 0 for delay in delays),
        recent_delays_days=delays[-5:],
        average_invoice_amount=(
            sum(settled_amounts) / len(settled_amounts)
            if settled_amounts
            else 0.0
        ),
        current_outstanding_amount=outstanding,
    )


def _calibration_bins(
    predictions: list[float],
    actuals: list[bool],
    bin_count: int = 5,
) -> list[dict]:
    bins = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = [
            (prediction, actual)
            for prediction, actual in zip(predictions, actuals)
            if (
                lower <= prediction < upper
                or (index == bin_count - 1 and lower <= prediction <= upper)
            )
        ]
        if not selected:
            continue

        mean_predicted = sum(item[0] for item in selected) / len(selected)
        observed_rate = sum(item[1] for item in selected) / len(selected)
        bins.append(
            {
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "count": len(selected),
                "mean_predicted_late_rate": round(mean_predicted, 4),
                "observed_late_rate": round(observed_rate, 4),
                "absolute_calibration_gap": round(
                    abs(mean_predicted - observed_rate),
                    4,
                ),
            }
        )
    return bins


def backtest_business(
    session: Session,
    business_id: str,
    min_history: int = 3,
) -> dict:
    if min_history < 0:
        raise ValueError("min_history must be non-negative")

    invoices = session.scalars(
        select(Invoice)
        .options(selectinload(Invoice.payments))
        .where(Invoice.business_id == business_id)
        .order_by(Invoice.invoice_date, Invoice.id)
    ).all()

    customer_invoices: dict[int, list[Invoice]] = {}
    for invoice in invoices:
        customer_invoices.setdefault(invoice.customer_id, []).append(invoice)

    scored = []
    skipped_cold_start = 0

    for invoice in invoices:
        paid_total = sum(
            (payment.amount for payment in invoice.payments),
            Decimal("0"),
        )
        if paid_total < invoice.amount:
            continue

        history = _history_before_invoice(
            customer_invoices[invoice.customer_id],
            invoice,
        )
        if history.invoice_count < min_history:
            skipped_cold_start += 1
            continue

        prediction = predict_invoice_risk(
            InvoiceInput(
                due_date=invoice.due_date,
                amount=float(invoice.amount),
                buyer=history,
            )
        )
        actual_payment_date = max(
            payment.payment_date for payment in invoice.payments
        )
        actual_delay_days = (actual_payment_date - invoice.due_date).days
        actual_late = actual_delay_days > 0

        scored.append(
            {
                "invoice_id": invoice.id,
                "invoice_number": invoice.invoice_number,
                "customer_id": invoice.customer_id,
                "invoice_date": invoice.invoice_date,
                "actual_payment_date": actual_payment_date,
                "actual_delay_days": actual_delay_days,
                "actual_late": actual_late,
                "predicted_late_probability": prediction["late_probability"],
                "predicted_payment_date": prediction["expected_payment_date"],
                "predicted_delay_days": prediction["expected_delay_days"],
            }
        )

    if not scored:
        return {
            "business_id": business_id,
            "model_version": MODEL_VERSION,
            "min_history": min_history,
            "invoices_seen": len(invoices),
            "eligible_settled_invoices": sum(
                1
                for invoice in invoices
                if sum((payment.amount for payment in invoice.payments), Decimal("0"))
                >= invoice.amount
            ),
            "backtested_invoices": 0,
            "skipped_cold_start": skipped_cold_start,
            "date_mae_days": None,
            "mean_brier_error": None,
            "late_classification_accuracy": None,
            "calibration_bins": [],
        }

    predictions = [
        float(item["predicted_late_probability"]) for item in scored
    ]
    actuals = [bool(item["actual_late"]) for item in scored]
    date_errors = [
        abs(
            (
                item["actual_payment_date"]
                - item["predicted_payment_date"]
            ).days
        )
        for item in scored
    ]
    brier_errors = [
        (prediction - float(actual)) ** 2
        for prediction, actual in zip(predictions, actuals)
    ]
    correct = sum(
        (prediction >= 0.5) == actual
        for prediction, actual in zip(predictions, actuals)
    )

    return {
        "business_id": business_id,
        "model_version": MODEL_VERSION,
        "min_history": min_history,
        "invoices_seen": len(invoices),
        "backtested_invoices": len(scored),
        "skipped_cold_start": skipped_cold_start,
        "date_mae_days": round(sum(date_errors) / len(date_errors), 2),
        "mean_brier_error": round(sum(brier_errors) / len(brier_errors), 6),
        "late_classification_accuracy": round(correct / len(scored), 4),
        "observed_late_rate": round(sum(actuals) / len(actuals), 4),
        "mean_predicted_late_rate": round(sum(predictions) / len(predictions), 4),
        "calibration_bins": _calibration_bins(predictions, actuals),
    }
