from decimal import Decimal
from datetime import date, datetime, timezone

from backend.app.db import Base, SessionLocal, engine
from backend.app.entities import Business, Customer, Invoice, Payment, RiskPrediction
from backend.app.evaluator import evaluate_prediction, evaluation_summary


def test_prediction_evaluation_and_summary():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        business = Business(id="evaluation-test", name="Evaluation Test")
        session.add(business)
        session.flush()

        customer = Customer(
            business_id=business.id,
            external_key="acme-eval",
            name="Acme Evaluation",
        )
        session.add(customer)
        session.flush()

        invoice = Invoice(
            business_id=business.id,
            customer_id=customer.id,
            invoice_number="EVAL-1",
            invoice_date=date(2026, 9, 1),
            due_date=date(2026, 9, 15),
            amount=Decimal("100000.00"),
        )
        session.add(invoice)
        session.flush()

        prediction = RiskPrediction(
            invoice_id=invoice.id,
            predicted_at=datetime.now(timezone.utc),
            late_probability=0.80,
            expected_delay_days=10,
            expected_payment_date=date(2026, 9, 25),
            cash_at_risk=Decimal("80000.00"),
            model_version="baseline-v0.1",
            reasons="[]",
        )
        session.add(prediction)
        session.flush()

        session.add(Payment(
            invoice_id=invoice.id,
            payment_date=date(2026, 9, 28),
            amount=Decimal("100000.00"),
        ))
        session.commit()

        result = evaluate_prediction(session, prediction.id)
        assert result["actual_late"] is True
        assert result["actual_delay_days"] == 13
        assert result["payment_date_error_days"] == 3
        assert result["probability_brier_error"] == 0.04

        summary = evaluation_summary(session, business.id)
        assert summary["evaluated_predictions"] == 1
        assert summary["date_mae_days"] == 3.0
        assert summary["late_classification_accuracy"] == 1.0

        repeat = evaluate_prediction(session, prediction.id)
        assert repeat["prediction_id"] == prediction.id
    finally:
        session.close()
