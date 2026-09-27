from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import db
from backend.app.entities import Business, Customer, Invoice, Payment
from backend.app.ml import (
    ML_MODEL_VERSION,
    active_model_status,
    train_business_model,
)


def test_temporal_model_training_uses_prior_history_and_writes_artifact(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    old_engine, old_session = db.engine, db.SessionLocal
    try:
        db.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        db.SessionLocal = sessionmaker(
            bind=db.engine,
            autoflush=False,
            autocommit=False,
        )
        db.init_db()

        session = db.SessionLocal()
        try:
            business = Business(id="ml-test", name="ML Test")
            customer = Customer(
                business_id=business.id,
                external_key="buyer-ml",
                name="Buyer ML",
            )
            session.add_all([business, customer])
            session.flush()

            for index in range(20):
                invoice_date = date(2026, 1, 1) + timedelta(days=index * 5)
                due_date = invoice_date + timedelta(days=10)
                delay = 0 if index < 10 else 8

                invoice = Invoice(
                    business_id=business.id,
                    customer_id=customer.id,
                    invoice_number=f"ML-{index + 1}",
                    invoice_date=invoice_date,
                    due_date=due_date,
                    amount=Decimal("100000.00"),
                )
                session.add(invoice)
                session.flush()
                session.add(
                    Payment(
                        invoice_id=invoice.id,
                        payment_date=due_date + timedelta(days=delay),
                        amount=Decimal("100000.00"),
                    )
                )

            session.commit()

            status_before = active_model_status(business.id)
            assert status_before["active"] is False

            result = train_business_model(
                session,
                business.id,
                min_history=3,
                test_fraction=0.30,
            )

            assert result["model_version"] == ML_MODEL_VERSION
            assert result["train_count"] >= 8
            assert result["test_count"] >= 4
            assert result["candidate_metrics"]["mean_brier_error"] >= 0
            assert result["baseline_metrics"]["mean_brier_error"] >= 0
            assert result["promotion_status"] in {"promoted", "candidate_only"}

            candidate = tmp_path / ".models" / "ml-test.candidate.json"
            assert candidate.exists()

            status_after = active_model_status(business.id)
            if result["promotion_status"] == "promoted":
                assert status_after["active"] is True
                assert status_after["model_version"] == ML_MODEL_VERSION
            else:
                assert status_after["active"] is False
        finally:
            session.close()
    finally:
        db.engine, db.SessionLocal = old_engine, old_session
