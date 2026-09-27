from datetime import date, datetime, timedelta
from decimal import Decimal
import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app import db
from backend.app.entities import Business, Customer, Invoice, ModelRegistry, Payment
from backend.app.ml import (
    ML_MODEL_VERSION,
    active_model_status,
    predict_invoice_risk_with_active_model,
    rollback_active_model,
    train_business_model,
)


def test_temporal_model_training_persists_registry_entry(monkeypatch):
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

            status_before = active_model_status(session, business.id)
            assert status_before["active"] is False

            result = train_business_model(
                session,
                business.id,
                min_history=3,
                test_fraction=0.30,
            )

            assert result["model_version"].startswith(ML_MODEL_VERSION + "-")
            assert result["train_count"] >= 8
            assert result["test_count"] >= 4
            assert result["candidate_metrics"]["mean_brier_error"] >= 0
            assert result["baseline_metrics"]["mean_brier_error"] >= 0
            assert result["promotion_status"] in {"promoted", "candidate_only"}

            rows = session.scalars(
                select(ModelRegistry).where(
                    ModelRegistry.business_id == business.id
                )
            ).all()
            assert len(rows) == 1
            assert rows[0].model_version == result["model_version"]
            assert rows[0].artifact

            status_after = active_model_status(session, business.id)
            if result["promotion_status"] == "promoted":
                assert status_after["active"] is True
                assert status_after["model_version"] == result["model_version"]
            else:
                assert status_after["active"] is False
        finally:
            session.close()
    finally:
        db.engine, db.SessionLocal = old_engine, old_session


def test_model_registry_supports_explicit_rollback():
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
            business = Business(id="rollback-test", name="Rollback Test")
            session.add(business)
            session.flush()

            first = ModelRegistry(
                business_id=business.id,
                model_version="ml-v0.1-first",
                status="superseded",
                trained_at=datetime(2026, 9, 1),
                train_count=20,
                test_count=6,
                metrics='{"candidate":{}}',
                artifact='{"model_version":"ml-v0.1-first"}',
                reason="Previous promoted model",
            )
            second = ModelRegistry(
                business_id=business.id,
                model_version="ml-v0.1-second",
                status="active",
                trained_at=datetime(2026, 9, 2),
                train_count=22,
                test_count=7,
                metrics='{"candidate":{}}',
                artifact='{"model_version":"ml-v0.1-second"}',
                reason="Current promoted model",
            )
            session.add_all([first, second])
            session.commit()

            result = rollback_active_model(
                session,
                business.id,
                target_version="ml-v0.1-first",
            )

            assert result["status"] == "rolled_back"
            assert result["rolled_back_from"] == "ml-v0.1-second"
            assert result["active_model_version"] == "ml-v0.1-first"

            status = active_model_status(session, business.id)
            assert status["active"] is True
            assert status["model_version"] == "ml-v0.1-first"

            states = {
                row.model_version: row.status
                for row in session.scalars(
                    select(ModelRegistry).where(
                        ModelRegistry.business_id == business.id
                    )
                )
            }
            assert states["ml-v0.1-second"] == "rolled_back"
            assert states["ml-v0.1-first"] == "active"
        finally:
            session.close()
    finally:
        db.engine, db.SessionLocal = old_engine, old_session



def test_active_model_prediction_preserves_model_lineage():
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
            business = Business(id="lineage-test", name="Lineage Test")
            customer = Customer(
                business_id=business.id,
                external_key="buyer-lineage",
                name="Buyer Lineage",
            )
            session.add_all([business, customer])
            session.flush()

            invoice = Invoice(
                business_id=business.id,
                customer_id=customer.id,
                invoice_number="LIVE-1",
                invoice_date=date(2026, 9, 20),
                due_date=date(2026, 10, 1),
                amount=Decimal("100000.00"),
            )
            session.add(invoice)
            session.flush()
            session.add(
                ModelRegistry(
                    business_id=business.id,
                    model_version="ml-v0.1-active",
                    status="active",
                    trained_at=datetime(2026, 9, 20),
                    train_count=20,
                    test_count=6,
                    metrics='{"candidate":{"mean_brier_error":0.1,"date_mae_days":2.0}}',
                    artifact=json.dumps({
                        "model_version": "ml-v0.1-active",
                        "scaler_mean": [0.0] * 10,
                        "scaler_scale": [1.0] * 10,
                        "classifier_coefficients": [0.0] * 10,
                        "classifier_intercept": 0.0,
                        "regressor_coefficients": [0.0] * 10,
                        "regressor_intercept": 2.0,
                        "train_count": 20,
                    }),
                    reason="Test active model",
                )
            )
            session.commit()

            result = predict_invoice_risk_with_active_model(session, invoice.id)

            assert result["model_version"] == "ml-v0.1-active"
            assert "ml-v0.1-active" in result["reasons"][-1]
        finally:
            session.close()
    finally:
        db.engine, db.SessionLocal = old_engine, old_session
