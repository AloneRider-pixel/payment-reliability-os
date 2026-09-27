from datetime import date, datetime, timedelta
from decimal import Decimal
import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app import db
from backend.app.entities import (
    Business,
    Customer,
    Invoice,
    Payment,
    PredictionEvaluation,
    RiskPrediction,
    ScheduledJobRun,
)
from backend.app.scheduler import (
    JOB_EVALUATE,
    JOB_MAINTAIN_MODEL,
    JOB_REFRESH_ACTIONS,
    JOB_REFRESH_RISK,
    daily_slot,
    run_scheduled_jobs,
)


def _session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    old_engine, old_session = db.engine, db.SessionLocal
    db.engine, db.SessionLocal = engine, Session
    db.init_db()
    return old_engine, old_session, Session()


def test_daily_scheduler_is_idempotent_and_runs_all_stages():
    old_engine, old_session, session = _session()
    try:
        business = Business(id="schedule-test", name="Schedule Test")
        customer = Customer(
            business_id=business.id,
            external_key="buyer-schedule",
            name="Buyer Schedule",
        )
        session.add_all([business, customer])
        session.flush()

        invoice = Invoice(
            business_id=business.id,
            customer_id=customer.id,
            invoice_number="SCH-1",
            invoice_date=date(2026, 8, 1),
            due_date=date(2026, 8, 15),
            amount=Decimal("100000.00"),
        )
        session.add(invoice)
        session.flush()
        session.add(
            Payment(
                invoice_id=invoice.id,
                payment_date=date(2026, 8, 20),
                amount=Decimal("100000.00"),
            )
        )
        session.add(
            RiskPrediction(
                invoice_id=invoice.id,
                predicted_at=datetime(2026, 8, 10, 10, 0),
                late_probability=Decimal("0.80"),
                expected_delay_days=Decimal("5.0"),
                expected_payment_date=date(2026, 8, 20),
                cash_at_risk=Decimal("80000.00"),
                model_version="baseline-v1.0",
                reasons=json.dumps(["scheduled-test"]),
            )
        )
        session.commit()

        slot = daily_slot(date(2026, 9, 28))
        first = run_scheduled_jobs(
            session,
            scheduled_for=slot,
            as_of=date(2026, 9, 28),
            business_id=business.id,
        )

        assert first["businesses_seen"] == 1
        assert first["completed"] == 4
        assert first["failed"] == 0
        assert first["idempotent_skips"] == 0

        runs = session.scalars(
            select(ScheduledJobRun).where(
                ScheduledJobRun.business_id == business.id,
                ScheduledJobRun.scheduled_for == slot,
            )
        ).all()
        assert {run.job_type for run in runs} == {
            JOB_EVALUATE,
            JOB_MAINTAIN_MODEL,
            JOB_REFRESH_RISK,
            JOB_REFRESH_ACTIONS,
        }
        assert all(run.status == "completed" for run in runs)

        second = run_scheduled_jobs(
            session,
            scheduled_for=slot,
            as_of=date(2026, 9, 28),
            business_id=business.id,
        )
        assert second["completed"] == 0
        assert second["failed"] == 0
        assert second["idempotent_skips"] == 4

        evaluated = session.scalar(select(PredictionEvaluation))
        assert evaluated is not None
    finally:
        session.close()
        db.engine, db.SessionLocal = old_engine, old_session


def test_daily_slot_is_stable_for_same_operational_date():
    assert daily_slot(date(2026, 9, 28)) == datetime(2026, 9, 28, 2, 0)
    assert daily_slot(date(2026, 9, 28)) == daily_slot(date(2026, 9, 28))
    assert daily_slot(date(2026, 9, 29)) - daily_slot(date(2026, 9, 28)) == timedelta(days=1)
