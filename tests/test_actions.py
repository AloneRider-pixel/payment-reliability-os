from datetime import date, datetime, timedelta
from decimal import Decimal
import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app import db
from backend.app.actions import (
    ACTION_VERSION,
    action_summary,
    generate_collection_actions,
    list_collection_actions,
    update_collection_action_status,
)
from backend.app.entities import (
    Business,
    CollectionAction,
    Customer,
    Invoice,
    Payment,
    RiskPrediction,
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


def test_generate_collection_actions_prioritizes_overdue_risk():
    old_engine, old_session, session = _session()
    try:
        business = Business(id="actions-test", name="Actions Test")
        customer = Customer(
            business_id=business.id,
            external_key="buyer-actions",
            name="Buyer Actions",
        )
        session.add_all([business, customer])
        session.flush()

        overdue_invoice = Invoice(
            business_id=business.id,
            customer_id=customer.id,
            invoice_number="A-1",
            invoice_date=date(2026, 8, 1),
            due_date=date(2026, 9, 1),
            amount=Decimal("100000.00"),
        )
        future_invoice = Invoice(
            business_id=business.id,
            customer_id=customer.id,
            invoice_number="A-2",
            invoice_date=date(2026, 8, 20),
            due_date=date(2026, 10, 5),
            amount=Decimal("50000.00"),
        )
        session.add_all([overdue_invoice, future_invoice])
        session.flush()

        session.add_all([
            RiskPrediction(
                invoice_id=overdue_invoice.id,
                predicted_at=datetime(2026, 9, 28, 10, 0),
                late_probability=Decimal("0.92"),
                expected_delay_days=Decimal("18.0"),
                expected_payment_date=date(2026, 9, 19),
                cash_at_risk=Decimal("92000.00"),
                model_version="ml-test-1",
                reasons=json.dumps(["high late risk"]),
            ),
            RiskPrediction(
                invoice_id=future_invoice.id,
                predicted_at=datetime(2026, 9, 28, 10, 0),
                late_probability=Decimal("0.20"),
                expected_delay_days=Decimal("1.0"),
                expected_payment_date=date(2026, 10, 6),
                cash_at_risk=Decimal("10000.00"),
                model_version="baseline-v1.0",
                reasons=json.dumps(["low risk"]),
            ),
        ])
        session.commit()

        result = generate_collection_actions(
            session,
            business.id,
            as_of=date(2026, 9, 28),
        )

        assert result["action_version"] == ACTION_VERSION
        assert result["actions_created"] == 2
        assert result["settled_skipped"] == 0
        assert result["unscored_skipped"] == 0

        actions = list_collection_actions(session, business.id)
        assert len(actions) == 2
        assert actions[0]["invoice_number"] == "A-1"
        assert actions[0]["action_type"] == "ESCALATION_REVIEW"
        assert actions[0]["days_overdue"] == 27
        assert actions[0]["priority_score"] > actions[1]["priority_score"]
        assert "outstanding exposure" in actions[0]["reason"]
    finally:
        session.close()
        db.engine, db.SessionLocal = old_engine, old_session


def test_collection_action_refreshes_on_same_prediction_without_duplicates():
    old_engine, old_session, session = _session()
    try:
        business = Business(id="actions-refresh", name="Actions Refresh")
        customer = Customer(
            business_id=business.id,
            external_key="buyer-refresh",
            name="Buyer Refresh",
        )
        session.add_all([business, customer])
        session.flush()

        invoice = Invoice(
            business_id=business.id,
            customer_id=customer.id,
            invoice_number="R-1",
            invoice_date=date(2026, 9, 1),
            due_date=date(2026, 10, 1),
            amount=Decimal("40000.00"),
        )
        session.add(invoice)
        session.flush()
        prediction = RiskPrediction(
            invoice_id=invoice.id,
            predicted_at=datetime(2026, 9, 28, 10, 0),
            late_probability=Decimal("0.80"),
            expected_delay_days=Decimal("10.0"),
            expected_payment_date=date(2026, 10, 11),
            cash_at_risk=Decimal("32000.00"),
            model_version="ml-refresh-1",
            reasons=json.dumps(["risk"]),
        )
        session.add(prediction)
        session.commit()

        first = generate_collection_actions(
            session,
            business.id,
            as_of=date(2026, 9, 28),
        )
        second = generate_collection_actions(
            session,
            business.id,
            as_of=date(2026, 9, 29),
        )

        assert first["actions_created"] == 1
        assert second["actions_created"] == 0
        assert second["actions_updated"] == 1

        rows = session.scalars(
            select(CollectionAction).where(
                CollectionAction.business_id == business.id
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].days_overdue == 0
    finally:
        session.close()
        db.engine, db.SessionLocal = old_engine, old_session


def test_collection_action_status_and_summary():
    old_engine, old_session, session = _session()
    try:
        business = Business(id="actions-status", name="Actions Status")
        session.add(business)
        session.flush()

        action = CollectionAction(
            business_id=business.id,
            invoice_id=1,
            prediction_id=None,
            model_version="baseline-v1.0",
            action_type="PRE_DUE_REMINDER",
            priority_score=35.0,
            status="open",
            due_date=date(2026, 10, 1),
            days_overdue=0,
            amount=Decimal("25000.00"),
            reason="Send a reminder.",
            created_at=datetime(2026, 9, 28, 10, 0),
            updated_at=datetime(2026, 9, 28, 10, 0),
        )
        # Use a real invoice FK target so SQLite constraints remain portable.
        customer = Customer(
            business_id=business.id,
            external_key="buyer-status",
            name="Buyer Status",
        )
        session.add(customer)
        session.flush()
        invoice = Invoice(
            business_id=business.id,
            customer_id=customer.id,
            invoice_number="S-1",
            invoice_date=date(2026, 9, 1),
            due_date=date(2026, 10, 1),
            amount=Decimal("25000.00"),
        )
        session.add(invoice)
        session.flush()
        action.invoice_id = invoice.id
        session.add(action)
        session.commit()

        summary = action_summary(session, business.id)
        assert summary["open_actions"] == 1
        assert summary["cash_in_action_queue"] == 25000.0
        assert summary["next_action"] == "PRE_DUE_REMINDER"

        updated = update_collection_action_status(
            session,
            action.id,
            "completed",
        )
        assert updated["status"] == "completed"

        assert action_summary(session, business.id)["open_actions"] == 0

        reopened = update_collection_action_status(
            session,
            action.id,
            "open",
        )
        assert reopened["status"] == "open"

        try:
            update_collection_action_status(session, action.id, "invalid")
            raise AssertionError("invalid status should raise ValueError")
        except ValueError:
            pass
    finally:
        session.close()
        db.engine, db.SessionLocal = old_engine, old_session
