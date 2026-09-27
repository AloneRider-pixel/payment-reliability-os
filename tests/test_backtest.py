from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app import db
from backend.app.entities import Business, Customer, Invoice, Payment, RiskPrediction
from backend.app.evaluator import backtest_business


def test_historical_backtest_is_chronological_and_read_only():
    old_engine, old_session = db.engine, db.SessionLocal
    try:
        with TemporaryDirectory() as tmp:
            db.engine = create_engine(
                f"sqlite:///{Path(tmp) / 'backtest.db'}",
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
                business = Business(id="backtest-test", name="Backtest Test")
                customer = Customer(
                    business_id=business.id,
                    external_key="buyer-1",
                    name="Buyer One",
                )
                session.add_all([business, customer])
                session.flush()

                fixtures = [
                    ("INV-1", date(2026, 1, 1), date(2026, 1, 10), date(2026, 1, 10)),
                    ("INV-2", date(2026, 1, 10), date(2026, 1, 20), date(2026, 1, 25)),
                    ("INV-3", date(2026, 2, 1), date(2026, 2, 10), date(2026, 2, 9)),
                    ("INV-4", date(2026, 2, 10), date(2026, 2, 20), date(2026, 3, 1)),
                    ("INV-5", date(2026, 3, 1), date(2026, 3, 10), date(2026, 3, 10)),
                ]

                for number, invoice_date, due_date, payment_date in fixtures:
                    invoice = Invoice(
                        business_id=business.id,
                        customer_id=customer.id,
                        invoice_number=number,
                        invoice_date=invoice_date,
                        due_date=due_date,
                        amount=Decimal("100000.00"),
                    )
                    session.add(invoice)
                    session.flush()
                    session.add(
                        Payment(
                            invoice_id=invoice.id,
                            payment_date=payment_date,
                            amount=Decimal("100000.00"),
                        )
                    )
                session.commit()

                result = backtest_business(session, business.id, min_history=2)

                assert result["invoices_seen"] == 5
                assert result["backtested_invoices"] == 3
                assert result["skipped_cold_start"] == 2
                assert result["mean_brier_error"] is not None
                assert result["date_mae_days"] is not None
                assert result["late_classification_accuracy"] is not None
                assert result["observed_late_rate"] == 0.6667
                assert result["calibration_bins"]
                assert session.scalars(select(RiskPrediction)).all() == []
            finally:
                session.close()
    finally:
        db.engine, db.SessionLocal = old_engine, old_session
