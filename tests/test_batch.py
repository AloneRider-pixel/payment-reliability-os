from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import db

def test_batch_analyzes_only_outstanding_invoices():
    old_engine, old_session = db.engine, db.SessionLocal
    try:
        with TemporaryDirectory() as tmp:
            db.engine = create_engine(
                f"sqlite:///{Path(tmp) / 'batch.db'}",
                connect_args={"check_same_thread": False},
            )
            db.SessionLocal = sessionmaker(bind=db.engine, autoflush=False, autocommit=False)
            db.init_db()

            from backend.app.main import app
            client = TestClient(app)

            invoices = b"invoice_number,customer,invoice_date,due_date,amount\nINV-1,Acme,2026-09-01,2026-09-15,100000\nINV-2,Acme,2026-09-20,2026-10-05,150000\n"
            payments = b"invoice_number,payment_date,amount_paid\nINV-1,2026-09-25,100000\n"

            assert client.post(
                "/imports/invoices?business_id=batch-test",
                files={"file": ("invoices.csv", invoices, "text/csv")},
            ).status_code == 200
            assert client.post(
                "/imports/payments?business_id=batch-test",
                files={"file": ("payments.csv", payments, "text/csv")},
            ).status_code == 200

            result = client.post("/risk/analyze-all?business_id=batch-test")
            assert result.status_code == 200
            assert result.json()["invoices_seen"] == 2
            assert result.json()["settled_skipped"] == 1
            assert result.json()["predictions_created"] == 1
    finally:
        db.engine, db.SessionLocal = old_engine, old_session