from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app import db


def test_ingestion_and_scoring():
    old_engine, old_session = db.engine, db.SessionLocal
    try:
        with TemporaryDirectory() as tmp:
            db.engine = create_engine(
                f"sqlite:///{Path(tmp) / 'test.db'}",
                connect_args={"check_same_thread": False},
            )
            db.SessionLocal = sessionmaker(
                bind=db.engine,
                autoflush=False,
                autocommit=False,
            )
            db.init_db()

            from backend.app.main import app

            client = TestClient(app)
            invoice_csv = (
                b"invoice_number,customer,invoice_date,due_date,amount\n"
                b"INV-1,Acme,2026-09-01,2026-09-15,100000\n"
            )
            payment_csv = (
                b"invoice_number,payment_date,amount_paid\n"
                b"INV-1,2026-09-25,100000\n"
            )

            response = client.post(
                "/imports/invoices?business_id=test-import",
                files={"file": ("invoices.csv", invoice_csv, "text/csv")},
            )
            assert response.status_code == 200
            assert response.json()["created"] == 1

            response = client.post(
                "/imports/payments?business_id=test-import",
                files={"file": ("payments.csv", payment_csv, "text/csv")},
            )
            assert response.status_code == 200
            assert response.json()["created"] == 1

            customers = client.get("/customers?business_id=test-import")
            assert customers.status_code == 200
            assert len(customers.json()) == 1

            customer_id = customers.json()[0]["id"]
            score = client.get(f"/customers/{customer_id}/score")
            assert score.status_code == 200
            assert score.json()["grade"] in {"A", "B", "C", "D", "E", "F"}

            invoice_rows = client.get("/dashboard?business_id=test-import")
            assert invoice_rows.status_code == 200
            assert invoice_rows.json()["invoice_count"] == 1

            prediction = client.post("/risk/analyze/1")
            assert prediction.status_code == 200
            assert 0 < prediction.json()["late_probability"] <= 0.99

            predictions = client.get("/predictions?business_id=test-import")
            assert predictions.status_code == 200
            assert len(predictions.json()) == 1
    finally:
        db.engine, db.SessionLocal = old_engine, old_session
