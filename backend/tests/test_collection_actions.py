from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from backend.app import actions
from backend.app.entities import CollectionAction, Invoice, Payment, RiskPrediction


class FakeScalarRows:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, invoice):
        self.invoice = invoice
        self.added = None
        self.committed = False

    def scalars(self, statement):
        return FakeScalarRows([self.invoice])

    def scalar(self, statement):
        return None

    def add(self, value):
        self.added = value

    def commit(self):
        self.committed = True


def test_collection_action_uses_outstanding_amount(monkeypatch):
    invoice = Invoice(
        id=1,
        business_id="biz-1",
        customer_id=1,
        invoice_number="INV-1",
        invoice_date=date(2026, 9, 1),
        due_date=date(2026, 9, 15),
        amount=Decimal("1000.00"),
        currency="INR",
    )
    invoice.payments = [
        Payment(
            invoice_id=1,
            payment_date=date(2026, 9, 20),
            amount=Decimal("400.00"),
        )
    ]
    prediction = RiskPrediction(
        id=10,
        invoice_id=1,
        predicted_at=__import__("datetime").datetime(2026, 9, 21),
        late_probability=0.80,
        expected_delay_days=10.0,
        expected_payment_date=date(2026, 9, 25),
        cash_at_risk=Decimal("480.00"),
        model_version="baseline-v1.0",
        reasons="[]",
    )

    session = FakeSession(invoice)
    monkeypatch.setattr(actions, "_latest_prediction", lambda _session, _invoice_id: prediction)
    monkeypatch.setattr(
        actions,
        "_action_plan",
        lambda **_: {
            "action_type": "PRIORITY_COLLECTION",
            "priority_score": 80,
            "days_overdue": 6,
            "next_step": "Confirm payment",
            "reason": "Overdue",
        },
    )

    result = actions.generate_collection_actions(session, "biz-1", as_of=date(2026, 9, 21))

    assert result["actions_created"] == 1
    assert isinstance(session.added, CollectionAction)
    assert session.added.amount == Decimal("600.00")
    assert session.committed is True
