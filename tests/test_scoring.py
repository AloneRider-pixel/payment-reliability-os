from datetime import date

from backend.app.models import BuyerHistory, InvoiceInput
from backend.app.scoring import predict_invoice_risk, score_buyer

def test_good_buyer_scores_high():
    result = score_buyer(BuyerHistory(
        payment_delays_days=[0, 0, 1, 0, 0],
        invoice_count=5,
        late_invoice_count=1,
        average_invoice_amount=100000,
    ))
    assert result["score"] >= 80
    assert result["grade"] in {"A", "B", "C"}

def test_bad_buyer_scores_low():
    result = score_buyer(BuyerHistory(
        payment_delays_days=[15, 20, 18, 25, 17],
        invoice_count=5,
        late_invoice_count=5,
        average_invoice_amount=200000,
    ))
    assert result["score"] < 30
    assert result["grade"] in {"E", "F"}

def test_invoice_risk_has_explainable_output():
    result = predict_invoice_risk(InvoiceInput(
        due_date=date(2026, 10, 15),
        amount=300000,
        buyer=BuyerHistory(
            payment_delays_days=[10, 15, 20],
            invoice_count=3,
            late_invoice_count=3,
            average_invoice_amount=200000,
        ),
    ))
    assert 0 < result["late_probability"] <= 0.99
    assert result["expected_payment_date"] >= date(2026, 10, 15)
    assert result["cash_at_risk"] > 0
    assert result["reasons"]
