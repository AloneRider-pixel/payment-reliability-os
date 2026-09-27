from datetime import date

from backend.app.features import extract_buyer_features
from backend.app.models import BuyerHistory, InvoiceInput
from backend.app.scoring import predict_invoice_risk, score_buyer


def test_good_buyer_scores_high():
    result = score_buyer(
        BuyerHistory(
            payment_delays_days=[0, 0, 1, 0, 0],
            invoice_count=5,
            late_invoice_count=1,
            average_invoice_amount=100000,
        )
    )
    assert result["score"] >= 80
    assert result["grade"] in {"A", "B", "C"}
    assert result["features"]["on_time_rate"] == 0.8
    assert result["factors"]
    assert abs(
        sum(factor["contribution"] for factor in result["factors"]) - result["score"]
    ) < 0.1


def test_bad_buyer_scores_low():
    result = score_buyer(
        BuyerHistory(
            payment_delays_days=[15, 20, 18, 25, 17],
            invoice_count=5,
            late_invoice_count=5,
            average_invoice_amount=200000,
        )
    )
    assert result["score"] < 30
    assert result["grade"] in {"E", "F"}
    assert result["features"]["late_payment_rate"] == 1.0


def test_feature_engine_handles_single_observation():
    features = extract_buyer_features(
        BuyerHistory(
            payment_delays_days=[7],
            invoice_count=1,
            late_invoice_count=1,
            recent_delays_days=[7],
            average_invoice_amount=50000,
            current_outstanding_amount=12000,
        )
    )
    assert features["on_time_rate"] == 0.0
    assert features["average_delay_days"] == 7.0
    assert features["median_delay_days"] == 7.0
    assert features["p90_delay_days"] == 7.0
    assert features["recent_trend_days"] == 0.0
    assert features["payment_consistency"] == 1.0
    assert features["current_outstanding_exposure"] == 12000.0
    assert features["evidence_count"] == 1


def test_cold_start_is_explicit():
    result = score_buyer(BuyerHistory())
    assert result["score"] == 50.0
    assert result["grade"] == "N/A"
    assert result["confidence"] == "LOW"
    assert result["model_version"] == "baseline-v1.0"
    assert result["factors"] == []


def test_invoice_risk_has_explainable_output():
    result = predict_invoice_risk(
        InvoiceInput(
            due_date=date(2026, 10, 15),
            amount=300000,
            buyer=BuyerHistory(
                payment_delays_days=[10, 15, 20],
                invoice_count=3,
                late_invoice_count=3,
                average_invoice_amount=200000,
            ),
        )
    )
    assert 0 < result["late_probability"] <= 0.99
    assert result["expected_payment_date"] >= date(2026, 10, 15)
    assert result["cash_at_risk"] > 0
    assert result["reasons"]
    assert result["model_version"] == "baseline-v1.0"
