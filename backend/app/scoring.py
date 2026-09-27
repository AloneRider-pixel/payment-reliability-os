from datetime import timedelta

from backend.app.features import extract_buyer_features
from backend.app.models import BuyerHistory, InvoiceInput

MODEL_VERSION = "baseline-v1.0"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    if score >= 50:
        return "E"
    return "F"


def _factor(name: str, value: float, weight: float, contribution: float, direction: str) -> dict:
    return {
        "name": name,
        "value": round(value, 4),
        "weight": weight,
        "contribution": round(contribution, 2),
        "direction": direction,
    }


def score_buyer(history: BuyerHistory) -> dict:
    features = extract_buyer_features(history)
    evidence_count = features["evidence_count"]

    if evidence_count == 0:
        return {
            "score": 50.0,
            "grade": "N/A",
            "late_probability": 0.50,
            "expected_delay_days": 0.0,
            "reasons": ["No settled payment history is available"],
            "evidence_count": 0,
            "confidence": "LOW",
            "model_version": MODEL_VERSION,
            "features": features,
            "factors": [],
        }

    late_rate = features["late_payment_rate"]
    avg_delay = features["average_delay_days"]
    trend_days = features["recent_trend_days"]
    consistency = features["payment_consistency"]

    late_score = (1.0 - late_rate) * 100.0
    delay_score = (1.0 - _clamp(max(avg_delay, 0.0) / 30.0, 0.0, 1.0)) * 100.0
    consistency_score = consistency * 100.0
    trend_score = (1.0 - _clamp(max(trend_days, 0.0) / 15.0, 0.0, 1.0)) * 100.0

    factors = [
        _factor("Late-payment rate", late_rate, 0.50, late_score * 0.50, "risk"),
        _factor("Average delay", avg_delay, 0.30, delay_score * 0.30, "risk"),
        _factor("Payment consistency", consistency, 0.10, consistency_score * 0.10, "stability"),
        _factor("Recent trend", trend_days, 0.10, trend_score * 0.10, "trend"),
    ]

    score = round(
        _clamp(sum(factor["contribution"] for factor in factors), 0.0, 100.0),
        1,
    )

    # Probability is deliberately separate from the score: the score is
    # an operational behavior index, while this estimates the chance of
    # a late payment on a future invoice.
    late_probability = _clamp(
        late_rate * 0.60
        + _clamp(max(avg_delay, 0.0) / 30.0, 0.0, 1.0) * 0.25
        + _clamp(max(trend_days, 0.0) / 15.0, 0.0, 1.0) * 0.10
        + (1.0 - consistency) * 0.05,
        0.01,
        0.99,
    )

    confidence = "HIGH" if evidence_count >= 12 else "MEDIUM" if evidence_count >= 5 else "LOW"

    reasons = []
    if late_rate >= 0.50:
        reasons.append(f"High historical late-payment rate ({late_rate:.0%})")
    elif late_rate > 0:
        reasons.append(f"Historical late-payment rate ({late_rate:.0%})")
    else:
        reasons.append("No historical late payments in the supplied history")

    if avg_delay > 0:
        reasons.append(f"Average payment delay is {avg_delay:.1f} days")
    if trend_days > 2:
        reasons.append(f"Recent payment behavior is deteriorating ({trend_days:.1f} days)")
    elif trend_days < -2:
        reasons.append(f"Recent payment behavior is improving ({abs(trend_days):.1f} days)")
    if consistency < 0.60:
        reasons.append("Payment timing is highly variable")
    reasons.append(f"Evidence: {evidence_count} settled invoices")

    return {
        "score": score,
        "grade": _grade(score),
        "late_probability": round(late_probability, 4),
        "expected_delay_days": round(max(avg_delay, 0.0), 1),
        "reasons": reasons,
        "evidence_count": evidence_count,
        "confidence": confidence,
        "model_version": MODEL_VERSION,
        "features": features,
        "factors": factors,
    }


def predict_invoice_risk(payload: InvoiceInput) -> dict:
    buyer_result = score_buyer(payload.buyer)
    average_invoice_amount = float(payload.buyer.average_invoice_amount)
    amount_ratio = payload.amount / average_invoice_amount if average_invoice_amount > 0 else 1.0
    anomaly_penalty = _clamp(max(amount_ratio - 1.0, 0.0) / 2.0, 0.0, 1.0)

    late_probability = _clamp(
        buyer_result["late_probability"] * 0.85 + anomaly_penalty * 0.15,
        0.01,
        0.99,
    )
    expected_delay = round(
        buyer_result["expected_delay_days"] * (1.0 + 0.25 * anomaly_penalty),
        1,
    )
    expected_date = payload.due_date + timedelta(days=round(expected_delay))
    cash_at_risk = round(payload.amount * late_probability, 2)

    reasons = list(buyer_result["reasons"])
    if amount_ratio >= 1.5 and average_invoice_amount > 0:
        reasons.append(
            f"Current invoice is {amount_ratio:.1f}x the buyer's average invoice amount"
        )

    return {
        "late_probability": round(late_probability, 4),
        "expected_delay_days": expected_delay,
        "expected_payment_date": expected_date,
        "cash_at_risk": cash_at_risk,
        "reasons": reasons,
        "model_version": MODEL_VERSION,
    }
