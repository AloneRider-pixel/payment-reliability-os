from datetime import timedelta

from backend.app.models import BuyerHistory, InvoiceInput

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    if score >= 50: return "E"
    return "F"

def score_buyer(history: BuyerHistory) -> dict:
    delays = history.payment_delays_days
    recent = history.recent_delays_days or delays[-5:]
    evidence_count = len(delays)

    if evidence_count == 0:
        return {
            "score": 50.0,
            "grade": "N/A",
            "late_probability": 0.50,
            "expected_delay_days": 0.0,
            "reasons": ["No settled payment history is available"],
            "evidence_count": 0,
            "confidence": "LOW",
        }

    count = max(history.invoice_count, evidence_count, 1)
    late_rate = history.late_invoice_count / count
    if history.late_invoice_count == 0 and delays:
        late_rate = sum(d > 0 for d in delays) / len(delays)

    avg_delay = sum(delays) / len(delays)
    recent_avg = sum(recent) / len(recent) if recent else avg_delay
    delay_penalty = _clamp(avg_delay / 30.0, 0, 1)
    trend_penalty = _clamp((recent_avg - avg_delay) / 15.0, -1, 1)

    score = 100.0 - late_rate * 45 - delay_penalty * 35 - max(trend_penalty, 0) * 15
    score = round(_clamp(score, 0, 100), 1)

    late_probability = _clamp(
        late_rate * 0.65 + delay_penalty * 0.25 + max(trend_penalty, 0) * 0.10,
        0.01,
        0.99,
    )

    confidence = "HIGH" if evidence_count >= 12 else "MEDIUM" if evidence_count >= 5 else "LOW"
    reasons = []
    if late_rate >= 0.5:
        reasons.append(f"High historical late-payment rate ({late_rate:.0%})")
    elif late_rate > 0:
        reasons.append(f"Historical late-payment rate ({late_rate:.0%})")
    else:
        reasons.append("No historical late payments in the supplied history")
    if avg_delay > 0:
        reasons.append(f"Average payment delay is {avg_delay:.1f} days")
    if recent_avg > avg_delay + 2:
        reasons.append("Recent payment behavior is deteriorating")
    elif recent_avg < avg_delay - 2:
        reasons.append("Recent payment behavior is improving")
    reasons.append(f"Evidence: {evidence_count} settled invoices")

    return {
        "score": score,
        "grade": _grade(score),
        "late_probability": round(late_probability, 4),
        "expected_delay_days": round(max(avg_delay, 0), 1),
        "reasons": reasons,
        "evidence_count": evidence_count,
        "confidence": confidence,
    }

def predict_invoice_risk(payload: InvoiceInput) -> dict:
    buyer_result = score_buyer(payload.buyer)
    amount_ratio = (
        payload.amount / payload.buyer.average_invoice_amount
        if payload.buyer.average_invoice_amount > 0
        else 1.0
    )
    anomaly_penalty = _clamp(max(amount_ratio - 1.0, 0) / 2.0, 0, 1)
    late_probability = _clamp(
        buyer_result["late_probability"] * 0.85 + anomaly_penalty * 0.15,
        0.01,
        0.99,
    )
    expected_delay = round(buyer_result["expected_delay_days"] * (1.0 + 0.25 * anomaly_penalty), 1)
    expected_date = payload.due_date + timedelta(days=round(expected_delay))
    cash_at_risk = round(payload.amount * late_probability, 2)
    reasons = list(buyer_result["reasons"])
    if amount_ratio >= 1.5 and payload.buyer.average_invoice_amount > 0:
        reasons.append(f"Current invoice is {amount_ratio:.1f}x the buyer's average invoice amount")
    return {
        "late_probability": round(late_probability, 4),
        "expected_delay_days": expected_delay,
        "expected_payment_date": expected_date,
        "cash_at_risk": cash_at_risk,
        "reasons": reasons,
    }