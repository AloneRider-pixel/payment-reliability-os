from statistics import median, pstdev

from backend.app.models import BuyerHistory


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = 0.90 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def extract_buyer_features(history: BuyerHistory) -> dict:
    delays = [float(delay) for delay in history.payment_delays_days]
    recent = [float(delay) for delay in (history.recent_delays_days or delays[-5:])]
    evidence_count = len(delays)

    if not delays:
        return {
            "on_time_rate": 0.0,
            "late_payment_rate": 0.0,
            "average_delay_days": 0.0,
            "median_delay_days": 0.0,
            "p90_delay_days": 0.0,
            "recent_average_delay_days": 0.0,
            "recent_trend_days": 0.0,
            "payment_consistency": 0.0,
            "invoice_amount_mean": round(float(history.average_invoice_amount), 2),
            "current_outstanding_exposure": round(float(history.current_outstanding_amount), 2),
            "evidence_count": 0,
        }

    on_time_count = sum(delay <= 0 for delay in delays)
    late_count = sum(delay > 0 for delay in delays)
    average_delay = sum(delays) / len(delays)
    median_delay = median(delays)
    p90_delay = _p90(delays)
    recent_average = sum(recent) / len(recent) if recent else average_delay
    trend_days = recent_average - average_delay
    variability = pstdev(delays) if len(delays) > 1 else 0.0

    # A 15-day standard deviation is treated as the boundary between
    # consistent and highly variable payment behavior for this baseline.
    payment_consistency = 1.0 - _clamp(variability / 15.0, 0.0, 1.0)

    return {
        "on_time_rate": round(on_time_count / len(delays), 4),
        "late_payment_rate": round(late_count / len(delays), 4),
        "average_delay_days": round(average_delay, 2),
        "median_delay_days": round(float(median_delay), 2),
        "p90_delay_days": round(p90_delay, 2),
        "recent_average_delay_days": round(recent_average, 2),
        "recent_trend_days": round(trend_days, 2),
        "payment_consistency": round(payment_consistency, 4),
        "invoice_amount_mean": round(float(history.average_invoice_amount), 2),
        "current_outstanding_exposure": round(float(history.current_outstanding_amount), 2),
        "evidence_count": evidence_count,
    }
