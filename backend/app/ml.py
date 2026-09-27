import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from math import exp
from pathlib import Path

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.entities import Invoice
from backend.app.evaluator import _history_before_invoice
from backend.app.models import BuyerHistory, InvoiceInput
from backend.app.scoring import MODEL_VERSION as BASELINE_MODEL_VERSION
from backend.app.scoring import predict_invoice_risk

ML_MODEL_VERSION = "ml-v0.1"
FEATURE_NAMES = [
    "on_time_rate",
    "late_payment_rate",
    "average_delay_days",
    "median_delay_days",
    "p90_delay_days",
    "recent_average_delay_days",
    "recent_trend_days",
    "payment_consistency",
    "invoice_amount_ratio",
    "outstanding_to_avg_invoice",
]


def _artifact_dir() -> Path:
    configured = Path.cwd() / ".models"
    configured.mkdir(parents=True, exist_ok=True)
    return configured


def _artifact_path(business_id: str, candidate: bool = False) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in business_id)
    suffix = ".candidate.json" if candidate else ".json"
    return _artifact_dir() / f"{safe_id}{suffix}"


def _feature_vector(history: BuyerHistory, amount: float) -> list[float]:
    features = history.model_dump()
    avg_invoice = float(features["average_invoice_amount"])
    amount_ratio = amount / avg_invoice if avg_invoice > 0 else 1.0
    outstanding_ratio = (
        float(features["current_outstanding_amount"]) / avg_invoice
        if avg_invoice > 0
        else 0.0
    )
    return [
        float(features["payment_delays_days"] and (
            sum(delay <= 0 for delay in features["payment_delays_days"])
            / len(features["payment_delays_days"])
        ) or 0.0),
        float(
            features["late_invoice_count"] / max(features["invoice_count"], 1)
        ),
        float(
            sum(features["payment_delays_days"])
            / max(len(features["payment_delays_days"]), 1)
        ),
        float(
            sorted(features["payment_delays_days"])[
                len(features["payment_delays_days"]) // 2
            ]
            if features["payment_delays_days"]
            else 0.0
        ),
        float(_p90(features["payment_delays_days"])),
        float(
            sum(features["recent_delays_days"])
            / max(len(features["recent_delays_days"]), 1)
            if features["recent_delays_days"]
            else 0.0
        ),
        float(
            (
                sum(features["recent_delays_days"])
                / max(len(features["recent_delays_days"]), 1)
            )
            - (
                sum(features["payment_delays_days"])
                / max(len(features["payment_delays_days"]), 1)
            )
        ),
        float(
            max(
                0.0,
                1.0
                - _clamp(
                    _pstdev(features["payment_delays_days"]) / 15.0,
                    0.0,
                    1.0,
                ),
            )
        ),
        float(amount_ratio),
        float(outstanding_ratio),
    ]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pstdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = 0.90 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _settled(invoice: Invoice) -> bool:
    return sum((payment.amount for payment in invoice.payments), Decimal("0")) >= invoice.amount


def _collect_samples(
    session: Session,
    business_id: str,
    min_history: int,
) -> tuple[list[dict], int]:
    invoices = session.scalars(
        select(Invoice)
        .options(selectinload(Invoice.payments))
        .where(Invoice.business_id == business_id)
        .order_by(Invoice.invoice_date, Invoice.id)
    ).all()

    customer_invoices: dict[int, list[Invoice]] = {}
    for invoice in invoices:
        customer_invoices.setdefault(invoice.customer_id, []).append(invoice)

    samples: list[dict] = []
    skipped_cold_start = 0

    for invoice in invoices:
        if not _settled(invoice):
            continue

        history = _history_before_invoice(
            customer_invoices[invoice.customer_id],
            invoice,
        )
        if history.invoice_count < min_history:
            skipped_cold_start += 1
            continue

        actual_payment_date = max(payment.payment_date for payment in invoice.payments)
        actual_delay_days = (actual_payment_date - invoice.due_date).days
        samples.append(
            {
                "invoice": invoice,
                "history": history,
                "features": _feature_vector(history, float(invoice.amount)),
                "actual_delay_days": actual_delay_days,
                "actual_late": actual_delay_days > 0,
            }
        )

    return samples, skipped_cold_start


def _binary_metrics(
    late_probabilities: list[float],
    delays: list[int],
    predicted_delays: list[float],
    actual_late: list[bool],
) -> dict:
    brier = sum(
        (prediction - float(actual)) ** 2
        for prediction, actual in zip(late_probabilities, actual_late)
    ) / len(late_probabilities)
    date_mae = sum(
        abs(float(delay) - predicted)
        for delay, predicted in zip(delays, predicted_delays)
    ) / len(delays)
    correct = sum(
        (prediction >= 0.5) == actual
        for prediction, actual in zip(late_probabilities, actual_late)
    )
    return {
        "date_mae_days": round(date_mae, 2),
        "mean_brier_error": round(brier, 6),
        "late_classification_accuracy": round(correct / len(late_probabilities), 4),
    }


def _better_than_baseline(candidate: dict, baseline: dict) -> bool:
    return (
        candidate["mean_brier_error"] <= baseline["mean_brier_error"]
        and candidate["date_mae_days"] <= baseline["date_mae_days"]
    )


def _serialize_model(
    business_id: str,
    scaler: StandardScaler,
    classifier: LogisticRegression,
    regressor: Ridge,
    metrics: dict,
    train_count: int,
    test_count: int,
    promotion_status: str,
) -> dict:
    return {
        "business_id": business_id,
        "model_version": ML_MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coefficients": classifier.coef_[0].tolist(),
        "classifier_intercept": float(classifier.intercept_[0]),
        "regressor_coefficients": regressor.coef_.tolist(),
        "regressor_intercept": float(regressor.intercept_),
        "train_count": train_count,
        "test_count": test_count,
        "promotion_status": promotion_status,
        "metrics": metrics,
    }


def train_business_model(
    session: Session,
    business_id: str,
    min_history: int = 3,
    test_fraction: float = 0.30,
) -> dict:
    if min_history < 0:
        raise ValueError("min_history must be non-negative")
    if not 0.20 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.20 and 0.50")

    samples, skipped_cold_start = _collect_samples(
        session, business_id, min_history
    )
    if len(samples) < 12:
        raise ValueError(
            f"Need at least 12 eligible settled invoices; found {len(samples)}"
        )

    split_index = int(len(samples) * (1.0 - test_fraction))
    split_index = max(8, min(split_index, len(samples) - 4))
    train_samples = samples[:split_index]
    test_samples = samples[split_index:]

    y_train = [sample["actual_late"] for sample in train_samples]
    if len(set(y_train)) < 2:
        raise ValueError("Training window must contain both late and on-time outcomes")

    scaler = StandardScaler()
    x_train = scaler.fit_transform([sample["features"] for sample in train_samples])
    x_test = scaler.transform([sample["features"] for sample in test_samples])

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(x_train, y_train)

    regressor = Ridge(alpha=10.0)
    regressor.fit(
        x_train,
        [max(0.0, float(sample["actual_delay_days"])) for sample in train_samples],
    )

    ml_probabilities = [
        float(probability[1])
        for probability in classifier.predict_proba(x_test)
    ]
    ml_delays = [
        max(0.0, float(delay))
        for delay in regressor.predict(x_test)
    ]
    actual_delays = [sample["actual_delay_days"] for sample in test_samples]
    actual_late = [sample["actual_late"] for sample in test_samples]

    baseline_probabilities = []
    baseline_delays = []
    for sample in test_samples:
        baseline = predict_invoice_risk(
            InvoiceInput(
                due_date=sample["invoice"].due_date,
                amount=float(sample["invoice"].amount),
                buyer=sample["history"],
            )
        )
        baseline_probabilities.append(float(baseline["late_probability"]))
        baseline_delays.append(float(baseline["expected_delay_days"]))

    ml_metrics = _binary_metrics(
        ml_probabilities,
        actual_delays,
        ml_delays,
        actual_late,
    )
    baseline_metrics = _binary_metrics(
        baseline_probabilities,
        actual_delays,
        baseline_delays,
        actual_late,
    )
    promoted = _better_than_baseline(ml_metrics, baseline_metrics)

    artifact = _serialize_model(
        business_id,
        scaler,
        classifier,
        regressor,
        {
            "candidate": ml_metrics,
            "baseline": baseline_metrics,
            "delta": {
                "date_mae_days": round(
                    ml_metrics["date_mae_days"] - baseline_metrics["date_mae_days"],
                    2,
                ),
                "mean_brier_error": round(
                    ml_metrics["mean_brier_error"] - baseline_metrics["mean_brier_error"],
                    6,
                ),
            },
        },
        len(train_samples),
        len(test_samples),
        "promoted" if promoted else "candidate_only",
    )

    candidate_path = _artifact_path(business_id, candidate=True)
    candidate_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    active_path = _artifact_path(business_id)
    if promoted:
        active_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    return {
        "business_id": business_id,
        "model_version": ML_MODEL_VERSION,
        "baseline_model_version": BASELINE_MODEL_VERSION,
        "promotion_status": artifact["promotion_status"],
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "skipped_cold_start": skipped_cold_start,
        "candidate_metrics": ml_metrics,
        "baseline_metrics": baseline_metrics,
        "delta": artifact["metrics"]["delta"],
    }


def _sigmoid(value: float) -> float:
    value = _clamp(value, -35.0, 35.0)
    return 1.0 / (1.0 + exp(-value))


def _load_active_model(business_id: str) -> dict | None:
    path = _artifact_path(business_id)
    if not path.exists():
        return None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if artifact.get("promotion_status") != "promoted":
        return None
    return artifact


def active_model_status(business_id: str) -> dict:
    active = _load_active_model(business_id)
    if active:
        return {
            "business_id": business_id,
            "active": True,
            "model_version": active["model_version"],
            "trained_at": active["trained_at"],
            "train_count": active["train_count"],
            "test_count": active["test_count"],
            "promotion_status": active["promotion_status"],
            "metrics": active["metrics"],
        }

    candidate_path = _artifact_path(business_id, candidate=True)
    if candidate_path.exists():
        try:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            candidate = None
        if candidate:
            return {
                "business_id": business_id,
                "active": False,
                "model_version": BASELINE_MODEL_VERSION,
                "candidate_model_version": candidate.get("model_version"),
                "promotion_status": candidate.get("promotion_status", "candidate_only"),
                "trained_at": candidate.get("trained_at"),
                "train_count": candidate.get("train_count"),
                "test_count": candidate.get("test_count"),
                "metrics": candidate.get("metrics"),
            }

    return {
        "business_id": business_id,
        "active": False,
        "model_version": BASELINE_MODEL_VERSION,
        "promotion_status": "baseline_only",
    }


def predict_invoice_risk_with_active_model(
    session: Session,
    invoice_id: int,
) -> dict:
    invoice = session.scalar(
        select(Invoice)
        .options(selectinload(Invoice.payments), selectinload(Invoice.customer))
        .where(Invoice.id == invoice_id)
    )
    if not invoice:
        raise ValueError("invoice not found")

    # For live inference, use the complete settled buyer history available
    # at prediction time instead of the historical backtest cutoff.
    from backend.app.repository import buyer_history

    history = buyer_history(session, invoice.customer_id)
    baseline_input = InvoiceInput(
        due_date=invoice.due_date,
        amount=float(invoice.amount),
        buyer=history,
    )
    baseline = predict_invoice_risk(baseline_input)
    artifact = _load_active_model(
        session.scalar(select(Invoice.business_id).where(Invoice.id == invoice_id))
        or ""
    )
    if not artifact:
        return baseline

    vector = _feature_vector(history, float(invoice.amount))
    means = artifact["scaler_mean"]
    scales = [scale if abs(scale) > 1e-12 else 1.0 for scale in artifact["scaler_scale"]]
    scaled = [(value - mean) / scale for value, mean, scale in zip(vector, means, scales)]

    classifier_score = artifact["classifier_intercept"] + sum(
        coefficient * value
        for coefficient, value in zip(artifact["classifier_coefficients"], scaled)
    )
    late_probability = round(_clamp(_sigmoid(classifier_score), 0.01, 0.99), 4)

    predicted_delay = max(
        0.0,
        artifact["regressor_intercept"]
        + sum(
            coefficient * value
            for coefficient, value in zip(
                artifact["regressor_coefficients"],
                scaled,
            )
        ),
    )
    expected_delay = round(predicted_delay, 1)
    expected_date = invoice.due_date + timedelta(days=round(expected_delay))
    cash_at_risk = round(float(invoice.amount) * late_probability, 2)

    reasons = list(baseline["reasons"])
    reasons.append(
        f"ML model trained on {artifact['train_count']} historical invoices"
    )

    return {
        "late_probability": late_probability,
        "expected_delay_days": expected_delay,
        "expected_payment_date": expected_date,
        "cash_at_risk": cash_at_risk,
        "reasons": reasons,
        "model_version": artifact["model_version"],
    }
