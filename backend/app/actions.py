from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.entities import CollectionAction, Invoice, RiskPrediction
from backend.app.ml import predict_invoice_risk_with_active_model


ACTION_VERSION = "action-v0.1"


def _days_until_due(due_date: date, as_of: date) -> int:
    return (due_date - as_of).days


def _action_plan(
    *,
    late_probability: float,
    expected_delay_days: float,
    due_date: date,
    amount: Decimal,
    as_of: date,
) -> dict:
    days_overdue = max(0, (as_of - due_date).days)
    days_until_due = _days_until_due(due_date, as_of)

    urgency = min(days_overdue / 30.0, 1.0)
    risk_component = late_probability
    delay_component = min(max(expected_delay_days, 0.0) / 30.0, 1.0)

    priority_score = round(
        100.0
        * (
            0.55 * risk_component
            + 0.30 * urgency
            + 0.15 * delay_component
        ),
        2,
    )

    if days_overdue >= 15 and late_probability >= 0.70:
        action_type = "ESCALATION_REVIEW"
        next_step = "Escalate to the account owner for a collections review."
    elif days_overdue >= 1 and late_probability >= 0.60:
        action_type = "PRIORITY_COLLECTION"
        next_step = "Contact the buyer and confirm a committed payment date."
    elif days_overdue >= 1:
        action_type = "COLLECTION_FOLLOW_UP"
        next_step = "Follow up on the overdue balance and record the buyer's response."
    elif late_probability >= 0.70:
        action_type = "PRE_DUE_PRIORITY"
        next_step = "Contact the buyer before the due date and confirm payment readiness."
    elif days_until_due <= 3 or late_probability >= 0.40:
        action_type = "PRE_DUE_REMINDER"
        next_step = "Send a routine payment reminder and confirm invoice details."
    else:
        action_type = "MONITOR"
        next_step = "Monitor the invoice; no immediate collection action is recommended."

    reasons = [
        f"{late_probability:.0%} modeled late-payment probability",
        f"{expected_delay_days:.1f} days expected delay",
    ]
    if days_overdue:
        reasons.append(f"{days_overdue} days overdue")
    elif days_until_due <= 3:
        reasons.append(f"due in {days_until_due} days")
    if amount > Decimal("0"):
        reasons.append(f"₹{amount:,.0f} invoice exposure")

    return {
        "action_type": action_type,
        "priority_score": priority_score,
        "days_overdue": days_overdue,
        "next_step": next_step,
        "reason": "; ".join(reasons),
    }


def _latest_prediction(
    session: Session,
    invoice_id: int,
) -> RiskPrediction | None:
    return session.scalar(
        select(RiskPrediction)
        .where(RiskPrediction.invoice_id == invoice_id)
        .order_by(RiskPrediction.predicted_at.desc(), RiskPrediction.id.desc())
    )


def generate_collection_actions(
    session: Session,
    business_id: str,
    as_of: date | None = None,
) -> dict:
    as_of = as_of or date.today()
    invoices = session.scalars(
        select(Invoice)
        .where(Invoice.business_id == business_id)
        .order_by(Invoice.due_date, Invoice.id)
    ).all()

    created = 0
    updated = 0
    skipped_settled = 0
    skipped_unscored = 0

    for invoice in invoices:
        paid = sum((payment.amount for payment in invoice.payments), Decimal("0"))
        if paid >= invoice.amount:
            skipped_settled += 1
            continue

        prediction = _latest_prediction(session, invoice.id)
        if not prediction:
            skipped_unscored += 1
            continue

        plan = _action_plan(
            late_probability=float(prediction.late_probability),
            expected_delay_days=float(prediction.expected_delay_days),
            due_date=invoice.due_date,
            amount=invoice.amount,
            as_of=as_of,
        )

        action = session.scalar(
            select(CollectionAction)
            .where(
                CollectionAction.invoice_id == invoice.id,
                CollectionAction.prediction_id == prediction.id,
            )
        )

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if action:
            action.model_version = prediction.model_version
            action.action_type = plan["action_type"]
            action.priority_score = plan["priority_score"]
            action.due_date = invoice.due_date
            action.days_overdue = plan["days_overdue"]
            action.amount = invoice.amount
            action.reason = plan["next_step"] + " " + plan["reason"]
            action.updated_at = now
            if action.status == "open":
                updated += 1
            continue

        session.add(
            CollectionAction(
                business_id=business_id,
                invoice_id=invoice.id,
                prediction_id=prediction.id,
                model_version=prediction.model_version,
                action_type=plan["action_type"],
                priority_score=plan["priority_score"],
                status="open",
                due_date=invoice.due_date,
                days_overdue=plan["days_overdue"],
                amount=invoice.amount,
                reason=plan["next_step"] + " " + plan["reason"],
                created_at=now,
                updated_at=now,
            )
        )
        created += 1

    session.commit()

    return {
        "business_id": business_id,
        "action_version": ACTION_VERSION,
        "as_of": as_of,
        "invoices_seen": len(invoices),
        "actions_created": created,
        "actions_updated": updated,
        "settled_skipped": skipped_settled,
        "unscored_skipped": skipped_unscored,
    }


def list_collection_actions(
    session: Session,
    business_id: str,
    status: str = "open",
    limit: int = 100,
) -> list[dict]:
    rows = session.execute(
        select(CollectionAction, Invoice, RiskPrediction)
        .join(Invoice, CollectionAction.invoice_id == Invoice.id)
        .outerjoin(
            RiskPrediction,
            CollectionAction.prediction_id == RiskPrediction.id,
        )
        .where(
            CollectionAction.business_id == business_id,
            CollectionAction.status == status,
        )
        .order_by(
            CollectionAction.priority_score.desc(),
            CollectionAction.days_overdue.desc(),
            CollectionAction.id.desc(),
        )
        .limit(limit)
    ).all()

    return [
        {
            "action_id": action.id,
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "action_type": action.action_type,
            "priority_score": float(action.priority_score),
            "status": action.status,
            "due_date": action.due_date,
            "days_overdue": action.days_overdue,
            "amount": float(action.amount),
            "late_probability": (
                float(prediction.late_probability) if prediction else None
            ),
            "expected_delay_days": (
                float(prediction.expected_delay_days) if prediction else None
            ),
            "model_version": action.model_version,
            "reason": action.reason,
            "created_at": action.created_at,
            "updated_at": action.updated_at,
        }
        for action, invoice, prediction in rows
    ]


def update_collection_action_status(
    session: Session,
    action_id: int,
    status: str,
) -> dict:
    allowed = {"open", "completed", "dismissed"}
    if status not in allowed:
        raise ValueError(f"status must be one of: {', '.join(sorted(allowed))}")

    action = session.get(CollectionAction, action_id)
    if not action:
        raise ValueError("collection action not found")

    action.status = status
    action.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.commit()

    return {
        "action_id": action.id,
        "status": action.status,
        "updated_at": action.updated_at,
    }


def action_summary(
    session: Session,
    business_id: str,
) -> dict:
    rows = session.scalars(
        select(CollectionAction)
        .where(
            CollectionAction.business_id == business_id,
            CollectionAction.status == "open",
        )
        .order_by(CollectionAction.priority_score.desc())
    ).all()

    return {
        "business_id": business_id,
        "open_actions": len(rows),
        "priority_actions": sum(
            action.action_type in {"PRIORITY_COLLECTION", "ESCALATION_REVIEW"}
            for action in rows
        ),
        "overdue_actions": sum(action.days_overdue > 0 for action in rows),
        "cash_in_action_queue": round(
            sum((action.amount for action in rows), Decimal("0")),
            2,
        ),
        "next_action": (
            rows[0].action_type if rows else None
        ),
    }
