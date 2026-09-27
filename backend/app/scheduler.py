import json
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.actions import generate_collection_actions
from backend.app.entities import (
    Business,
    Invoice,
    PredictionEvaluation,
    RiskPrediction,
    ScheduledJobRun,
)
from backend.app.evaluator import evaluate_prediction
from backend.app.ml import retrain_if_needed, predict_invoice_risk_with_active_model

SCHEDULER_VERSION = "scheduler-v0.1"
JOB_EVALUATE = "evaluate_predictions"
JOB_MAINTAIN_MODEL = "maintain_model"
JOB_REFRESH_RISK = "refresh_risk"
JOB_REFRESH_ACTIONS = "refresh_actions"
JOB_TYPES = (
    JOB_EVALUATE,
    JOB_MAINTAIN_MODEL,
    JOB_REFRESH_RISK,
    JOB_REFRESH_ACTIONS,
)


def daily_slot(as_of: date | None = None) -> datetime:
    day = as_of or date.today()
    return datetime(day.year, day.month, day.day, 2, 0, 0)


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _claim(
    session: Session,
    business_id: str,
    job_type: str,
    scheduled_for: datetime,
) -> ScheduledJobRun | None:
    existing = session.scalar(
        select(ScheduledJobRun).where(
            ScheduledJobRun.business_id == business_id,
            ScheduledJobRun.job_type == job_type,
            ScheduledJobRun.scheduled_for == scheduled_for,
        )
    )
    if existing:
        return None

    run = ScheduledJobRun(
        business_id=business_id,
        job_type=job_type,
        scheduled_for=scheduled_for,
        started_at=_utc_now_naive(),
        status="running",
        result="",
    )
    session.add(run)
    try:
        session.commit()
    except Exception:
        session.rollback()
        return None
    return run


def _finish(
    session: Session,
    run: ScheduledJobRun,
    status: str,
    result: dict,
) -> None:
    run.status = status
    run.finished_at = _utc_now_naive()
    run.result = json.dumps(result, default=str)
    session.commit()


def _evaluate_pending_predictions(session: Session, business_id: str) -> dict:
    rows = session.execute(
        select(RiskPrediction, PredictionEvaluation)
        .join(
            Invoice,
            RiskPrediction.invoice_id == Invoice.id,
        )
        .outerjoin(
            PredictionEvaluation,
            PredictionEvaluation.prediction_id == RiskPrediction.id,
        )
        .where(
            Invoice.business_id == business_id,
            PredictionEvaluation.id.is_(None),
        )
        .order_by(RiskPrediction.predicted_at)
    ).all()

    evaluated = 0
    skipped = 0
    for prediction, _ in rows:
        try:
            evaluate_prediction(session, prediction.id)
            evaluated += 1
        except ValueError:
            session.rollback()
            skipped += 1

    return {
        "predictions_seen": len(rows),
        "evaluated": evaluated,
        "skipped": skipped,
    }


def _refresh_risk(session: Session, business_id: str) -> dict:
    now = _utc_now_naive()
    invoices = session.scalars(
        select(Invoice)
        .options(selectinload(Invoice.payments))
        .where(Invoice.business_id == business_id)
        .order_by(Invoice.id)
    ).all()

    created = 0
    settled = 0
    recent = 0

    for invoice in invoices:
        paid = sum(
            (payment.amount for payment in invoice.payments),
            Decimal("0"),
        )
        if paid >= invoice.amount:
            settled += 1
            continue

        latest = session.scalar(
            select(RiskPrediction)
            .where(RiskPrediction.invoice_id == invoice.id)
            .order_by(RiskPrediction.predicted_at.desc(), RiskPrediction.id.desc())
        )
        if latest and latest.predicted_at >= now.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ):
            recent += 1
            continue

        result = predict_invoice_risk_with_active_model(session, invoice.id)
        session.add(
            RiskPrediction(
                invoice_id=invoice.id,
                predicted_at=now,
                late_probability=result["late_probability"],
                expected_delay_days=result["expected_delay_days"],
                expected_payment_date=result["expected_payment_date"],
                cash_at_risk=Decimal(str(result["cash_at_risk"])),
                model_version=result["model_version"],
                reasons=json.dumps(result["reasons"]),
            )
        )
        created += 1

    session.commit()
    return {
        "invoices_seen": len(invoices),
        "predictions_created": created,
        "settled_skipped": settled,
        "already_refreshed": recent,
    }


def _run_job(
    session: Session,
    business_id: str,
    job_type: str,
    scheduled_for: datetime,
    as_of: date,
) -> dict:
    run = _claim(session, business_id, job_type, scheduled_for)
    if run is None:
        return {
            "business_id": business_id,
            "job_type": job_type,
            "scheduled_for": scheduled_for,
            "status": "already_completed_or_claimed",
        }

    try:
        if job_type == JOB_EVALUATE:
            result = _evaluate_pending_predictions(session, business_id)
        elif job_type == JOB_MAINTAIN_MODEL:
            result = retrain_if_needed(session, business_id)
        elif job_type == JOB_REFRESH_RISK:
            result = _refresh_risk(session, business_id)
        elif job_type == JOB_REFRESH_ACTIONS:
            result = generate_collection_actions(
                session,
                business_id,
                as_of=as_of,
            )
        else:
            raise ValueError(f"unsupported scheduled job: {job_type}")

        _finish(session, run, "completed", result)
        return {
            "business_id": business_id,
            "job_type": job_type,
            "scheduled_for": scheduled_for,
            "status": "completed",
            "result": result,
        }
    except Exception as exc:
        session.rollback()
        fresh = session.get(ScheduledJobRun, run.id)
        if fresh:
            _finish(
                session,
                fresh,
                "failed",
                {"error": str(exc)},
            )
        return {
            "business_id": business_id,
            "job_type": job_type,
            "scheduled_for": scheduled_for,
            "status": "failed",
            "error": str(exc),
        }


def run_scheduled_jobs(
    session: Session,
    scheduled_for: datetime | None = None,
    as_of: date | None = None,
    business_id: str | None = None,
) -> dict:
    as_of = as_of or date.today()
    scheduled_for = scheduled_for or daily_slot(as_of)

    if scheduled_for.tzinfo is not None:
        scheduled_for = scheduled_for.astimezone(timezone.utc).replace(tzinfo=None)

    businesses = session.scalars(
        select(Business)
        .where(Business.id == business_id)
        .order_by(Business.id)
        if business_id
        else select(Business).order_by(Business.id)
    ).all()

    runs = []
    for business in businesses:
        for job_type in JOB_TYPES:
            runs.append(
                _run_job(
                    session,
                    business.id,
                    job_type,
                    scheduled_for,
                    as_of,
                )
            )

    return {
        "scheduler_version": SCHEDULER_VERSION,
        "scheduled_for": scheduled_for,
        "as_of": as_of,
        "businesses_seen": len(businesses),
        "jobs": runs,
        "completed": sum(item["status"] == "completed" for item in runs),
        "failed": sum(item["status"] == "failed" for item in runs),
        "idempotent_skips": sum(
            item["status"] == "already_completed_or_claimed"
            for item in runs
        ),
    }


def recent_job_runs(
    session: Session,
    business_id: str,
    limit: int = 20,
) -> list[dict]:
    rows = session.scalars(
        select(ScheduledJobRun)
        .where(ScheduledJobRun.business_id == business_id)
        .order_by(
            ScheduledJobRun.scheduled_for.desc(),
            ScheduledJobRun.id.desc(),
        )
        .limit(limit)
    ).all()

    return [
        {
            "job_run_id": row.id,
            "job_type": row.job_type,
            "scheduled_for": row.scheduled_for,
            "started_at": row.started_at,
            "finished_at": row.finished_at,
            "status": row.status,
            "result": json.loads(row.result) if row.result else {},
        }
        for row in rows
    ]
