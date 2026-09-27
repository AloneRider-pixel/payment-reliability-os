import argparse
import json
from datetime import date, datetime

from backend.app import db
from backend.app.scheduler import daily_slot, run_scheduled_jobs


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Payment Reliability OS scheduled jobs.")
    parser.add_argument("--business-id", default=None)
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD operational date")
    parser.add_argument(
        "--scheduled-for",
        default=None,
        help="ISO datetime used as the idempotency slot; defaults to the daily 02:00 UTC slot",
    )
    args = parser.parse_args()

    db.init_db()
    session = db.SessionLocal()
    try:
        as_of = _parse_date(args.as_of) or date.today()
        scheduled_for = _parse_datetime(args.scheduled_for) or daily_slot(as_of)
        result = run_scheduled_jobs(
            session,
            scheduled_for=scheduled_for,
            as_of=as_of,
            business_id=args.business_id,
        )
        print(json.dumps(result, default=str, indent=2))
        return 1 if result["failed"] else 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
