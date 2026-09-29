from unittest.mock import Mock

import pytest
from sqlalchemy.exc import IntegrityError

from backend.app.scheduler import _claim


def test_claim_returns_none_when_unique_slot_is_already_claimed():
    session = Mock()
    session.scalar.return_value = None
    session.commit.side_effect = IntegrityError("INSERT", {}, Exception("duplicate"))

    result = _claim(
        session,
        business_id="biz-1",
        job_type="refresh_risk",
        scheduled_for=__import__("datetime").datetime(2026, 9, 29, 2, 0),
    )

    assert result is None
    session.rollback.assert_called_once()


def test_claim_does_not_hide_unexpected_database_errors():
    session = Mock()
    session.scalar.return_value = None
    failure = RuntimeError("database unavailable")
    session.commit.side_effect = failure

    with pytest.raises(RuntimeError, match="database unavailable"):
        _claim(
            session,
            business_id="biz-1",
            job_type="refresh_risk",
            scheduled_for=__import__("datetime").datetime(2026, 9, 29, 2, 0),
        )

    session.rollback.assert_not_called()
