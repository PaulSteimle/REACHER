"""Tests for the human-readable duration formatter."""
from REACHER.utils.formatting import format_duration


def test_format_duration_seconds():
    assert format_duration(0) == "0.00s"
    assert format_duration(0.05) == "0.05s"
    assert format_duration(5.83) == "5.83s"
    assert format_duration(59.99) == "59.99s"


def test_format_duration_minutes():
    assert format_duration(60) == "1m 0.0s"
    assert format_duration(65) == "1m 5.0s"
    assert format_duration(125.5) == "2m 5.5s"
    assert format_duration(3599) == "59m 59.0s"


def test_format_duration_hours():
    assert format_duration(3600) == "1h 0m 0s"
    assert format_duration(3661) == "1h 1m 1s"
    assert format_duration(7322) == "2h 2m 2s"


def test_format_duration_rollover_from_rounding():
    # 7199.6s = 1h 59m 59.6s. round(59.6)=60 -> minute rollover ->
    # 60 minutes -> hour rollover -> "2h 0m 0s".
    assert format_duration(7199.6) == "2h 0m 0s"


def test_format_duration_negative_input_treated_as_zero():
    assert format_duration(-5) == "0.00s"
