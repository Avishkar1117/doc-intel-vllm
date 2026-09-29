"""Date matching for SROIE scoring (D-038): a time appended to a date must not make a
correct calendar date score as wrong, but a genuinely different date still must."""

import pytest

from docintel.eval.sroie import dates_match


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("15/01/2019 11:05:16 AM", "15/01/2019"),
        ("07 Mar 2018 18:22", "07 MAR 2018"),
        ("2019-01-15", "15/01/2019"),
        # gold is stripped too: the same parser serves both sides
        ("15/01/2019", "15/01/2019 11:05"),
    ],
)
def test_same_calendar_date_matches(predicted: str, gold: str) -> None:
    assert dates_match(predicted, gold)


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("26-03-19", "26-03-18"),
        ("28/04/2017 10:00", "26/04/2017"),
        # only a trailing clock time is removed, not arbitrary trailing text
        ("15/01/2019 garbage", "15/01/2019"),
    ],
)
def test_different_or_unparseable_date_does_not_match(predicted: str, gold: str) -> None:
    assert not dates_match(predicted, gold)


def test_none_handling() -> None:
    assert dates_match(None, None)
    assert not dates_match(None, "15/01/2019")
    assert not dates_match("15/01/2019", None)
