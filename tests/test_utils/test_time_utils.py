import pytest

from clisops.utils.time_utils import AnyCalendarDateTime, str_to_AnyCalendarDateTime


class TestAnyCalendarDateTime:

    def test_simple(self):
        d = AnyCalendarDateTime(1997, 4, 9, 0, 0, 0)
        assert d.value == "1997-04-09T00:00:00"

    def test_str_to_calendar(self):
        d = str_to_AnyCalendarDateTime("1999-01-31")
        d.sub_day()
        assert d.value.startswith("1999-01-30")

        d.add_day()
        d.add_day()
        assert d.value.startswith("1999-02-01")

        d = str_to_AnyCalendarDateTime("1999-02-31")
        d.sub_day()
        assert d.value.startswith("1999-02-30")

        d = str_to_AnyCalendarDateTime("1999-01-01T12:13:14")
        d.sub_day()
        assert d.value == "1998-12-31T12:13:14"

    def test_irregular_datetimes(self):
        # looks like no year
        d = str_to_AnyCalendarDateTime("01-02")
        # first value is taken as the year
        assert d.value == "1-02-01T00:00:00"

        # month is 00
        with pytest.raises(ValueError) as exc:
            str_to_AnyCalendarDateTime("1999-00-33")
        assert (
            str(exc.value)
            == "Invalid input 0 for month. Expected value between 1 and 12."
        )

        # month is 13
        with pytest.raises(ValueError) as exc:
            str_to_AnyCalendarDateTime("1999-13-22")
        assert (
            str(exc.value)
            == "Invalid input 13 for month. Expected value between 1 and 12."
        )

        # hour is 27
        with pytest.raises(ValueError) as exc:
            str_to_AnyCalendarDateTime("1999-01-22T27:00:00")
        assert (
            str(exc.value)
            == "Invalid input 27 for hour. Expected value between 0 and 23."
        )
