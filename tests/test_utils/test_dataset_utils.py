import pytest
import xarray as xr

from clisops.utils.dataset_utils import check_date_exists_in_calendar

from .._common import CMIP6_SICONC


def test_add_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-29T00:00:00"

    new_date = check_date_exists_in_calendar(da, date, "add")

    assert new_date == "2012-03-01T00:00:00"


def test_sub_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-30T00:00:00"

    new_date = check_date_exists_in_calendar(da, date, "sub")

    assert new_date == "2012-02-28T00:00:00"


def test_invalid_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-29T00:00:00"

    with pytest.raises(Exception) as exc:
        check_date_exists_in_calendar(da, date, "odd")
    assert (
        str(exc.value)
        == "Invalid value for day: odd. This should be either 'sub' to indicate subtracting a day or 'add' for adding a day."
    )


def test_could_not_find_date():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-00-01T00:00:00"

    with pytest.raises(Exception) as exc:
        check_date_exists_in_calendar(da, date, "add")
    assert (
        str(exc.value)
        == "Could not find an existing date near 2012-00-01T00:00:00 in the calendar of the xarray object: noleap"
    )
