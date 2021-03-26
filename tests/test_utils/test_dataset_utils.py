import pytest
import xarray as xr

from clisops.utils.dataset_utils import adjust_date_to_calendar

from .._common import CMIP6_SICONC


def test_add_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-29T00:00:00"

    new_date = adjust_date_to_calendar(da, date, "forwards")

    assert new_date == "2012-03-01T00:00:00"


def test_sub_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-30T00:00:00"

    new_date = adjust_date_to_calendar(da, date, "backwards")

    assert new_date == "2012-02-28T00:00:00"


def test_invalid_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-29T00:00:00"

    with pytest.raises(Exception) as exc:
        adjust_date_to_calendar(da, date, "odd")
    assert (
        str(exc.value)
        == "Invalid value for direction: odd. This should be either 'backwards' to indicate subtracting a day or 'forwards' for adding a day."
    )


def test_date_out_of_expected_range():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-00-01T00:00:00"

    with pytest.raises(Exception) as exc:
        adjust_date_to_calendar(da, date, "forwards")
    assert (
        str(exc.value) == "Invalid input 0 for month. Expected value between 1 and 12."
    )
