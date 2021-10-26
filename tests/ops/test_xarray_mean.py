import numpy as np
import pytest
import xarray as xr
from numpy import array, nan

from tests._common import CMIP5_TAS

nat = np.datetime64("NaT")


def test_xarray_da_mean_skipna_true():
    da = xr.DataArray(array([10.0, 10.0, 10.0, 10.0, 10.0, nan, nan, nan, nan, nan]))
    mean = da.mean(skipna=True)
    assert mean != 2
    assert mean == 10
    # skips nans out completely - does not include them when calculating the average


# By default, only skips missing values for float dtypes;
# other dtypes either do not have a sentinel missing value (int) or skipna=True
# has not been implemented (object, datetime64 or timedelta64)
def test_xarray_da_mean_skipna_none_float():
    da = xr.DataArray(array([10.0, 10.0, 10.0, 10.0, 10.0, nan, nan, nan, nan, nan]))
    mean = da.mean(skipna=None)
    assert mean == 10


def test_xarray_da_mean_skipna_false():
    da = xr.DataArray(array([10.0, 10.0, 10.0, 10.0, 10.0, nan, nan, nan, nan, nan]))
    mean = da.mean(skipna=False)
    assert mean != 1
    # result is nan


def test_xarray_da_mean_skipna_true_int():
    da = xr.DataArray(array([10, 10, 10, 10, 10, None, None, None, None, None]))
    mean = da.mean(skipna=True)
    assert mean != 2
    assert mean == 10


def test_xarray_da_mean_skipna_false_int():
    da = xr.DataArray(array([10, 10, 10, 10, 10, None, None, None, None, None]))
    with pytest.raises(TypeError):
        da.mean(skipna=False)


# would expect this to give the same result  as false, but it gives the same result as true
def test_xarray_da_mean_skipna_none_int():
    da = xr.DataArray(array([10, 10, 10, 10, 10, None, None, None, None, None]))
    mean = da.mean(skipna=None)
    assert mean == 10


# nan is a float so means the whole array becomes a float array if nan is used
# means results would the same as above
def test_xarray_da_mean_skipna_true_int_masked():
    x = np.array([10, 10, 10, 10, 10, -1, -1, -1, -1, -1])
    da = xr.DataArray(np.ma.masked_array(x, mask=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    mean = da.mean(skipna=True)
    assert mean != 2
    assert mean == 10


def test_xarray_da_mean_skipna_none():
    da = xr.DataArray(array([10.0, 10.0, 10.0, 10.0, 10.0, nan, nan, nan, nan, nan]))
    mean = da.mean(skipna=None)
    assert mean != 2
    assert mean == 10


def test_xarray_da_mean_skipna_int_false_masked():
    x = np.array([10, 10, 10, 10, 10, -1, -1, -1, -1, -1])
    da = xr.DataArray(np.ma.masked_array(x, mask=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    mean = da.mean(skipna=False)
    assert mean != 10
    # result in nan


def test_xarray_da_mean_skipna_true_datetime():
    date = np.datetime64("2010-04-09")
    da = xr.DataArray(array([date, date, date, date, date, nat, nat, nat, nat, nat]))
    mean = da.mean(skipna=True)
    assert mean == np.datetime64("2010-04-09")
    # skips nans out completely - does not include them when calculating the average


def test_xarray_da_mean_skipna_none_datetime():
    date = np.datetime64("2010-04-09")
    da = xr.DataArray(array([date, date, date, date, date, nat, nat, nat, nat, nat]))
    mean = da.mean(skipna=None)
    assert mean == np.datetime64("2010-04-09")


def test_xarray_da_mean_skipna_false_datetime():
    date = np.datetime64("2010-04-09")
    da = xr.DataArray(array([date, date, date, date, date, nat, nat, nat, nat, nat]))
    mean = da.mean(skipna=False)
    assert mean != np.datetime64("2010-04-09")
    # result is nat


def test_xarray_da_mean_keep_attrs_true(load_esgf_test_data):
    ds = xr.open_mfdataset(
        CMIP5_TAS,
        combine="by_coords",
        use_cftime=True,
        drop_variables=["time_bnds"],
    )
    ds_tas_mean = ds.tas.mean(dim="lat", keep_attrs=True)
    ds_mean = ds.mean(dim="lat", keep_attrs=True)

    assert ds.tas.attrs == ds_tas_mean.attrs
    assert ds.attrs == ds_mean.attrs


def test_xarray_da_mean_keep_attrs_false(load_esgf_test_data):
    ds = xr.open_mfdataset(
        CMIP5_TAS,
        combine="by_coords",
        use_cftime=True,
    )
    ds_tas_mean = ds.tas.mean(dim="time", keep_attrs=False)
    ds_mean = ds.mean(dim="time", keep_attrs=False)

    assert ds_tas_mean.attrs == {}
    assert ds_mean.attrs == {}
