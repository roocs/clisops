import numpy as np
import xarray as xr

from .._common import CMIP5_TAS

nan = np.nan


def test_xarray_da_mean_skipna_true():
    da = xr.DataArray(np.array([10.0, 10.0, 10.0, 10.0, 10.0, nan, nan, nan, nan, nan]))
    mean = da.mean(skipna=True)
    assert mean != 2
    assert mean == 10
    # skips nans out completely - does not include them when calculating the average


def test_xarray_da_mean_skipna_false():
    da = xr.DataArray(np.array([10.0, 10.0, 10.0, 10.0, 10.0, nan, nan, nan, nan, nan]))
    mean = da.mean(skipna=False)
    assert mean != 1
    # result is nan


def test_xarray_da_mean_skipna_none():
    da = xr.DataArray(np.array([10.0, 10.0, 10.0, 10.0, 10.0, nan, nan, nan, nan, nan]))
    mean = da.mean(skipna=None)
    print(mean)
    assert mean == 10


def test_xarray_da_mean_keep_attrs_true():
    ds = xr.open_mfdataset(
        CMIP5_TAS,
        combine="by_coords",
        use_cftime=True,
    )
    ds_tas_mean = ds.tas.mean(dim="lat", keep_attrs=True)
    ds_mean = ds.mean(dim="lat", keep_attrs=True)

    assert ds.tas.attrs == ds_tas_mean.attrs
    assert ds.attrs == ds_mean.attrs


def test_xarray_da_mean_keep_attrs_false():
    ds = xr.open_mfdataset(
        CMIP5_TAS,
        combine="by_coords",
        use_cftime=True,
    )
    ds_tas_mean = ds.tas.mean(dim="time", keep_attrs=False)
    ds_mean = ds.mean(dim="time", keep_attrs=False)

    assert ds_tas_mean.attrs == {}
    assert ds_mean.attrs == {}
