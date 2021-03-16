import os
from math import isclose

import numpy as np
import pytest
import xarray as xr
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_type

from clisops.ops.subset import subset

from ._common import CMIP6_RLDS_ONE_TIME_STEP


def open_dataset():
    # use real dataset to get full longitude data
    return xr.open_dataset(CMIP6_RLDS_ONE_TIME_STEP)


def setup_test():
    ds = open_dataset()

    # gets longitude by the correct name as used in the dataset
    lon = get_coord_by_type(ds, "longitude")

    return ds, lon


def calculate_offset(x):
    # get resolution of data
    ds, lon = setup_test()
    res = lon.values[1] - lon.values[0]

    # work out how many to roll by to roll data by 1
    index = 1 / res

    # calculate the corresponding offset needed to change data by x
    offset = int(x * index)

    return offset


def test_roll_lon_minus_180(load_esgf_test_data):
    # test rolling longitude by -180
    ds, lon = setup_test()

    # check longitude is 0 to 360 initially
    assert isclose(lon.values.min(), 0, abs_tol=10 ** 2)
    assert isclose(lon.values.max(), 360, abs_tol=10 ** 2)

    # roll longitude by -180
    ds = ds.roll(shifts={f"{lon.name}": -180}, roll_coords=True)

    # doesn't roll as far as we want as it hasn't taken the resolution of the grid into account
    assert ds.lon.values[0] == 90.0
    assert ds.lon.values[-1] == 87.5

    # min and max of the data are still the same
    assert ds.lon.values.min() == 0
    assert ds.lon.values.max() == 357.5


def test_roll_lon_minus_180_use_res(load_esgf_test_data):
    # test rolling longitude by -180
    ds, lon = setup_test()

    # work out how much to roll by
    offset = calculate_offset(-180)

    # roll longitude by calculated offset
    ds_roll = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)

    # longitude data array is rolled to [180.0..275.0..357.5,0..75..177.5]
    assert ds_roll.lon.values[0] == 180.0
    assert ds_roll.lon.values[-1] == 177.5

    # min and max still the same
    assert ds_roll.lon.values.min() == 0
    assert ds_roll.lon.values.max() == 357.5

    # rlds values are not equal - they have been rolled
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        ds_roll.rlds.values,
        ds.rlds.values,
    )


def test_roll_lon_plus_180(load_esgf_test_data):
    # test rolling longitude by 180
    ds, lon = setup_test()

    ds = ds.roll(shifts={f"{lon.name}": 180}, roll_coords=True)

    # doesn't roll as far as we want as it hasn't taken the resolution of the grid into account
    assert ds.lon.values[0] == 270.0
    assert ds.lon.values[-1] == 267.5

    assert ds.lon.values.min() == 0
    assert ds.lon.values.max() == 357.5


def test_roll_lon_plus_180_use_res(load_esgf_test_data):
    # test rolling longitude by -180
    ds, lon = setup_test()

    # work out how much to roll by
    offset = calculate_offset(180)

    ds = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)

    assert ds.lon.values[0] == 180.0
    assert ds.lon.values[-1] == 177.5

    assert ds.lon.values.min() == 0
    assert ds.lon.values.max() == 357.5


def test_plus_minus_180_equal(load_esgf_test_data):
    # check that rolling +180 and -180 gives the same result - when taking the resolution into account
    ds, lon = setup_test()

    # work out how much to roll by
    offset_minus = calculate_offset(-180)
    offset_plus = calculate_offset(180)

    ds_minus = ds.roll(shifts={f"{lon.name}": offset_minus}, roll_coords=True)
    ds_plus = ds.roll(shifts={f"{lon.name}": offset_plus}, roll_coords=True)

    # values of rlds are equal - rolling by -180 and 180 (taking res into account) is the same
    np.testing.assert_allclose(ds_minus.rlds.values, ds_plus.rlds.values)


@pytest.mark.skip(reason="rolling now done within subset")
def test_xarray_roll_lon(tmpdir, load_esgf_test_data):
    ds, lon = setup_test()

    # work out how much to roll by
    offset = calculate_offset(180)

    ds_roll = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)

    # testing after rolling still raises an error
    with pytest.raises(NotImplementedError):
        subset(
            ds=ds_roll,
            area=(-50.0, -90.0, 100.0, 90.0),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )


@pytest.mark.skip(reason="rolling now done within subset")
def test_convert_lon_coords(tmpdir, load_esgf_test_data):
    # test reassigning coords to convert to -180 to 180 for comparison
    ds, lon = setup_test()

    ds.coords[lon.name] = (ds.coords[lon.name] + 180) % 360 - 180
    ds = ds.sortby(ds[lon.name])

    assert isclose(ds.lon.values.min(), -180, abs_tol=10 ** 2)
    assert isclose(ds.lon.values.max(), 180, abs_tol=10 ** 2)

    result = subset(
        ds=ds,
        area=(-50.0, -90.0, 100.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    assert result


@pytest.mark.skip(reason="rolling now done within subset")
def test_roll_convert_lon_coords(load_esgf_test_data):
    ds, lon = setup_test()
    # work out how much to roll by
    offset = calculate_offset(180)

    ds_roll = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=False)

    # check roll with roll_coords=False actually does something
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        ds_roll.rlds.values,
        ds.rlds.values,
    )

    ds_roll.coords[lon.name] = ds_roll.coords[lon.name] - 180

    assert isclose(ds_roll.lon.values.min(), -180, abs_tol=10 ** 2)
    assert isclose(ds_roll.lon.values.max(), 180, abs_tol=10 ** 2)

    result = subset(
        ds=ds_roll,
        area=(-50.0, -90.0, 100.0, 90.0),
        output_type="xarray",
    )

    assert result


def test_roll_compare_roll_coords(load_esgf_test_data):
    ds, lon = setup_test()
    # work out how much to roll by
    offset = calculate_offset(180)

    ds_roll_coords = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)
    ds_not_roll_coords = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=False)

    # check rlds values the same with/without rolling coords
    np.testing.assert_array_equal(
        ds_roll_coords.rlds.values,
        ds_not_roll_coords.rlds.values,
    )

    # check lat doesn't change with/without rolling coords
    np.testing.assert_array_equal(
        ds_roll_coords.lat.values,
        ds_not_roll_coords.lat.values,
    )

    # check time doesn't change with/without rolling coords
    np.testing.assert_array_equal(
        ds_roll_coords.time.values,
        ds_not_roll_coords.time.values,
    )

    # check lon changes with/without rolling coords
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        ds_roll_coords.lon.values,
        ds_not_roll_coords.lon.values,
    )


@pytest.mark.skip(reason="rolling now done within subset")
def test_compare_methods(load_esgf_test_data):

    # run subset with rolling then assigning
    ds, lon = setup_test()

    # work out how much to roll by
    offset = calculate_offset(180)

    ds_roll = ds.roll(
        shifts={f"{lon.name}": offset}, roll_coords=False
    )  # roll coords set to false

    ds_roll.coords[lon.name] = ds_roll.coords[lon.name] - 180

    assert isclose(ds_roll.lon.values.min(), -180, abs_tol=10 ** 2)
    assert isclose(ds_roll.lon.values.max(), 180, abs_tol=10 ** 2)

    result1 = subset(
        ds=ds_roll,
        area=(-50.0, -90.0, 100.0, 90.0),
        output_type="xarray",
    )

    assert result1

    # run subset assign then sort by
    ds, lon = setup_test()

    ds.coords[lon.name] = (ds.coords[lon.name] + 180) % 360 - 180
    ds = ds.sortby(ds[lon.name])

    assert isclose(ds.lon.values.min(), -180, abs_tol=10 ** 2)
    assert isclose(ds.lon.values.max(), 180, abs_tol=10 ** 2)

    result2 = subset(
        ds=ds,
        area=(-50.0, -90.0, 100.0, 90.0),
        output_type="xarray",
    )

    assert result2

    # data of main variable is the same
    np.testing.assert_allclose(result1[0].rlds.values, result2[0].rlds.values)


@pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
def test_irregular_grid_dataset(load_esgf_test_data):
    ds = xr.open_mfdataset(
        "/badc/cmip6/data/CMIP6/ScenarioMIP/NCC/NorESM2-MM/"
        "ssp370/r1i1p1f1/Ofx/sftof/gn/v20191108/*.nc"
    )
    lon = get_coord_by_type(ds, "longitude", ignore_aux_coords=False)

    assert "lon" not in ds.dims

    with pytest.raises(ValueError) as exc:
        ds.roll(shifts={f"{lon.name}": 180}, roll_coords=False)
    assert str(exc.value) == "dimensions ['longitude'] do not exist"


@pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
def test_3d_grid_dataset(load_esgf_test_data):
    ds = xr.open_mfdataset(
        "/badc/cmip6/data/CMIP6/ScenarioMIP/NCC/NorESM2-MM/ssp370/r1i1p1f1/Amon/ta/gn/v20191108/*.nc"
    )
    lon = get_coord_by_type(ds, "longitude", ignore_aux_coords=False)

    assert "lon" in ds.dims

    offset = 180

    ds_roll_coords = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)
    ds_not_roll_coords = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=False)

    # check plev doesn't change with/without rolling coords
    np.testing.assert_array_equal(
        ds_roll_coords.plev.values,
        ds_not_roll_coords.plev.values,
    )
