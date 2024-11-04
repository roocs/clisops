import os
from math import isclose

import numpy as np
import pytest
import xarray as xr

from clisops.utils.dataset_utils import get_coord_by_type


class TestRoll:

    @staticmethod
    def calculate_offset(x, lon):
        # get resolution of data
        res = lon.values[1] - lon.values[0]

        # work out how many to roll by to roll data by 1
        index = 1 / res

        # calculate the corresponding offset needed to change data by x
        offset = int(x * index)

        return offset

    def test_roll_lon_minus_180(self, mini_esgf_data):
        ds = xr.open_dataset(mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"])
        # gets longitude by the correct name as used in the dataset
        lon = ds[get_coord_by_type(ds, "longitude")]

        # check longitude is 0 to 360 initially
        assert isclose(lon.values.min(), 0, abs_tol=10**2)
        assert isclose(lon.values.max(), 360, abs_tol=10**2)

        # roll longitude by -180
        ds = ds.roll(shifts={f"{lon.name}": -180}, roll_coords=True)

        # doesn't roll as far as we want as it hasn't taken the resolution of the grid into account
        assert ds.lon.values[0] == 90.0
        assert ds.lon.values[-1] == 87.5

        # min and max of the data are still the same
        assert ds.lon.values.min() == 0
        assert ds.lon.values.max() == 357.5

    def test_roll_lon_minus_180_use_res(self, mini_esgf_data):
        ds = xr.open_dataset(mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"])
        # gets longitude by the correct name as used in the dataset
        lon = ds[get_coord_by_type(ds, "longitude")]

        # work out how much to roll by
        offset = self.calculate_offset(-180, lon)

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

    def test_roll_lon_plus_180(self, mini_esgf_data):
        ds = xr.open_dataset(mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"])
        # gets longitude by the correct name as used in the dataset
        lon = ds[get_coord_by_type(ds, "longitude")]

        ds = ds.roll(shifts={f"{lon.name}": 180}, roll_coords=True)

        # doesn't roll as far as we want as it hasn't taken the resolution of the grid into account
        assert ds.lon.values[0] == 270.0
        assert ds.lon.values[-1] == 267.5

        assert ds.lon.values.min() == 0
        assert ds.lon.values.max() == 357.5

    def test_roll_lon_plus_180_use_res(self, mini_esgf_data):
        ds = xr.open_dataset(mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"])
        # gets longitude by the correct name as used in the dataset
        lon = ds[get_coord_by_type(ds, "longitude")]

        # work out how much to roll by
        offset = self.calculate_offset(180, lon)

        ds = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)

        assert ds.lon.values[0] == 180.0
        assert ds.lon.values[-1] == 177.5

        assert ds.lon.values.min() == 0
        assert ds.lon.values.max() == 357.5

    def test_plus_minus_180_equal(self, mini_esgf_data):
        # check that rolling +180 and -180 gives the same result - when taking the resolution into account
        ds = xr.open_dataset(mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"])
        # gets longitude by the correct name as used in the dataset
        lon = ds[get_coord_by_type(ds, "longitude")]

        # work out how much to roll by
        offset_minus = self.calculate_offset(-180, lon)
        offset_plus = self.calculate_offset(180, lon)

        ds_minus = ds.roll(shifts={f"{lon.name}": offset_minus}, roll_coords=True)
        ds_plus = ds.roll(shifts={f"{lon.name}": offset_plus}, roll_coords=True)

        # values of rlds are equal - rolling by -180 and 180 (taking res into account) is the same
        np.testing.assert_allclose(ds_minus.rlds.values, ds_plus.rlds.values)

    def test_roll_compare_roll_coords(self, mini_esgf_data):
        ds = xr.open_dataset(mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"])
        # gets longitude by the correct name as used in the dataset
        lon = ds[get_coord_by_type(ds, "longitude")]

        # work out how much to roll by
        offset = self.calculate_offset(180, lon)

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


def test_irregular_grid_dataset(mini_esgf_data):
    ds = xr.open_dataset(mini_esgf_data["CMIP6_SIMASS_DEGEN"])
    lon = get_coord_by_type(ds, "longitude", ignore_aux_coords=False)

    assert "lon" not in ds.dims

    with pytest.raises(ValueError) as exc:
        ds.roll(shifts={f"{lon}": 180}, roll_coords=False)
    assert str(exc.value) in [
        "dimensions ['longitude'] do not exist",
        "Dimensions ['longitude'] not found in data dimensions ('i', 'j', 'time', 'bnds', 'vertices')",
    ]


def test_3d_grid_dataset(mini_esgf_data):
    ds = xr.open_mfdataset(mini_esgf_data["CMIP6_TA"])
    lon = get_coord_by_type(ds, "longitude", ignore_aux_coords=False)

    assert "lon" in ds.dims

    offset = 180

    ds_roll_coords = ds.roll(shifts={f"{lon}": offset}, roll_coords=True)
    ds_not_roll_coords = ds.roll(shifts={f"{lon}": offset}, roll_coords=False)

    # check plev doesn't change with/without rolling coords
    np.testing.assert_array_equal(
        ds_roll_coords.plev.values,
        ds_not_roll_coords.plev.values,
    )
