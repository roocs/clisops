import os

import numpy as np
import pytest
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue

from clisops.core import average
from clisops.utils import get_file


class TestAverageOverDims:
    nc_file = get_file("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")

    def test_average_no_dims(self):
        ds = xr.open_dataset(self.nc_file)

        avg_ds = average.average_over_dims(ds)

        assert avg_ds == ds

    def test_average_one_dim(self):
        ds = xr.open_dataset(self.nc_file)

        avg_ds = average.average_over_dims(ds, ["latitude"])

        # time has not been averaged
        assert len(avg_ds.time) == 12

        # lat has been averaged over
        assert "lat" not in avg_ds.dims

    def test_average_two_dims(self):
        ds = xr.open_dataset(self.nc_file)

        avg_ds = average.average_over_dims(ds, ["latitude", "time"])

        # time has been averaged over
        assert "time" not in avg_ds.dims

        # lat has been averaged over
        assert "lat" not in avg_ds.dims

    def test_average_wrong_dim(self):
        ds = xr.open_dataset(self.nc_file)

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds, ["wrong", "latitude"])
        assert (
            str(exc.value)
            == "Dimensions for averaging must be one of ['time', 'level', 'latitude', 'longitude']"
        )

    def test_undetected_dim(self):
        ds = xr.open_dataset(self.nc_file)

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds, ["level", "time"])
        assert (
            str(exc.value)
            == "Requested dimensions were not found in input dataset: {'level'}."
        )

    def test_average_undetected_dim_ignore(self):
        ds = xr.open_dataset(self.nc_file)

        # exception should not be raised as ignore_undetected_dims set to True
        avg_ds = average.average_over_dims(
            ds, ["level", "time"], ignore_undetected_dims=True
        )

        # time has been averaged over
        assert "time" not in avg_ds.dims

    def test_average_wrong_format(self):
        ds = xr.open_dataset(self.nc_file)

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds, [0, "time"])
        assert (
            str(exc.value)
            == "Dimensions for averaging must be one of ['time', 'level', 'latitude', 'longitude']"
        )
