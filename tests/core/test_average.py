import os

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from pkg_resources import parse_version
from roocs_utils.exceptions import InvalidParameterValue

from clisops.core import average
from clisops.utils import get_file

from .._common import XCLIM_TESTS_DATA as TESTS_DATA

try:
    import xesmf

    if parse_version(xesmf.__version__) < parse_version("0.6.2"):
        raise ImportError
except ImportError:
    xesmf = None


@pytest.mark.skipif(xesmf is None, reason="xESMF >= 0.6.2 is needed for average_shape.")
class TestAverageShape:
    nc_file = get_file("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")
    lons_2d_nc_file = get_file("cmip6/sic_SImon_CCCma-CanESM5_ssp245_r13i1p2f1_2020.nc")
    nc_file_neglons = get_file("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc")
    meridian_geojson = os.path.join(TESTS_DATA, "cmip5", "meridian.json")
    meridian_multi_geojson = os.path.join(TESTS_DATA, "cmip5", "meridian_multi.json")
    poslons_geojson = os.path.join(TESTS_DATA, "cmip5", "poslons.json")
    eastern_canada_geojson = os.path.join(TESTS_DATA, "cmip5", "eastern_canada.json")
    southern_qc_geojson = os.path.join(TESTS_DATA, "cmip5", "southern_qc_geojson.json")
    small_geojson = os.path.join(TESTS_DATA, "cmip5", "small_geojson.json")
    multi_regions_geojson = os.path.join(TESTS_DATA, "cmip5", "multi_regions.json")

    def test_wraps(self, tmp_netcdf_filename):
        ds = xr.open_dataset(self.nc_file)

        # xESMF has a problem with averaging over dataset when non-averaged variables are present...
        avg = average.average_shape(ds.tas, self.meridian_geojson)

        # Check attributes are copied
        assert avg.attrs["units"] == ds.tas.attrs["units"]

        # No time subsetting should occur.
        assert len(avg.time) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(avg.isel(time=0), 285.533, 3)

        # Test with Dataset input
        davg = average.average_shape(ds, self.meridian_geojson).tas
        xr.testing.assert_equal(davg, avg)

        # With multiple polygons
        poly = gpd.read_file(self.meridian_multi_geojson)

        avg = average.average_shape(ds, poly).tas
        np.testing.assert_array_almost_equal(avg.isel(time=0), 280.965, 3)

    def test_no_wraps(self, tmp_netcdf_filename):
        ds = xr.open_dataset(self.nc_file)

        avg = average.average_shape(ds.tas, self.poslons_geojson)

        # No time subsetting should occur.
        assert len(avg.time) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(avg.isel(time=0), 276.152, 3)

    def test_all_neglons(self):
        ds = xr.open_dataset(self.nc_file_neglons)

        avg = average.average_shape(ds.tasmax, self.southern_qc_geojson)

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(avg.isel(time=0), 269.257, 3)

    # 2D lat/lon grids are buggy with current xesmf
    # def test_rotated_pole_with_time(self):
    #     ds = xr.open_dataset(self.lons_2d_nc_file)

    #     avg = average.average_shape(ds.rename(vertices='bounds'), self.eastern_canada_geojson)

    def test_average_multiregions(self):
        ds = xr.open_dataset(self.nc_file)
        regions = gpd.read_file(self.multi_regions_geojson).set_index("id")
        avg = average.average_shape(ds.tas, shape=regions)
        np.testing.assert_array_almost_equal(
            avg.isel(time=0), [268.620, 278.290, 277.863], decimal=3
        )
        np.testing.assert_array_equal(avg.geom, ["Québec", "Europe", "Newfoundland"])

    def test_non_overlapping_regions(self):
        ds = xr.open_dataset(self.nc_file_neglons)
        regions = gpd.read_file(self.meridian_geojson)

        with pytest.raises(ValueError):
            average.average_shape(ds.tasmax, shape=regions)


class TestAverageOverDims:
    nc_file = get_file("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")

    def test_average_no_dims(self):
        ds = xr.open_dataset(self.nc_file)

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds)
        assert str(exc.value) == "At least one dimension for averaging must be provided"

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
