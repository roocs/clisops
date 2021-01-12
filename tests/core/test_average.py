import os

import geopandas as gpd
import numpy as np
import xarray as xr

from clisops.core import average
from clisops.utils import get_file

from .._common import XCLIM_TESTS_DATA as TESTS_DATA


class TestAverageShape:
    nc_file = get_file("cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc")
    lons_2d_nc_file = get_file("CRCM5/tasmax_bby_198406_se.nc")
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

        # No time subsetting should occur.
        assert len(avg.time) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            avg.isel(time=0), 284.98243933
        )

        poly = gpd.read_file(self.meridian_multi_geojson)
        avg = average.average_shape(ds.tas, poly)
        np.testing.assert_array_almost_equal(
            avg.isel(time=0), 280.67990737
        )

    def test_no_wraps(self, tmp_netcdf_filename):
        ds = xr.open_dataset(self.nc_file)

        avg = average.average_shape(ds.tas, self.poslons_geojson)

        # No time subsetting should occur.
        assert len(avg.time) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            avg.isel(time=0), 276.17126511
        )

    def test_all_neglons(self):
        ds = xr.open_dataset(self.nc_file_neglons)

        avg = average.average_shape(ds.tasmax, self.southern_qc_geojson)

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            avg.isel(time=0), 269.25454934
        )

    # Test not working with cf_xarray 0.3.1 (issue xESMF#55)
    # Also, we need lon_bnds and lat_bnds, which are unavailable and uninferable.
    # def test_rotated_pole_with_time(self):
    #     ds = xr.open_dataset(self.lons_2d_nc_file)

    #     with pytest.warns(None) as record:
    #         sub = subset.subset_shape(
    #             ds,
    #             self.eastern_canada_geojson,
    #         )

    #     # Should only have 15 days of data.
    #     assert len(sub.tasmax) == 15
    #     # Average max temperature at surface for region on June 1st, 1984 (time=0)
    #     np.testing.assert_allclose(float(np.mean(sub.tasmax.isel(time=0))), 289.634968)

    def test_average_multiregions(self):
        ds = xr.open_dataset(self.nc_file)
        regions = gpd.read_file(self.multi_regions_geojson).set_index("id")
        avg = average.average_shape(ds.tas, shape=regions)
        np.testing.assert_array_almost_equal(avg.isel(time=0), [268.30972367, 277.23981999, 277.58614891])
        np.testing.assert_array_equal(avg.geom, ['Québec', 'Europe', 'Newfoundland'])
