import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

from clisops.core import average
from clisops.core.regrid import XESMF_MINIMUM_VERSION
from clisops.exceptions import InvalidParameterValue
from clisops.utils import dataset_utils as xu

try:
    import xesmf

    if Version(xesmf.__version__) < Version(XESMF_MINIMUM_VERSION):
        raise ImportError()
except ImportError:
    xesmf = None


@pytest.mark.skipif(
    xesmf is None,
    reason=f"xesmf >= {XESMF_MINIMUM_VERSION} is needed for average_shape",
)
class TestAverageShape:
    # Fetch remote netcdf files
    nc_file = "cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc"
    lons_2d_nc_file = "cmip6/sic_SImon_CCCma-CanESM5_ssp245_r13i1p2f1_2020.nc"
    nc_file_neglons = "NRCANdaily/nrcan_canada_daily_tasmax_1990.nc"

    def test_wraps(self, tmp_netcdf_filename, nimbus, clisops_test_data):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        # xESMF has a problem with averaging over dataset when non-averaged variables are present...
        avg = average.average_shape(ds.tas, clisops_test_data["meridian_geojson"])

        # Check attributes are copied
        assert avg.attrs["units"] == ds.tas.attrs["units"]

        # No time subsetting should occur.
        assert len(avg.time) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(avg.isel(time=0), 285.533, 3)

        # Test with Dataset input
        davg = average.average_shape(ds, clisops_test_data["meridian_geojson"]).tas
        xr.testing.assert_equal(davg, avg)

        # With multiple polygons
        poly = gpd.read_file(clisops_test_data["meridian_multi_geojson"])

        avg = average.average_shape(ds, poly).tas
        np.testing.assert_array_almost_equal(avg.isel(time=0), 280.965, 3)

    def test_no_wraps(self, tmp_netcdf_filename, nimbus, clisops_test_data):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        avg = average.average_shape(ds.tas, clisops_test_data["poslons_geojson"])

        # No time subsetting should occur.
        assert len(avg.time) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(avg.isel(time=0), 276.152, 3)

    def test_all_neglons(self, nimbus, clisops_test_data):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file_neglons))

        avg = average.average_shape(ds.tasmax, clisops_test_data["southern_qc_geojson"])

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(avg.isel(time=0), 269.257, 3)

    # 2D lat/lon grids are buggy with current xesmf
    # def test_rotated_pole_with_time(self):
    #     ds = xr.open_dataset(self.lons_2d_nc_file)

    #     avg = average.average_shape(ds.rename(vertices='bounds'), self.eastern_canada_geojson)

    def test_average_multiregions(self, nimbus, clisops_test_data):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))
        regions = gpd.read_file(clisops_test_data["multi_regions_geojson"]).set_index(
            "id"
        )
        avg = average.average_shape(ds.tas, shape=regions)
        np.testing.assert_array_almost_equal(
            avg.isel(time=0), [268.620, 278.290, 277.863], decimal=3
        )
        np.testing.assert_array_equal(avg.geom, ["Québec", "Europe", "Newfoundland"])

    def test_non_overlapping_regions(self, nimbus, clisops_test_data):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file_neglons))
        regions = gpd.read_file(clisops_test_data["meridian_geojson"])

        with pytest.raises(ValueError):
            average.average_shape(ds.tasmax, shape=regions)


class TestAverageOverDims:
    nc_file = "cmip5/tas_Amon_CanESM2_rcp85_r1i1p1_200701-200712.nc"

    def test_average_no_dims(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds)
        assert str(exc.value) == "At least one dimension for averaging must be provided"

    def test_average_one_dim(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        avg_ds = average.average_over_dims(ds, ["latitude"])

        # time has not been averaged
        assert len(avg_ds.time) == 12

        # lat has been averaged over
        assert "lat" not in avg_ds.dims

    def test_average_two_dims(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        avg_ds = average.average_over_dims(ds, ["latitude", "time"])

        # time has been averaged over
        assert "time" not in avg_ds.dims

        # lat has been averaged over
        assert "lat" not in avg_ds.dims

    def test_average_wrong_dim(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds, ["wrong", "latitude"])
        assert (
            str(exc.value)
            == "Dimensions for averaging must be one of ['time', 'level', 'latitude', 'longitude', 'realization']"
        )

    def test_undetected_dim(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds, ["level", "time"])
        assert (
            str(exc.value)
            == "Requested dimensions were not found in input dataset: {'level'}."
        )

    def test_average_undetected_dim_ignore(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        # exception should not be raised as ignore_undetected_dims set to True
        avg_ds = average.average_over_dims(
            ds, ["level", "time"], ignore_undetected_dims=True
        )

        # time has been averaged over
        assert "time" not in avg_ds.dims

    def test_average_wrong_format(self, nimbus):
        ds = xr.open_dataset(nimbus.fetch(self.nc_file))

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_over_dims(ds, [0, "time"])
        assert (
            str(exc.value)
            == "Dimensions for averaging must be one of ['time', 'level', 'latitude', 'longitude', 'realization']"
        )


class TestAverageTime:
    month_ds = "CMIP5_RH"

    def test_average_month(self, mini_esgf_data):
        ds = xu.open_xr_dataset(mini_esgf_data["CMIP6_SICONC_DAY"])
        assert ds.time.shape == (60225,)

        avg_ds = average.average_time(ds, freq="month")
        assert avg_ds.time.shape == (1980,)

    def test_average_year(self, mini_esgf_data):
        ds = xu.open_xr_dataset(mini_esgf_data[self.month_ds])
        assert ds.time.shape == (1752,)

        avg_ds = average.average_time(ds, freq="year")
        assert avg_ds.time.shape == (147,)

    def test_no_freq(self, mini_esgf_data):
        ds = xu.open_xr_dataset(mini_esgf_data[self.month_ds])

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_time(ds, freq=None)  # noqa

        assert str(exc.value) == "At least one frequency for averaging must be provided"

    def test_incorrect_freq(self, mini_esgf_data):
        ds = xu.open_xr_dataset(mini_esgf_data[self.month_ds])

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_time(ds, freq="wrong")
        assert (
            str(exc.value)
            == "Time frequency for averaging must be one of ['day', 'month', 'year']."
        )

    def test_freq_wrong_format(self, mini_esgf_data):
        ds = xu.open_xr_dataset(mini_esgf_data[self.month_ds])

        with pytest.raises(InvalidParameterValue) as exc:
            average.average_time(ds, freq=0)  # noqa

        assert str(exc.value) == "At least one frequency for averaging must be provided"

    def test_no_time(self, mini_esgf_data):
        ds = xu.open_xr_dataset(mini_esgf_data[self.month_ds])
        ds = ds.drop_dims("time")

        with pytest.raises(Exception) as exc:
            average.average_time(ds, freq="year")
        assert str(exc.value) == "Time dimension could not be found"
