import os

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon

from clisops.core import subset
from clisops.utils import get_file

from .._common import XCLIM_TESTS_DATA as TESTS_DATA

try:
    import xesmf
except ImportError:
    xesmf = None

try:
    import pygeos
except ImportError:
    pygeos = None


@pytest.fixture(params=[True, False])
def toggle_pygeos(request):
    if request.param:
        if pygeos is None:
            pytest.skip("pygeos not installed")
        else:
            yield request.param
    else:
        # Do not use pygeos
        mod = subset.pygeos
        # Monkeypatch core.subset to imitate the absence of pygeos.
        subset.pygeos = None
        yield request.param
        subset.pygeos = mod


class TestSubsetTime:
    nc_poslons = get_file("cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc")

    def test_simple(self):
        da = xr.open_dataset(self.nc_poslons).tas
        yr_st = "2050"
        yr_ed = "2059"

        out = subset.subset_time(da, start_date=yr_st, end_date=yr_ed)
        out1 = subset.subset_time(da, start_date=f"{yr_st}-01", end_date=f"{yr_ed}-12")
        out2 = subset.subset_time(
            da, start_date=f"{yr_st}-01-01", end_date=f"{yr_ed}-12-31"
        )
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out, out2)
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))

    def test_time_dates_outofbounds(self):
        da = xr.open_dataset(self.nc_poslons).tas
        yr_st = "1776"
        yr_ed = "2077"

        with pytest.warns(None) as record:  # noqa
            out = subset.subset_time(
                da, start_date=f"{yr_st}-01", end_date=f"{yr_ed}-01"
            )
        np.testing.assert_array_equal(out.time.dt.year.min(), da.time.dt.year.min())
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())

        assert (
            '"start_date" not found within input date time range. Defaulting to minimum time step in xarray object.'
            in [str(q.message) for q in record]
        )
        assert (
            '"end_date" not found within input date time range. Defaulting to maximum time step in xarray object.'
            in [str(q.message) for q in record]
        )

    def test_warnings(self):
        da = xr.open_dataset(self.nc_poslons).tas

        with pytest.raises(ValueError) as record:
            subset.subset_time(da, start_date="2059", end_date="2050")

        with pytest.raises(TypeError):
            subset.subset_time(da, start_yr=2050, end_yr=2059)

        with pytest.warns(None) as record:  # noqa
            subset.subset_time(
                da,
                start_date=2050,  # noqa
                end_date=2055,  # noqa
            )
        assert (
            'start_date and end_date require dates in (type: str) using formats of "%Y", "%Y-%m" or "%Y-%m-%d".'
            in [str(q.message) for q in record]
        )

        with pytest.warns(None) as record:  # noqa
            subset.subset_time(
                da, start_date="2064-01-01T00:00:00", end_date="2065-02-01T03:12:01"
            )

        assert (
            '"start_date" has been nudged to nearest valid time step in xarray object.'
            in [str(q.message) for q in record]
        )

        assert (
            '"end_date" has been nudged to nearest valid time step in xarray object.'
            in [str(q.message) for q in record]
        )

    def test_time_start_only(self):
        da = xr.open_dataset(self.nc_poslons).tas
        yr_st = "2050"

        # start date only
        with pytest.warns(None):  # noqa
            out = subset.subset_time(da, start_date=f"{yr_st}-01")
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())

        with pytest.warns(None):  # noqa
            out = subset.subset_time(da, start_date=f"{yr_st}-07")
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.max(), da.time.max())

        with pytest.warns(None):  # noqa
            out = subset.subset_time(da, start_date=f"{yr_st}-07-15")
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.min().dt.day, 15)
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.max(), da.time.max())

    def test_time_end_only(self):
        da = xr.open_dataset(self.nc_poslons).tas
        yr_ed = "2059"

        # end date only
        with pytest.warns(None):  # noqa
            out = subset.subset_time(da, end_date=f"{yr_ed}-01")
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 1)
        np.testing.assert_array_equal(out.time.max().dt.day, 31)
        np.testing.assert_array_equal(out.time.min(), da.time.min())

        with pytest.warns(None):  # noqa
            out = subset.subset_time(da, end_date=f"{yr_ed}-06-15")
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 6)
        np.testing.assert_array_equal(out.time.max().dt.day, 15)
        np.testing.assert_array_equal(out.time.min(), da.time.min())

    def test_time_incomplete_years(self):
        da = xr.open_dataset(self.nc_poslons).tas
        yr_st = "2050"
        yr_ed = "2059"

        out = subset.subset_time(
            da, start_date=f"{yr_st}-07-01", end_date=f"{yr_ed}-06-30"
        )
        out1 = subset.subset_time(da, start_date=f"{yr_st}-07", end_date=f"{yr_ed}-06")
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))
        np.testing.assert_array_equal(out.time.min().dt.month, 7)
        np.testing.assert_array_equal(out.time.min().dt.day, 1)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.max().dt.month, 6)
        np.testing.assert_array_equal(out.time.max().dt.day, 30)


class TestSubsetGridPoint:
    nc_poslons = get_file("cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc")
    nc_tasmax_file = get_file("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin_file = get_file("NRCANdaily/nrcan_canada_daily_tasmin_1990.nc")
    nc_2dlonlat = get_file("CRCM5/tasmax_bby_198406_se.nc")

    def test_time_simple(self):
        da = xr.open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))
        lon = -72.4
        lat = 46.1
        yr_st = "2050"
        yr_ed = "2059"

        out = subset.subset_gridpoint(
            da, lon=lon, lat=lat, start_date=yr_st, end_date=yr_ed
        )
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        np.testing.assert_array_equal(len(np.unique(out.time.dt.year)), 10)
        np.testing.assert_array_equal(out.time.dt.year.max(), int(yr_ed))
        np.testing.assert_array_equal(out.time.dt.year.min(), int(yr_st))

    def test_dataset(self):
        da = xr.open_mfdataset(
            [self.nc_tasmax_file, self.nc_tasmin_file],
            combine="by_coords",
        )
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        np.testing.assert_array_equal(out.tasmin.shape, out.tasmax.shape)

    @pytest.mark.parametrize(
        "lon,lat", [([-72.4], [46.1]), ([-67.4, -67.3], [43.1, 46.1])]
    )
    @pytest.mark.parametrize("add_distance", [True, False])
    def test_simple(self, lat, lon, add_distance):
        da = xr.open_dataset(self.nc_tasmax_file).tasmax

        out = subset.subset_gridpoint(da, lon=lon, lat=lat, add_distance=add_distance)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        assert ("site" in out.dims) ^ (len(lat) == 1)
        assert ("distance" in out.coords) ^ (not add_distance)

    def test_irregular(self):
        da = xr.open_dataset(self.nc_2dlonlat).tasmax
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        assert "site" not in out.dims

        lon = [-72.4, -67.1]
        lat = [46.1, 48.2]
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)
        assert "site" in out.dims

        # dask for lon lat
        da.lon.chunk({"rlon": 10})
        da.lat.chunk({"rlon": 10})
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        # test_irregular transposed:
        da1 = xr.open_dataset(self.nc_2dlonlat).tasmax
        dims = list(da1.dims)
        dims.reverse()
        daT = xr.DataArray(np.transpose(da1.values), dims=dims)
        for d in daT.dims:
            args = dict()
            args[d] = da1[d]
            daT = daT.assign_coords(**args)
        daT = daT.assign_coords(lon=(["rlon", "rlat"], np.transpose(da1.lon.values)))
        daT = daT.assign_coords(lat=(["rlon", "rlat"], np.transpose(da1.lat.values)))

        out1 = subset.subset_gridpoint(daT, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out1.lon, lon, 1)
        np.testing.assert_almost_equal(out1.lat, lat, 1)
        np.testing.assert_array_equal(out, out1)

        # Dataset with tasmax, lon and lat as data variables (i.e. lon, lat not coords of tasmax)
        daT1 = xr.DataArray(np.transpose(da1.values), dims=dims)
        for d in daT1.dims:
            args = dict()
            args[d] = da1[d]
            daT1 = daT1.assign_coords(**args)
        dsT = xr.Dataset(data_vars=None, coords=daT1.coords)
        dsT["tasmax"] = daT1
        dsT["lon"] = xr.DataArray(np.transpose(da1.lon.values), dims=["rlon", "rlat"])
        dsT["lat"] = xr.DataArray(np.transpose(da1.lat.values), dims=["rlon", "rlat"])
        out2 = subset.subset_gridpoint(dsT, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out2.lon, lon, 1)
        np.testing.assert_almost_equal(out2.lat, lat, 1)
        np.testing.assert_array_equal(out, out2.tasmax)

        # Dataset with lon and lat as 1D arrays
        lon = -60
        lat = -45
        da = xr.DataArray(
            np.random.rand(5, 4),
            dims=("time", "site"),
            coords={"time": np.arange(5), "site": np.arange(4)},
        )
        ds = xr.Dataset(
            data_vars={
                "da": da,
                "lon": ("site", np.linspace(lon, lon + 10, 4)),
                "lat": ("site", np.linspace(lat, lat + 5, 4)),
            }
        )
        gp = subset.subset_gridpoint(ds, lon=lon, lat=lat)
        np.testing.assert_almost_equal(gp.lon, lon)
        np.testing.assert_almost_equal(gp.lat, lat)
        assert gp.site == 0

    def test_positive_lons(self):
        da = xr.open_dataset(self.nc_poslons).tas
        lon = -72.4
        lat = 46.1
        out = subset.subset_gridpoint(da, lon=lon, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

        out = subset.subset_gridpoint(da, lon=lon + 360, lat=lat)
        np.testing.assert_almost_equal(out.lon, lon + 360, 1)
        np.testing.assert_almost_equal(out.lat, lat, 1)

    def test_raise(self):
        da = xr.open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_gridpoint(
                da, lon=-72.4, lat=46.1, start_date="2055-03-15", end_date="2055-03-14"
            )
            subset.subset_gridpoint(
                da, lon=-72.4, lat=46.1, start_date="2055", end_date="2052"
            )
        da = xr.open_dataset(self.nc_2dlonlat).tasmax.drop_vars(names=["lon", "lat"])
        with pytest.raises(Exception):
            subset.subset_gridpoint(da, lon=-72.4, lat=46.1)

    def test_tolerance(self):
        da = xr.open_dataset(self.nc_poslons).tas
        lon = -72.5
        lat = 46.2
        out = subset.subset_gridpoint(da, lon=lon, lat=lat, tolerance=1)
        assert out.isnull().all()

        subset.subset_gridpoint(da, lon=lon, lat=lat, tolerance=1e5)


class TestSubsetBbox:
    nc_poslons = get_file("cmip3/tas.sresb1.giss_model_e_r.run1.atm.da.nc")
    nc_tasmax_file = get_file("NRCANdaily/nrcan_canada_daily_tasmax_1990.nc")
    nc_tasmin_file = get_file("NRCANdaily/nrcan_canada_daily_tasmin_1990.nc")
    nc_2dlonlat = get_file("CRCM5/tasmax_bby_198406_se.nc")
    lon = [-75.4, -68]
    lat = [44.1, 47.1]
    lonGCM = [-70.0, -60.0]
    latGCM = [43.0, 59.0]

    def test_dataset(self):
        da = xr.open_mfdataset(
            [self.nc_tasmax_file, self.nc_tasmin_file],
            combine="by_coords",
        )
        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))
        np.testing.assert_array_equal(out.tasmin.shape, out.tasmax.shape)

    def test_simple(self):
        da = xr.open_dataset(self.nc_tasmax_file).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat.values >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))

        da = xr.open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))
        yr_st = 2050
        yr_ed = 2059

        out = subset.subset_bbox(
            da,
            lon_bnds=self.lonGCM,
            lat_bnds=self.latGCM,
            start_date=str(yr_st),
            end_date=str(yr_ed),
        )
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lonGCM))
        assert np.all(out.lon <= np.max(self.lonGCM))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

        out = subset.subset_bbox(
            da, lon_bnds=self.lon, lat_bnds=self.lat, start_date=str(yr_st)
        )

        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))
        np.testing.assert_array_equal(out.time.dt.year.max(), da.time.dt.year.max())
        np.testing.assert_array_equal(out.time.dt.year.min(), yr_st)

        out = subset.subset_bbox(
            da, lon_bnds=self.lon, lat_bnds=self.lat, end_date=str(yr_ed)
        )

        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lon))
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))
        np.testing.assert_array_equal(out.time.dt.year.max(), yr_ed)
        np.testing.assert_array_equal(out.time.dt.year.min(), da.time.dt.year.min())

    def test_irregular(self):
        da = xr.open_dataset(self.nc_2dlonlat).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)

        # for irregular lat lon grids data matrix remains rectangular in native proj
        # but with data outside bbox assigned nans.  This means it can have lon and lats outside the bbox.
        # Check only non-nans gridcells using mask
        mask1 = ~(np.isnan(out.sel(time=out.time[0])))
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon.values[mask1.values] >= np.min(self.lon))
        assert np.all(out.lon.values[mask1.values] <= np.max(self.lon))
        assert np.all(out.lat.values[mask1.values] >= np.min(self.lat))
        assert np.all(out.lat.values[mask1.values] <= np.max(self.lat))

    def test_irregular_dataset(self):
        da = xr.open_dataset(self.nc_2dlonlat)
        out = subset.subset_bbox(da, lon_bnds=[-150, 100], lat_bnds=[10, 60])
        variables = list(da.data_vars)
        variables.pop(variables.index("tasmax"))
        # only tasmax should be subsetted/masked others should remain untouched
        for v in variables:
            assert out[v].dims == da[v].dims
            np.testing.assert_array_equal(out[v], da[v])

        # ensure results are equal to previous test on DataArray only
        out1 = subset.subset_bbox(da.tasmax, lon_bnds=[-150, 100], lat_bnds=[10, 60])
        np.testing.assert_array_equal(out1, out.tasmax)

        # additional test if dimensions have no coordinates
        da = da.drop_vars(["rlon", "rlat"])
        subset.subset_bbox(da.tasmax, lon_bnds=[-150, 100], lat_bnds=[10, 60])
        # We don't test for equality with previous datasets.
        # Without coords, sel defaults to isel which doesn't include the last element.

    def test_irregular_inverted_dataset(self):
        da = xr.open_dataset(self.nc_2dlonlat)
        da_rev = da.sortby("rlat", ascending=False).sortby("rlon", ascending=False)
        out = subset.subset_bbox(da, lon_bnds=[-150, -100], lat_bnds=[10, 60])
        out_rev = subset.subset_bbox(da_rev, lon_bnds=[-150, -100], lat_bnds=[10, 60])
        variables = list(da.data_vars)
        variables.pop(variables.index("tasmax"))
        # only tasmax should be subsetted/masked others should remain untouched
        for v in variables:
            assert out[v].dims == da[v].dims
            np.testing.assert_array_equal(out_rev[v], da[v])

        # ensure results are equal to previous test on DataArray only
        out1 = subset.subset_bbox(da.tasmax, lon_bnds=[-150, -100], lat_bnds=[10, 60])
        out1_rev = subset.subset_bbox(
            da_rev.tasmax, lon_bnds=[-150, -100], lat_bnds=[10, 60]
        )
        diff = (out1 - out1_rev).values
        np.testing.assert_array_equal(np.unique(diff[~np.isnan(diff)]), 0)

        # additional test if dimensions have no coordinates
        da_rev = da_rev.drop_vars(["rlon", "rlat"])
        subset.subset_bbox(da_rev.tasmax, lon_bnds=[-150, -100], lat_bnds=[10, 60])
        # We don't test for equality with previous datasets.
        # Without coords, sel defaults to isel which doesn't include the last element.

    # TODO: this test doesn't seem to do anything : remove?
    # Note - previous failure was from using a lon/lat bnds where there is no data. Changed below
    # def test_irregular_straight_lon_lat(self):
    #     ds = xr.open_dataset(self.nc_2dlonlat)
    #     # pass
    #     # N. AM data use appropriate zone to avoid fail
    #     subset.subset_bbox(ds.tasmax, lon_bnds=[-100, -80], lat_bnds=[40, 45])

    # test datasets with descending coords
    def test_inverted_coords(self):
        lon = np.linspace(-90, -60, 200)
        lat = np.linspace(40, 80, 100)
        da = xr.Dataset(
            data_vars=None, coords={"lon": np.flip(lon), "lat": np.flip(lat)}
        )
        da["data"] = xr.DataArray(
            np.random.rand(lon.size, lat.size), dims=["lon", "lat"]
        )

        out = subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(np.asarray(self.lon)))
        assert np.all(out.lon <= np.max(np.asarray(self.lon)))
        assert np.all(out.lat >= np.min(self.lat))
        assert np.all(out.lat <= np.max(self.lat))

    def test_badly_named_latlons(self):
        da = xr.open_dataset(self.nc_tasmax_file)
        extended_latlons = {"lat": "latitude", "lon": "longitude"}
        da_extended_names = da.rename(extended_latlons)
        out = subset.subset_bbox(
            da_extended_names, lon_bnds=self.lon, lat_bnds=self.lat
        )
        assert {"latitude", "longitude"}.issubset(out.dims)

        long_for_some_reason = {"lon": "long"}
        da_long = da.rename(long_for_some_reason)
        out = subset.subset_bbox(da_long, lon_bnds=self.lon, lat_bnds=self.lat)
        assert {"long"}.issubset(out.dims)

        lons_lats = {"lon": "lons", "lat": "lats"}
        da_lonslats = da.rename(lons_lats)
        out = subset.subset_bbox(da_lonslats, lon_bnds=self.lon, lat_bnds=self.lat)
        assert {"lons", "lats"}.issubset(out.dims)

    def test_single_bounds_rectilinear(self):
        da = xr.open_dataset(self.nc_tasmax_file).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        np.testing.assert_array_equal(out.lat, da.lat)
        assert np.all(out.lon <= np.max(self.lon))
        assert np.all(out.lon.values >= np.min(self.lon))

        out = subset.subset_bbox(da, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        np.testing.assert_array_equal(out.lon, da.lon)
        assert np.all(out.lat <= np.max(self.lat))
        assert np.all(out.lat.values >= np.min(self.lat))

    def test_single_bounds_curvilinear(self):
        da = xr.open_dataset(self.nc_2dlonlat).tasmax

        out = subset.subset_bbox(da, lon_bnds=self.lon)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        mask1 = ~(np.isnan(out.sel(time=out.time[0])))
        assert np.all(out.lon.values[mask1.values] <= np.max(self.lon))
        assert np.all(out.lon.values[mask1.values] >= np.min(self.lon))

        out = subset.subset_bbox(da, lat_bnds=self.lat)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        mask1 = ~(np.isnan(out.sel(time=out.time[0])))
        assert np.all(out.lat.values[mask1.values] <= np.max(self.lat))
        assert np.all(out.lat.values[mask1.values] >= np.min(self.lat))

    def test_positive_lons(self):
        da = xr.open_dataset(self.nc_poslons).tas

        out = subset.subset_bbox(da, lon_bnds=self.lonGCM, lat_bnds=self.latGCM)
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(np.asarray(self.lonGCM) + 360))
        assert np.all(out.lon <= np.max(np.asarray(self.lonGCM) + 360))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))

        out = subset.subset_bbox(
            da, lon_bnds=np.array(self.lonGCM) + 360, lat_bnds=self.latGCM
        )
        assert np.all(out.lon >= np.min(np.asarray(self.lonGCM) + 360))

    def test_time(self):
        da = xr.open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))

        out = subset.subset_bbox(
            da,
            lon_bnds=self.lonGCM,
            lat_bnds=self.latGCM,
            start_date="2050",
            end_date="2059",
        )
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lonGCM))
        assert np.all(out.lon <= np.max(self.lonGCM))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))
        np.testing.assert_array_equal(out.time.min().dt.year, 2050)
        np.testing.assert_array_equal(out.time.min().dt.month, 1)
        np.testing.assert_array_equal(out.time.min().dt.day, 1)
        np.testing.assert_array_equal(out.time.max().dt.year, 2059)
        np.testing.assert_array_equal(out.time.max().dt.month, 12)
        np.testing.assert_array_equal(out.time.max().dt.day, 31)

        out = subset.subset_bbox(
            da,
            lon_bnds=self.lonGCM,
            lat_bnds=self.latGCM,
            start_date="2050-02-05",
            end_date="2059-07-15",
        )
        assert out.lon.values.size != 0
        assert out.lat.values.size != 0
        assert np.all(out.lon >= np.min(self.lonGCM))
        assert np.all(out.lon <= np.max(self.lonGCM))
        assert np.all(out.lat >= np.min(self.latGCM))
        assert np.all(out.lat <= np.max(self.latGCM))
        np.testing.assert_array_equal(out.time.min().dt.year, 2050)
        np.testing.assert_array_equal(out.time.min().dt.month, 2)
        np.testing.assert_array_equal(out.time.min().dt.day, 5)
        np.testing.assert_array_equal(out.time.max().dt.year, 2059)
        np.testing.assert_array_equal(out.time.max().dt.month, 7)
        np.testing.assert_array_equal(out.time.max().dt.day, 15)

    def test_raise(self):
        # 1st case
        da = xr.open_dataset(self.nc_poslons).tas
        with pytest.raises(ValueError):
            subset.subset_bbox(
                da,
                lon_bnds=self.lonGCM,
                lat_bnds=self.latGCM,
                start_date="2056",
                end_date="2055",
            )

        # 2nd case
        da = xr.open_dataset(self.nc_2dlonlat).tasmax.drop_vars(names=["lon", "lat"])
        with pytest.raises(Exception):
            subset.subset_bbox(da, lon_bnds=self.lon, lat_bnds=self.lat)

        # 3rd case
        ds = xr.Dataset(
            data_vars={"var": (("lat", "lon"), np.ones((5, 10)))},
            coords={
                "lat": ("lat", np.zeros(5)),
                "lon": ("lon", np.arange(-10, 0, 1)),
            },
        )
        ds["lat"].attrs["standard_name"] = "latitude"
        ds["lon"].attrs["standard_name"] = "longitude"
        with pytest.raises(ValueError):
            subset.subset_bbox(ds, lon_bnds=(-0.1, 1.0))

    def test_warnings(self):
        da = xr.open_dataset(self.nc_poslons).tas
        da = da.assign_coords(lon=(da.lon - 360))

        with pytest.raises(TypeError):
            subset.subset_bbox(
                da, lon_bnds=self.lon, lat_bnds=self.lat, start_yr=2050, end_yr=2059
            )
        with pytest.warns(None) as record:  # noqa
            subset.subset_bbox(
                da,
                lon_bnds=self.lon,
                lat_bnds=self.lat,
                start_date="2050",
                end_date="2055",
            )
        assert (
            '"start_yr" and "end_yr" (type: int) are being deprecated. Temporal subsets will soon exclusively'
            ' support "start_date" and "end_date" (type: str) using formats of "%Y", "%Y-%m" or "%Y-%m-%d".'
            not in [str(q.message) for q in record]
        )


class TestSubsetShape:
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

    @staticmethod
    def compare_vals(ds, sub, vari, flag_2d=False):
        # check subsetted values against original
        imask = np.where(~np.isnan(sub[vari].isel(time=0)))
        if len(imask[0]) > 70:
            np.random.RandomState = 42
            ii = np.random.randint(0, len(imask[0]), 70)
        else:
            ii = np.arange(0, len(imask[0]))
        for i in zip(imask[0][ii], imask[1][ii]):
            if flag_2d:
                lat1 = sub.lat[i[0], i[1]]
                lon1 = sub.lon[i[0], i[1]]
                np.testing.assert_array_equal(
                    subset.subset_gridpoint(sub, lon=lon1, lat=lat1)[vari],
                    subset.subset_gridpoint(ds, lon=lon1, lat=lat1)[vari],
                )
            else:
                lat1 = sub.lat.isel(lat=i[0])
                lon1 = sub.lon.isel(lon=i[1])
                # print(lon1.values, lat1.values)
                np.testing.assert_array_equal(
                    sub[vari].sel(lon=lon1, lat=lat1), ds[vari].sel(lon=lon1, lat=lat1)
                )

    def test_wraps(self, tmp_netcdf_filename):
        ds = xr.open_dataset(self.nc_file)

        # Polygon crosses meridian, a warning should be raised
        with pytest.warns(UserWarning):
            sub = subset.subset_shape(ds, self.meridian_geojson)

        # No time subsetting should occur.
        assert len(sub.tas) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            float(np.mean(sub.tas.isel(time=0))), 285.064, 3
        )
        self.compare_vals(ds, sub, "tas")

        poly = gpd.read_file(self.meridian_multi_geojson)
        subtas = subset.subset_shape(ds.tas, poly)
        np.testing.assert_array_almost_equal(
            float(np.mean(subtas.isel(time=0))), 281.092, 3
        )

        assert sub.crs.prime_meridian_name == "Greenwich"
        assert sub.crs.grid_mapping_name == "latitude_longitude"

        sub.to_netcdf(tmp_netcdf_filename)
        assert tmp_netcdf_filename.exists()
        with xr.open_dataset(filename_or_obj=tmp_netcdf_filename) as f:
            assert {"tas", "crs"}.issubset(set(f.data_vars))
            subset.subset_shape(ds, self.meridian_multi_geojson)

    def test_no_wraps(self, tmp_netcdf_filename):
        ds = xr.open_dataset(self.nc_file)

        with pytest.warns(None) as record:  # noqa
            sub = subset.subset_shape(ds, self.poslons_geojson)

        self.compare_vals(ds, sub, "tas")

        # No time subsetting should occur.
        assert len(sub.tas) == 12

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            float(np.mean(sub.tas.isel(time=0))), 276.732, 3
        )
        # Check that no warnings are raised for meridian crossing
        assert (
            '"Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."'
            '" This feature is experimental. Output might not be accurate."'
            not in [str(q.message) for q in record]
        )

        assert sub.crs.prime_meridian_name == "Greenwich"
        assert sub.crs.grid_mapping_name == "latitude_longitude"

        sub.to_netcdf(tmp_netcdf_filename)
        assert tmp_netcdf_filename.exists()
        with xr.open_dataset(filename_or_obj=tmp_netcdf_filename) as f:
            assert {"tas", "crs"}.issubset(set(f.data_vars))
            subset.subset_shape(ds, self.poslons_geojson)

    def test_all_neglons(self):
        ds = xr.open_dataset(self.nc_file_neglons)

        with pytest.warns(None) as record:  # noqa
            sub = subset.subset_shape(ds, self.southern_qc_geojson)

        self.compare_vals(ds, sub, "tasmax")

        # Average temperature at surface for region in January (time=0)
        np.testing.assert_array_almost_equal(
            float(np.mean(sub.tasmax.isel(time=0))), 269.254, 3
        )
        # Check that no warnings are raised for meridian crossing
        assert (
            '"Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."'
            '" This feature is experimental. Output might not be accurate."'
            not in [q.message for q in record]
        )

    def test_rotated_pole_with_time(self):
        ds = xr.open_dataset(self.lons_2d_nc_file)

        with pytest.warns(None) as record:  # noqa
            sub = subset.subset_shape(
                ds,
                self.eastern_canada_geojson,
                start_date="1984-06-01",
                end_date="1984-06-15",
            )

        self.compare_vals(
            ds.sel(time=slice("1984-06-01", "1984-06-15")), sub, "tasmax", flag_2d=True
        )

        # Should only have 15 days of data.
        assert len(sub.tasmax) == 15
        # Average max temperature at surface for region on June 1st, 1984 (time=0)
        np.testing.assert_allclose(float(np.mean(sub.tasmax.isel(time=0))), 289.634968)
        # Check that no warnings are raised for meridian crossing
        assert (
            '"Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."'
            '" This feature is experimental. Output might not be accurate."'
            not in [str(q.message) for q in record]
        )

    def test_small_poly_buffer(self, tmp_netcdf_filename):
        ds = xr.open_dataset(self.nc_file)

        with pytest.raises(ValueError):
            subset.subset_shape(ds, self.small_geojson)

        with pytest.raises(ValueError):
            subset.subset_shape(ds, self.small_geojson, buffer=0.6)

        sub = subset.subset_shape(ds, self.small_geojson, buffer=5)
        self.compare_vals(ds, sub, "tas")
        assert len(sub.lon.values) == 3
        assert len(sub.lat.values) == 3

        assert sub.crs.prime_meridian_name == "Greenwich"
        assert sub.crs.grid_mapping_name == "latitude_longitude"

        sub.to_netcdf(tmp_netcdf_filename)
        assert tmp_netcdf_filename.exists()
        with xr.open_dataset(filename_or_obj=tmp_netcdf_filename) as f:
            assert {"tas", "crs"}.issubset(set(f.data_vars))

    def test_mask_multiregions(self, toggle_pygeos):
        ds = xr.open_dataset(self.nc_file)
        regions = gpd.read_file(self.multi_regions_geojson)
        regions.set_index("id")
        mask = subset.create_mask(
            x_dim=ds.lon, y_dim=ds.lat, poly=regions, wrap_lons=True
        )
        vals, counts = np.unique(mask.values[mask.notnull()], return_counts=True)
        np.testing.assert_array_equal(vals, [0, 1, 2])
        np.testing.assert_array_equal(counts, [58, 250, 22])

    @pytest.mark.skipif(
        xesmf is None, reason="xESMF >= 0.6.2 is needed for average_shape."
    )
    def test_weight_masks_multiregions(self):
        # rename is due to a small limitation of xESMF 0.5.2
        ds = xr.open_dataset(self.nc_file).rename(bnds="bounds")
        regions = gpd.read_file(self.multi_regions_geojson).set_index("id")
        masks = subset.create_weight_masks(ds, poly=regions)

        np.testing.assert_allclose(masks.sum(["lat", "lon"]), [1, 1, 1])
        np.testing.assert_array_equal(masks.geom.values, regions.index)
        np.testing.assert_allclose(masks.max("geom").sum(), 2.900, 3)

    def test_subset_multiregions(self):
        ds = xr.open_dataset(self.nc_file)
        regions = gpd.read_file(self.multi_regions_geojson)
        regions.set_index("id")
        ds_sub = subset.subset_shape(ds, shape=regions)
        assert ds_sub.notnull().sum() == 58 + 250 + 22
        assert ds_sub.tas.shape == (12, 14, 128)

    def test_vectorize_touches_polygons(self, toggle_pygeos):
        """Check that points touching the polygon are included in subset."""
        # Create simple polygon
        poly = Polygon([[0, 0], [1, 0], [1, 1]])
        shape = gpd.GeoDataFrame(geometry=[poly])
        # Create simple data array
        da = xr.DataArray(
            data=[[0, 1], [2, 3]],
            dims=("lon", "lat"),
            coords={"lon": [0, 1], "lat": [0, 1]},
        )
        sub = subset.subset_shape(da, shape=shape)
        assert sub.notnull().sum() == 3


class TestDistance:
    def test_values(self):
        # Check values are OK. Values taken from pyproj test.
        boston_lat = 42.0 + (15.0 / 60.0)
        boston_lon = -71.0 - (7.0 / 60.0)
        portland_lat = 45.0 + (31.0 / 60.0)
        portland_lon = -123.0 - (41.0 / 60.0)

        da = xr.DataArray(
            0, coords={"lon": [boston_lon], "lat": [boston_lat]}, dims=["lon", "lat"]
        )
        d = subset.distance(da, lon=portland_lon, lat=portland_lat)
        np.testing.assert_almost_equal(d, 4164074.239, decimal=3)

    def test_broadcasting(self):
        # Check output dimensions match lons and lats.
        lon = np.linspace(-180, 180, 20)
        lat = np.linspace(-90, 90, 30)
        da = xr.Dataset(data_vars=None, coords={"lon": lon, "lat": lat})
        da["data"] = xr.DataArray(
            np.random.rand(lon.size, lat.size), dims=["lon", "lat"]
        )

        d = subset.distance(da, lon=-34, lat=56).squeeze("site")
        assert d.dims == da.data.dims
        assert d.shape == da.data.shape
        assert d.units == "m"

        # Example of how to get the actual 2D indices.
        k = d.argmin()
        i, j = np.unravel_index(k, da.data.shape)
        assert d[i, j] == d.min()


class TestSubsetLevel:
    nc_plev = get_file(
        "cmip6/o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc"
    )
    plevs = [
        100000,
        92500,
        85000,
        70000,
        60000,
        50000,
        40000,
        30000,
        25000,
        20000,
        15000,
        10000,
        7000,
        5000,
        3000,
        2000,
        1000,
        500,
        100,
    ]

    def test_simple(self):
        da = xr.open_dataset(self.nc_plev).o3
        lev_st = 100000.0
        lev_ed = 100.0

        out = subset.subset_level(da, first_level=lev_st, last_level=lev_ed)
        out1 = subset.subset_level(da, first_level=f"{lev_st}", last_level=f"{lev_ed}")
        np.testing.assert_array_equal(out, out1)
        np.testing.assert_array_equal(out.plev.min(), lev_ed)
        np.testing.assert_array_equal(out.plev.max(), lev_st)

    def test_level_outofbounds(self):
        da = xr.open_dataset(self.nc_plev).o3
        lev_st = 10000000
        lev_ed = 10

        with pytest.warns(None) as record:  # noqa
            out = subset.subset_level(da, first_level=lev_st, last_level=lev_ed)

        np.testing.assert_array_equal(out.plev.min(), da.plev.min())
        np.testing.assert_array_equal(out.plev.max(), da.plev.max())

        assert (
            '"first_level" has been nudged to nearest valid level in xarray object.'
            in [str(q.message) for q in record]
        )
        assert (
            '"last_level" has been nudged to nearest valid level in xarray object.'
            in [str(q.message) for q in record]
        )

    def test_warnings(self):
        da = xr.open_dataset(self.nc_plev).o3

        with pytest.raises(TypeError):
            subset.subset_level(da, first_level=da, last_level=da)

        with pytest.warns(None) as record:  # noqa
            subset.subset_level(da, first_level="1000", last_level="1000")
        assert (
            '"first_level" should be a number, it has been converted to a float.'
            in [str(q.message) for q in record]
        )

        with pytest.warns(None) as record:  # noqa
            subset.subset_level(da, first_level=81200, last_level=54100.6)

        msgs = {
            '"first_level" has been nudged to nearest valid level in xarray object.',
            '"last_level" has been nudged to nearest valid level in xarray object.',
        }
        assert msgs.issubset({str(q.message) for q in record})

    def test_level_first_only(self):
        da = xr.open_dataset(self.nc_plev).o3
        lev_st = 100000

        # first level only
        with pytest.warns(None):  # noqa
            out = subset.subset_level(da, first_level=lev_st, last_level=lev_st)

        np.testing.assert_array_equal(out.plev.values[0], da.plev.values[0])
        np.testing.assert_array_equal(out.plev.values[-1], da.plev.values[0])

    def test_nudge_levels(self):
        da = xr.open_dataset(self.nc_plev).o3
        # good_levels = [40000, 30000]
        # good_indices = [6, 7]

        with pytest.warns(None):  # noqa
            out = subset.subset_level(da, first_level=37000.2342, last_level=27777)

        np.testing.assert_array_equal(out.plev.values[0], da.plev.values[7])
        np.testing.assert_array_equal(out.plev.values[-1], da.plev.values[7])

        with pytest.warns(None):  # noqa
            out = subset.subset_level(da, first_level=41562, last_level=29999)

        np.testing.assert_array_equal(out.plev.values[:], da.plev.values[6:8])


class TestGridPolygon:
    def test_rectilinear(self):
        pytest.importorskip("xesmf", "0.6.2")
        # CF-Compliant with bounds
        ds = xesmf.util.cf_grid_2d(-200, -100, 20, -60, 60, 10)
        poly = subset._rectilinear_grid_exterior_polygon(ds)
        assert poly == Polygon([(-200, -60), (-200, 60), (-100, 60), (-100, -60)])

        # Without bounds
        ds.lon.attrs.pop("bounds")
        ds.lat.attrs.pop("bounds")
        ds = xr.Dataset(dict(lon=ds.lon, lat=ds.lat))
        poly = subset._rectilinear_grid_exterior_polygon(ds)
        assert poly == Polygon([(-200, -60), (-200, 60), (-100, 60), (-100, -60)])

    @pytest.mark.parametrize("mode", ["bbox"])
    def test_curvilinear(self, mode):
        """
        Note that there is at least one error in the lat and lon vertices.
        >>> print(ds.vertices_latitude[0, 283].data)
        [-78.29248047  81.48125458 - 78.49452209 - 78.49452209]
        >>> print(ds.vertices_longitude[0, 283].data)
        [357.  73. 356. 356.]
        """
        from shapely.geometry import MultiPoint, Point

        fn = get_file("cmip6/sic_SImon_CCCma-CanESM5_ssp245_r13i1p2f1_2020.nc")
        ds = xr.open_dataset(fn)

        poly = subset._curvilinear_grid_exterior_polygon(ds, mode=mode)

        # Check that all grid centroids are within the polygon
        pts = list(map(Point, zip(ds.longitude.data.flat, ds.latitude.data.flat)))
        assert MultiPoint(pts).within(poly)


class TestShapeBboxIndexer:
    def test_rectilinear(self):
        # Create small polygon fitting in one cell.
        pytest.importorskip("xesmf", "0.6.2")
        x, y = -150, 35
        p = Point(x, y)
        ds = xesmf.util.cf_grid_2d(-200, 0, 20, -60, 60, 10)

        # Confirm that after subsetting, the polygon is still entirely within the grid.
        for b in [1, 10, 20]:
            pb = p.buffer(b)
            inds = subset.shape_bbox_indexer(ds, gpd.GeoDataFrame(geometry=[pb]))
            assert pb.within(subset.grid_exterior_polygon(ds.isel(inds)))

    def test_curvilinear(self):
        """This checks that a grid along lon/lat and a rotated grid are indexed identically for a geometry and a
        rotated geometry."""
        pytest.importorskip("xesmf", "0.6.2")
        from shapely.affinity import rotate

        ds = xesmf.util.grid_2d(0, 100, 10, 0, 60, 6)
        rds = rotated_grid_2d(0, 100, 10, 0, 60, 6, angle=45)

        geom = Polygon(([0, 0], [50, 0], [50, 30], [0, 30]))
        rgeom = rotate(geom, 45, origin=Point(0, 0))

        # The subsetted grid should have the same dimensions as the subsetted rotated grid.
        i = subset.shape_bbox_indexer(ds, gpd.GeoDataFrame(geometry=[geom]))
        ri = subset.shape_bbox_indexer(rds, gpd.GeoDataFrame(geometry=[rgeom]))
        assert ri == i


def rotated_grid_2d(lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat, angle):
    """Rotate lat lon by degree"""
    ds = xesmf.util.grid_2d(lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat)

    # Rotation matrix
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))

    fds = ds.stack(z=("x", "y"), z_b=("x_b", "y_b"))
    fds.lon[:], fds.lat[:] = np.matmul(r, xr.concat([fds.lon, fds.lat], dim="c").data)
    fds.lon_b[:], fds.lat_b[:] = np.matmul(
        r, xr.concat([fds.lon_b, fds.lat_b], dim="c").data
    )

    return fds.unstack(("z", "z_b"))
