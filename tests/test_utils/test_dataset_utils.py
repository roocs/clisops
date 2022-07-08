import cf_xarray as cfxr
import numpy as np
import pytest
import xarray as xr

import clisops.utils.dataset_utils as clidu

from .._common import (
    C3S_CORDEX_AFR_TAS,
    C3S_CORDEX_ANT_SFC_WIND,
    CMIP6_GFDL_EXTENT,
    CMIP6_IITM_EXTENT,
    CMIP6_OCE_HALO_CNRM,
    CMIP6_SICONC,
    CMIP6_TAS_ONE_TIME_STEP,
    CMIP6_TOS_ONE_TIME_STEP,
    CMIP6_UNSTR_ICON_A,
    CORDEX_TAS_ONE_TIMESTEP,
)


def test_add_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-29T00:00:00"

    new_date = clidu.adjust_date_to_calendar(da, date, "forwards")

    assert new_date == "2012-03-01T00:00:00"


def test_sub_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-30T00:00:00"

    new_date = clidu.adjust_date_to_calendar(da, date, "backwards")

    assert new_date == "2012-02-28T00:00:00"


def test_invalid_day():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-02-29T00:00:00"

    with pytest.raises(Exception) as exc:
        clidu.adjust_date_to_calendar(da, date, "odd")
    assert (
        str(exc.value)
        == "Invalid value for direction: odd. This should be either 'backwards' to indicate subtracting a day or 'forwards' for adding a day."
    )


def test_date_out_of_expected_range():
    da = xr.open_dataset(CMIP6_SICONC, use_cftime=True)
    date = "2012-00-01T00:00:00"

    with pytest.raises(Exception) as exc:
        clidu.adjust_date_to_calendar(da, date, "forwards")
    assert (
        str(exc.value) == "Invalid input 0 for month. Expected value between 1 and 12."
    )


def test_detect_coordinate_and_bounds():
    "Test detect_bounds and detect_coordinate functions."
    ds_a = xr.open_mfdataset(C3S_CORDEX_AFR_TAS, use_cftime=True, combine="by_coords")
    ds_b = xr.open_mfdataset(
        C3S_CORDEX_ANT_SFC_WIND, use_cftime=True, combine="by_coords"
    )
    ds_c = xr.open_dataset(CMIP6_UNSTR_ICON_A)
    ds_d = xr.open_dataset(CMIP6_OCE_HALO_CNRM)

    # check lat, lon are found
    lat_a = clidu.detect_coordinate(ds_a, "latitude")
    lon_a = clidu.detect_coordinate(ds_a, "longitude")
    lat_b = clidu.detect_coordinate(ds_b, "latitude")
    lon_b = clidu.detect_coordinate(ds_b, "longitude")
    lat_c = clidu.detect_coordinate(ds_c, "latitude")
    lon_c = clidu.detect_coordinate(ds_c, "longitude")
    lat_d = clidu.detect_coordinate(ds_d, "latitude")
    lon_d = clidu.detect_coordinate(ds_d, "longitude")

    # assert the correct variables have been found
    assert lat_a == "lat"
    assert lon_a == "lon"
    assert lat_b == "lat"
    assert lon_b == "lon"
    assert lat_c == "latitude"
    assert lon_c == "longitude"
    assert lat_d == "lat"
    assert lon_d == "lon"

    # assert detected bounds
    assert clidu.detect_bounds(ds_a, lat_a) == "lat_vertices"
    assert clidu.detect_bounds(ds_a, lon_a) == "lon_vertices"
    assert clidu.detect_bounds(ds_b, lat_b) is None
    assert clidu.detect_bounds(ds_b, lon_b) is None
    assert clidu.detect_bounds(ds_c, lat_c) == "latitude_bnds"
    assert clidu.detect_bounds(ds_c, lon_c) == "longitude_bnds"
    assert clidu.detect_bounds(ds_d, lat_d) == "lat_bnds"
    assert clidu.detect_bounds(ds_d, lon_d) == "lon_bnds"

    # test that latitude and longitude are still found when they are data variables
    # reset coords sets lat and lon as data variables
    ds_a = ds_a.reset_coords([lat_a, lon_a])
    ds_b = ds_b.reset_coords([lat_b, lon_b])
    ds_c = ds_c.reset_coords([lat_c, lon_c])
    ds_d = ds_d.reset_coords([lat_d, lon_d])
    assert lat_a == clidu.detect_coordinate(ds_a, "latitude")
    assert lon_a == clidu.detect_coordinate(ds_a, "longitude")
    assert lat_b == clidu.detect_coordinate(ds_b, "latitude")
    assert lon_b == clidu.detect_coordinate(ds_b, "longitude")
    assert lat_c == clidu.detect_coordinate(ds_c, "latitude")
    assert lon_c == clidu.detect_coordinate(ds_c, "longitude")
    assert lat_d == clidu.detect_coordinate(ds_d, "latitude")
    assert lon_d == clidu.detect_coordinate(ds_d, "longitude")


def test_detect_gridtype():
    "Test the function detect_gridtype"
    ds_a = xr.open_dataset(CMIP6_UNSTR_ICON_A, use_cftime=True)
    ds_b = xr.open_dataset(CMIP6_TOS_ONE_TIME_STEP, use_cftime=True)
    ds_c = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
    assert (
        clidu.detect_gridtype(
            ds_a,
            lat="latitude",
            lon="longitude",
            lat_bnds="latitude_bnds",
            lon_bnds="longitude_bnds",
        )
        == "unstructured"
    )
    assert (
        clidu.detect_gridtype(
            ds_b,
            lat="latitude",
            lon="longitude",
            lat_bnds="vertices_latitude",
            lon_bnds="vertices_longitude",
        )
        == "curvilinear"
    )
    assert (
        clidu.detect_gridtype(
            ds_c, lat="lat", lon="lon", lat_bnds="lat_bnds", lon_bnds="lon_bnds"
        )
        == "regular_lat_lon"
    )


def test_crosses_0_meridian():
    "Test the _crosses_0_meridian function"
    # Case 1 - longitude crossing 180° meridian
    lon = np.arange(160.0, 200.0, 1.0)

    # convert to [-180, 180], min and max now suggest 0-meridian crossing
    lon = np.where(lon > 180, lon - 360, lon)

    da = xr.DataArray(dims=["x"], coords={"x": lon})
    assert not clidu._crosses_0_meridian(da["x"])

    # Case 2 - regional dataset ranging [315 .. 66] but for whatever reason not defined on
    #          [-180, 180] longitude frame
    ds = xr.open_dataset(CORDEX_TAS_ONE_TIMESTEP)
    assert np.isclose(ds["lon"].min(), 0, atol=0.1)
    assert np.isclose(ds["lon"].max(), 360, atol=0.1)

    # Convert to -180, 180 frame and confirm crossing 0-meridian
    ds, ll, lu = clidu.cf_convert_between_lon_frames(ds, (-180, 180))
    assert np.isclose(ds["lon"].min(), -45.4, atol=0.1)
    assert np.isclose(ds["lon"].max(), 66.1, atol=0.1)
    assert clidu._crosses_0_meridian(ds["lon"])


def test_convert_interval_between_lon_frames():
    "Test the helper function _convert_interval_between_lon_frames"
    # Convert from 0,360 to -180,180 longitude frame and vice versa
    assert clidu._convert_interval_between_lon_frames(20, 60) == (20, 60)
    assert clidu._convert_interval_between_lon_frames(190, 200) == (-170, -160)
    assert clidu._convert_interval_between_lon_frames(-20, -90) == (270, 340)

    # Exception when crossing 0°- or 180°-meridian
    with pytest.raises(
        Exception,
        match="Cannot convert longitude interval if it includes the 0°- or 180°-meridian.",
    ):
        clidu._convert_interval_between_lon_frames(170, 300)
    with pytest.raises(
        Exception,
        match="Cannot convert longitude interval if it includes the 0°- or 180°-meridian.",
    ):
        clidu._convert_interval_between_lon_frames(-30, 10)


def test_convert_lon_frame_bounds():
    "Test the function cf_convert_between_lon_frames"
    # Load tutorial dataset defined on [200,330]
    ds = xr.tutorial.open_dataset("air_temperature")
    assert ds["lon"].min() == 200.0
    assert ds["lon"].max() == 330.0

    # Create bounds
    dsb = ds.cf.add_bounds("lon")

    # Convert to other lon frame
    conv, ll, lu = clidu.cf_convert_between_lon_frames(dsb, (-180, 180))

    assert conv["lon"].values[0] == -160.0
    assert conv["lon"].values[-1] == -30.0

    # Check bounds are containing the respective cell centers
    assert np.all(conv["lon"].values[:] > conv["lon_bounds"].values[0, :])
    assert np.all(conv["lon"].values[:] < conv["lon_bounds"].values[1, :])

    # Convert only lon_interval
    conv, ll, lu = clidu.cf_convert_between_lon_frames(dsb, (-180, -10))

    assert conv["lon"].min() == 200.0
    assert conv["lon"].max() == 330.0
    assert ll == 180.0
    assert lu == 350.0


def test_convert_lon_frame_shifted_bounds():
    ds = xr.open_dataset(CMIP6_GFDL_EXTENT, use_cftime=True)

    # confirm shifted frame
    assert np.isclose(ds["lon"].min(), -300.0, atol=0.5)
    assert np.isclose(ds["lon"].max(), 60.0, atol=0.5)

    # convert to [-180, 180]
    ds_a, ll, lu = clidu.cf_convert_between_lon_frames(ds, (-180, 180))
    assert (ll, lu) == (-180, 180)
    assert np.isclose(ds_a["lon"].min(), -180.0, atol=0.5)
    assert np.isclose(ds_a["lon"].max(), 180.0, atol=0.5)
    assert np.isclose(ds_a["lon_bnds"].min(), -180.0, atol=0.5)
    assert np.isclose(ds_a["lon_bnds"].max(), 180.0, atol=0.5)

    # convert to [0, 360]
    ds_b, ll, lu = clidu.cf_convert_between_lon_frames(ds, (0, 360))
    assert (ll, lu) == (0, 360)
    assert np.isclose(ds_b["lon"].min(), 0.0, atol=0.5)
    assert np.isclose(ds_b["lon"].max(), 360.0, atol=0.5)
    assert np.isclose(ds_b["lon_bnds"].min(), 0.0, atol=0.5)
    assert np.isclose(ds_b["lon_bnds"].max(), 360.0, atol=0.5)

    # convert intermediate result to [0, 360]
    ds_c, ll, lu = clidu.cf_convert_between_lon_frames(ds_a, (0, 360))
    assert (ll, lu) == (0, 360)
    assert np.isclose(ds_c["lon"].min(), 0.0, atol=0.5)
    assert np.isclose(ds_c["lon"].max(), 360.0, atol=0.5)
    assert np.isclose(ds_c["lon_bnds"].min(), 0.0, atol=0.5)
    assert np.isclose(ds_c["lon_bnds"].max(), 360.0, atol=0.5)

    # convert intermediate result to [-180, 180]
    ds_d, ll, lu = clidu.cf_convert_between_lon_frames(ds_a, (-180, 180))
    assert (ll, lu) == (-180, 180)
    assert np.isclose(ds_d["lon"].min(), -180.0, atol=0.5)
    assert np.isclose(ds_d["lon"].max(), 180.0, atol=0.5)
    assert np.isclose(ds_d["lon_bnds"].min(), -180.0, atol=0.5)
    assert np.isclose(ds_d["lon_bnds"].max(), 180.0, atol=0.5)

    # assert projection coordinate sorted
    assert np.all(ds_d["x"].values[1:] - ds_d["x"].values[:-1] > 0.0)
    assert np.all(ds_c["x"].values[1:] - ds_c["x"].values[:-1] > 0.0)


def test_convert_lon_frame_shifted_no_bounds():
    ds = xr.open_dataset(CMIP6_IITM_EXTENT, use_cftime=True)

    # confirm shifted frame
    assert np.isclose(ds["longitude"].min(), -280.0, atol=1.0)
    assert np.isclose(ds["longitude"].max(), 80.0, atol=1.0)

    # convert to [-180, 180]
    ds_a, ll, lu = clidu.cf_convert_between_lon_frames(ds, (-180, 180))
    assert (ll, lu) == (-180, 180)
    assert np.isclose(ds_a["longitude"].min(), -180.0, atol=1.0)
    assert np.isclose(ds_a["longitude"].max(), 180.0, atol=1.0)

    # convert to [0, 360]
    ds_b, ll, lu = clidu.cf_convert_between_lon_frames(ds, (0, 360))
    assert (ll, lu) == (0, 360)
    assert np.isclose(ds_b["longitude"].min(), 0.0, atol=1.0)
    assert np.isclose(ds_b["longitude"].max(), 360.0, atol=1.0)

    # convert intermediate result to [0, 360]
    ds_c, ll, lu = clidu.cf_convert_between_lon_frames(ds_a, (0, 360))
    assert (ll, lu) == (0, 360)
    assert np.isclose(ds_c["longitude"].min(), 0.0, atol=1.0)
    assert np.isclose(ds_c["longitude"].max(), 360.0, atol=1.0)

    # convert intermediate result to [-180, 180]
    ds_d, ll, lu = clidu.cf_convert_between_lon_frames(ds_a, (-180, 180))
    assert (ll, lu) == (-180, 180)
    assert np.isclose(ds_d["longitude"].min(), -180.0, atol=1.0)
    assert np.isclose(ds_d["longitude"].max(), 180.0, atol=1.0)

    # assert projection coordinate sorted
    assert np.all(ds_d["x"].values[1:] - ds_d["x"].values[:-1] > 0.0)
    assert np.all(ds_c["x"].values[1:] - ds_c["x"].values[:-1] > 0.0)


# todo: add a few more tests of cf_convert_lon_frame using xe.util functions to create regional and global datasets
