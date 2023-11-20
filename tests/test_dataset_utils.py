import cf_xarray as cfxr  # noqa
import numpy as np
import packaging.version
import pytest
import xarray as xr
from packaging.version import Version
from roocs_grids import get_grid_file

import clisops.utils.dataset_utils as clidu
from _common import (
    C3S_CORDEX_AFR_TAS,
    C3S_CORDEX_ANT_SFC_WIND,
    CMIP6_GFDL_EXTENT,
    CMIP6_IITM_EXTENT,
    CMIP6_OCE_HALO_CNRM,
    CMIP6_SICONC,
    CMIP6_TAS_ONE_TIME_STEP,
    CMIP6_TAS_PRECISION_A,
    CMIP6_TOS_ONE_TIME_STEP,
    CMIP6_UNSTR_ICON_A,
    CORDEX_TAS_ONE_TIMESTEP,
)
from clisops.core.regrid import XESMF_MINIMUM_VERSION

try:
    import xesmf

    if Version(xesmf.__version__) < Version(XESMF_MINIMUM_VERSION):
        raise ImportError()
except ImportError:
    xesmf = None

XESMF_IMPORT_MSG = (
    f"xESMF >= {XESMF_MINIMUM_VERSION} is needed for regridding functionalities."
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


def test_add_hor_CF_coord_attrs():
    "Test function to add standard attributes to horizontal coordinate variables."
    # Create basic dataset
    ds = xr.Dataset(
        data_vars={},
        coords={
            "lat": (["lat"], np.ones(1)),
            "lon": (["lon"], np.ones(1)),
            "lat_bnds": (["lat", "bnds"], np.ones((1, 2))),
            "lon_bnds": (["lon", "bnds"], np.ones((1, 2))),
        },
    )

    # Ensuring attributes have been added
    ds = clidu.add_hor_CF_coord_attrs(ds=ds)
    assert ds["lat"].attrs["bounds"] == "lat_bnds"
    assert ds["lon"].attrs["bounds"] == "lon_bnds"
    assert ds["lat"].attrs["units"] == "degrees_north"
    assert ds["lon"].attrs["units"] == "degrees_east"
    assert ds["lat"].attrs["axis"] == "Y"
    assert ds["lon"].attrs["axis"] == "X"
    assert ds["lat"].attrs["standard_name"] == "latitude"
    assert ds["lon"].attrs["standard_name"] == "longitude"

    # Ensuring attributes have been updated (and conflicting ones overwritten)
    ds["lat"].attrs["bounds"] = "lat_b"
    ds["lon"].attrs["standard_name"] = "lon"
    ds["lat_bnds"].attrs["custom"] = "custom"
    ds = clidu.add_hor_CF_coord_attrs(ds=ds, keep_attrs=True)
    assert ds["lat"].attrs["bounds"] == "lat_bnds"
    assert ds["lon"].attrs["bounds"] == "lon_bnds"
    assert ds["lat"].attrs["units"] == "degrees_north"
    assert ds["lon"].attrs["units"] == "degrees_east"
    assert ds["lat"].attrs["axis"] == "Y"
    assert ds["lon"].attrs["axis"] == "X"
    assert ds["lat"].attrs["standard_name"] == "latitude"
    assert ds["lon"].attrs["standard_name"] == "longitude"
    assert ds["lat_bnds"].attrs["custom"] == "custom"

    # Incorrect coordinate variable name supplied should lead to a KeyError
    with pytest.raises(KeyError) as exc:
        ds = clidu.add_hor_CF_coord_attrs(ds, lat="latitude")
    assert (
        str(exc.value)
        == "'Not all specified coordinate variables exist in the dataset.'"
    )


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_reformat_xESMF_to_CF():
    "Test reformat operation reformat_xESMF_to_CF"
    # Use xesmf utility function to create dataset with global grid
    ds = xesmf.util.grid_global(5.0, 5.0)

    # It should have certain variables defined
    assert all([coord in ds for coord in ["lat", "lon", "lat_b", "lon_b"]])
    assert all([dim in ds.dims for dim in ["x", "y", "x_b", "y_b"]])

    # Reformat
    ds.attrs["xesmf"] = xesmf.__version__
    ds_ref = clidu.reformat_xESMF_to_CF(ds=ds, keep_attrs=True)
    assert all([coord in ds_ref for coord in ["lat", "lon", "lat_bnds", "lon_bnds"]])
    assert all([dim in ds_ref.dims for dim in ["lat", "lon", "bnds"]])
    assert ds_ref.attrs["xesmf"] == xesmf.__version__


def test_reformat_SCRIP_to_CF():
    "Test reformat operation reformat_SCRIP_to_CF"
    # Load dataset in SCRIP format (using roocs_grids)
    ds = xr.open_dataset(get_grid_file("2pt5deg"))

    # It should have certain variables defined
    assert all(
        [
            coord in ds
            for coord in [
                "grid_center_lat",
                "grid_center_lon",
                "grid_corner_lat",
                "grid_corner_lon",
                "grid_dims",
                "grid_area",
                "grid_imask",
            ]
        ]
    )
    assert all([dim in ds.dims for dim in ["grid_corners", "grid_size", "grid_rank"]])

    # Reformat
    ds_ref = clidu.reformat_SCRIP_to_CF(ds=ds, keep_attrs=True)
    assert all([coord in ds_ref for coord in ["lat", "lon", "lat_bnds", "lon_bnds"]])
    assert all([dim in ds_ref.dims for dim in ["lat", "lon", "bnds"]])
    assert ds_ref.attrs["Conventions"] == "SCRIP"


def test_detect_shape_regular():
    "Test detect_shape function for a regular grid"
    # Load dataset
    ds = xr.open_dataset(get_grid_file("0pt25deg_era5_lsm"))

    # Detect shape
    nlat, nlon, ncells = clidu.detect_shape(
        ds, lat="latitude", lon="longitude", grid_type="regular_lat_lon"
    )

    # Assertion
    assert nlat == 721
    assert nlon == 1440
    assert ncells == nlat * nlon


def test_detect_shape_unstructured():
    "Test detect_shape function for an unstructured grid"
    # Load dataset
    ds = xr.open_dataset(CMIP6_UNSTR_ICON_A, use_cftime=True)

    # Detect shape
    nlat, nlon, ncells = clidu.detect_shape(
        ds, lat="latitude", lon="longitude", grid_type="unstructured"
    )

    # Assertion
    assert nlat == ncells
    assert nlon == ncells
    assert ncells == 20480


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_detect_format():
    "Test detect_format function"
    # Load/Create datasets in SCRIP, CF and xESMF format
    ds_cf = xr.open_dataset(get_grid_file("0pt25deg_era5_lsm"))
    ds_scrip = xr.open_dataset(get_grid_file("0pt25deg_era5"))
    ds_xesmf = xesmf.util.grid_global(5.0, 5.0)

    # Assertion
    assert clidu.detect_format(ds_cf) == "CF"
    assert clidu.detect_format(ds_scrip) == "SCRIP"
    assert clidu.detect_format(ds_xesmf) == "xESMF"


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
    """Test the function detect_gridtype."""
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
    """Test the _crosses_0_meridian function"""
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
    """Test the helper function _convert_interval_between_lon_frames"""
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
    """Test the function cf_convert_between_lon_frames"""
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

    # Check that bounds contain the respective cell centers
    if packaging.version.parse(xr.__version__) >= packaging.version.parse("2022.11.0"):
        assert np.all(
            conv["lon"].values[:] > conv["lon_bounds"].values[:, 0]
        )  # xarray >=2022.11.0 reorders dim values
        assert np.all(
            conv["lon"].values[:] < conv["lon_bounds"].values[:, 1]
        )  # xarray >=2022.11.0 reorders dim values
    else:
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


def test_calculate_bounds_curvilinear(load_esgf_test_data):
    "Test for bounds calculation for curvilinear grid"

    # Load CORDEX dataset (curvilinear grid)
    ds = xr.open_dataset(CORDEX_TAS_ONE_TIMESTEP).isel(
        {"rlat": range(100, 120), "rlon": range(100, 120)}
    )

    # Drop bounds variables and compute them
    ds_nb = ds.drop_vars(["lon_vertices", "lat_vertices"])
    ds_nb = clidu.generate_bounds_curvilinear(ds_nb, lat="lat", lon="lon")

    # Sort every cells vertices values
    for i in range(1, 19):
        for j in range(1, 19):
            ds.lat_vertices[i, j, :] = np.sort(ds.lat_vertices.values[i, j, :])
            ds.lon_vertices[i, j, :] = np.sort(ds.lon_vertices.values[i, j, :])
            ds_nb.lat_bnds[i, j, :] = np.sort(ds_nb.lat_bnds.values[i, j, :])
            ds_nb.lon_bnds[i, j, :] = np.sort(ds_nb.lon_bnds.values[i, j, :])

    # Assert all values are close (discard cells at edge of selected grid area)
    xr.testing.assert_allclose(
        ds.lat_vertices.isel({"rlat": range(1, 19), "rlon": range(1, 19)}),
        ds_nb.lat_bnds.isel({"rlat": range(1, 19), "rlon": range(1, 19)}),
        rtol=1e-06,
        atol=0,
    )
    xr.testing.assert_allclose(
        ds.lon_vertices.isel({"rlat": range(1, 19), "rlon": range(1, 19)}),
        ds_nb.lon_bnds.isel({"rlat": range(1, 19), "rlon": range(1, 19)}),
        rtol=1e-06,
        atol=0,
    )


def test_calculate_bounds_rectilinear(load_esgf_test_data):
    "Test for bounds calculation for rectilinear grid"

    # Load CORDEX dataset (curvilinear grid)
    ds = xr.open_dataset(CMIP6_TAS_PRECISION_A)

    # Drop bounds variables and compute them
    ds_nb = ds.drop_vars(["lon_bnds", "lat_bnds"])
    ds_nb = clidu.generate_bounds_rectilinear(ds_nb, lat="lat", lon="lon")

    # Assert all values are close
    xr.testing.assert_allclose(ds.lat_bnds, ds_nb.lat_bnds, rtol=1e-06, atol=0)
    xr.testing.assert_allclose(ds.lon_bnds, ds_nb.lon_bnds, rtol=1e-06, atol=0)
