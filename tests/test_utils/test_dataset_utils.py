import numpy as np
import pytest
import xarray as xr
from pkg_resources import parse_version
from roocs_grids import get_grid_file

import clisops.utils.dataset_utils as clidu
from clisops.core.regrid import XESMF_MINIMUM_VERSION

from .._common import CMIP6_SICONC, CMIP6_UNSTR_ICON_A

try:
    import xesmf

    if parse_version(xesmf.__version__) < parse_version(XESMF_MINIMUM_VERSION):
        raise ImportError
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


def test_detect_shape_irregular():
    "Test detect_shape function for an irregular grid"
    # Load dataset
    ds = xr.open_dataset(CMIP6_UNSTR_ICON_A, use_cftime=True)

    # Detect shape
    nlat, nlon, ncells = clidu.detect_shape(
        ds, lat="latitude", lon="longitude", grid_type="irregular"
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
