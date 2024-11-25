import os
from glob import glob
from pathlib import Path

import cf_xarray  # noqa
import numpy as np
import pytest
import xarray as xr
from packaging.version import Version
from roocs_grids import get_grid_file

import clisops.utils.dataset_utils as clidu
from clisops import CONFIG
from clisops.core.regrid import (
    XESMF_MINIMUM_VERSION,
    Grid,
    Weights,
    regrid,
    weights_cache_flush,
    weights_cache_init,
)
from clisops.ops.subset import subset
from clisops.utils.output_utils import FileLock

try:
    import xesmf

    if Version(xesmf.__version__) < Version(XESMF_MINIMUM_VERSION):
        raise ImportError("xesmf version is too old")
except ImportError:
    xesmf = None


# test from grid_id --predetermined
# test different types of grid e.g. unstructured, not supported type
# test for errors e.g.
# no lat/lon in dataset
# more than one latitude/longitude
# grid instructor tuple not correct length


XESMF_IMPORT_MSG = (
    f"xesmf >= {XESMF_MINIMUM_VERSION} is needed for regridding functionalities"
)


def test_grid_init_ds_tas_regular(mini_esgf_data):
    with xr.open_dataset(
        mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True
    ) as ds:
        grid = Grid(ds=ds)

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == ds.lat.name
        assert grid.lon == ds.lon.name
        assert grid.type == "regular_lat_lon"
        assert grid.extent == "global"
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        assert grid.lat_bnds == ds.lat_bnds.name
        assert grid.lon_bnds == ds.lon_bnds.name
        assert grid.nlat == 80
        assert grid.nlon == 180
        assert grid.ncells == 14400

    # not implemented yet
    # assert self.mask


def test_grid_init_ds_simass_degenerated_cells(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP6_SIMASS_DEGEN"], use_cftime=True) as ds:
        grid = Grid(ds=ds)

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "latitude"
        assert grid.lon == "longitude"
        assert grid.type == "curvilinear"
        assert grid.extent == "global"
        assert grid.contains_collapsed_cells
        assert grid.contains_smashed_cells
        assert grid.contains_duplicated_cells is False
        assert grid.lat_bnds == "vertices_latitude"
        assert grid.lon_bnds == "vertices_longitude"
        assert grid.nlat == 384
        assert grid.nlon == 360
        assert grid.ncells == 138240

        # Degenerated cells / Collapsing cells
        # We expect the mask to be the inverted coll_mask
        assert np.array_equal(grid.ds["mask"].data, grid.degen_mask.data)


def test_grid_init_ds_tos_degenerated_cells(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP6_TOS_LR_DEGEN"], use_cftime=True) as ds:
        grid = Grid(ds=ds)

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "latitude"
        assert grid.lon == "longitude"
        assert grid.type == "curvilinear"
        assert grid.extent == "global"
        assert grid.contains_collapsed_cells
        assert grid.contains_smashed_cells is False
        assert grid.contains_duplicated_cells
        assert grid.lat_bnds == "vertices_latitude"
        assert grid.lon_bnds == "vertices_longitude"
        assert grid.nlat == 220
        assert grid.nlon == 256
        assert grid.ncells == 56320

        # Degenerated cells / Collapsing cells
        assert np.array_equal(grid.ds["mask"].data, grid.degen_mask.data)
        assert np.array_equal(grid.ds["mask"].data, grid.coll_mask.data)

        # And to be equal
        assert list(np.where(grid.ds.mask.data.ravel() == 0)[0]) == [
            255,
            511,
            1279,
            2047,
            2303,
            3071,
            3327,
            3583,
            3839,
            4095,
            4863,
            5119,
            7679,
            9471,
            9727,
            9983,
            10751,
            12031,
            12287,
            37631,
            49407,
            53247,
            55039,
            55295,
            56063,
        ]


def test_grid_init_da_tas_regular(mini_esgf_data):
    with xr.open_dataset(
        mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True
    ) as ds:
        da = ds.tas
        grid = Grid(ds=da)

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == da.lat.name
        assert grid.lon == da.lon.name
        assert grid.type == "regular_lat_lon"
        assert grid.extent == "global"
        assert grid.contains_collapsed_cells is None
        assert grid.contains_duplicated_cells is False
        assert grid.lat_bnds is None
        assert grid.lon_bnds is None
        assert grid.nlat == 80
        assert grid.nlon == 180
        assert grid.ncells == 14400


def test_grid_init_ds_tos_curvilinear(mini_esgf_data):
    with xr.open_dataset(
        mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"], use_cftime=True
    ) as ds:
        grid = Grid(ds=ds)

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == ds.latitude.name
        assert grid.lon == ds.longitude.name
        assert grid.type == "curvilinear"
        assert grid.extent == "global"
        assert grid.lat_bnds == "vertices_latitude"
        assert grid.lon_bnds == "vertices_longitude"
        assert grid.contains_collapsed_cells
        assert grid.contains_duplicated_cells
        assert grid.nlat == 404  # 402 w/o halo  # this is number of 'j's
        assert grid.nlon == 802  # 800 w/o halo  # this is the number of 'i's
        assert grid.ncells == 324008  # 321600 w/o halo

        # not implemented yet
        # assert self.mask


def test_grid_init_ds_tas_cordex(mini_esgf_data):
    with xr.open_dataset(
        mini_esgf_data["CORDEX_TAS_ONE_TIMESTEP"], use_cftime=True
    ) as ds:
        grid = Grid(ds=ds)

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "lat"
        assert grid.lon == "lon"
        assert grid.type == "curvilinear"
        assert grid.extent == "regional"
        assert grid.lat_bnds == "lat_vertices"
        assert grid.lon_bnds == "lon_vertices"
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        assert grid.nlat == 201
        assert grid.nlon == 225
        assert grid.ncells == 45225

        ds = ds.drop_vars(["lat", "lon", "lat_vertices", "lon_vertices"])
        with pytest.raises(
            Exception,
            match="The grid format is not supported.",
        ):
            Grid(ds=ds)


def test_grid_init_ds_cordex_erroneous_bounds(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CORDEX_ERRONEOUS_BOUNDS"]) as ds:
        # Warnings will be raised due to erroneous bounds
        with pytest.warns(UserWarning) as issued_warnings:
            grid = Grid(ds=ds)
            if not issued_warnings:
                raise RuntimeError("Expected warnings.")

        issued_warnings = [
            w
            for w in issued_warnings
            if "Latitude and longitude bounds definition is invalid. The bounds will be dropped"
            in str(w.message)
        ]
        assert len(issued_warnings) == 1

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "lat"
        assert grid.lon == "lon"
        assert grid.type == "curvilinear"
        assert grid.extent == "global"  # extent in lon only!
        assert grid.lat_bnds is None
        assert grid.lon_bnds is None
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        assert grid.nlat == 133
        assert grid.nlon == 116
        assert grid.ncells == 15428


def test_grid_init_ds_tas_cordex_ant(mini_esgf_data):
    with xr.open_dataset(
        mini_esgf_data["CORDEX_TAS_ONE_TIMESTEP_ANT"], use_cftime=True
    ) as ds:

        # assert shifted lon frame
        assert np.isclose(ds["lon"].min(), -165.7, atol=0.5)
        assert np.isclose(ds["lon"].max(), 193.1, atol=0.5)

        # Create Grid object
        grid = Grid(ds=ds)

        # assert fixed shifted lon frame and further attributes
        assert np.isclose(grid.ds["lon"].min(), -180, atol=0.5)
        assert np.isclose(grid.ds["lon"].max(), 180, atol=0.5)
        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "lat"
        assert grid.lon == "lon"
        assert grid.type == "curvilinear"
        assert grid.extent == "global"  # only extent in longitude is checked
        assert grid.lat_bnds is None
        assert grid.lon_bnds is None
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        assert grid.nlat == 97
        assert grid.nlon == 125
        assert grid.ncells == 12125


@pytest.mark.slow
def test_grid_init_shifted_lon_frame_GFDL(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP6_GFDL_EXTENT"], use_cftime=True) as ds:

        # confirm shifted lon frame
        assert np.isclose(ds["lon"].min(), -300.0, atol=0.5)
        assert np.isclose(ds["lon"].max(), 60.0, atol=0.5)

        # Create Grid object
        grid = Grid(ds=ds)

        # assert fix of shifted lon frame
        assert np.isclose(grid.ds["lon"].min(), -180.0, atol=0.5)
        assert np.isclose(grid.ds["lon"].max(), 180.0, atol=0.5)
        assert np.isclose(grid.ds["lon_bnds"].min(), -180.0, atol=0.5)
        assert np.isclose(grid.ds["lon_bnds"].max(), 180.0, atol=0.5)

        assert grid.type == "curvilinear"
        assert grid.extent == "global"


def test_grid_init_shifted_lon_frame_IITM(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP6_IITM_EXTENT"], use_cftime=True) as ds:

        # confirm shifted lon frame
        assert np.isclose(ds["longitude"].min(), -280.0, atol=1.0)
        assert np.isclose(ds["longitude"].max(), 80.0, atol=1.0)

        # Create Grid object
        grid = Grid(ds=ds)

        # assert fix of shifted lon frame
        assert np.isclose(grid.ds["longitude"].min(), -180.0, atol=1.0)
        assert np.isclose(grid.ds["longitude"].max(), 180.0, atol=1.0)

        assert grid.type == "curvilinear"
        assert grid.extent == "global"


@pytest.mark.slow
def test_grid_init_unmasked_missing_lon_lat(mini_esgf_data):
    """
    Test that the grid is initialized correctly even if the dataset has unmasked missing lat/lon coordinates.
    """
    # Test case 1
    with xr.open_dataset(mini_esgf_data["CMIP6_EXTENT_UNMASKED"]) as ds:
        g = Grid(ds=ds)
        x0, x1, y0, y1 = clidu.determine_lon_lat_range(
            g.ds, "lon", "lat", "lon_bnds", "lat_bnds", apply_fix=False
        )
        for i, j in [(x0, -180), (x1, 180), (y0, -77.9), (y1, 89.8)]:
            assert np.isclose(i, j, atol=0.1)

        # Test case 2
        ds = xr.open_dataset(mini_esgf_data["CMIP6_UNTAGGED_MISSVALS"])
        g = Grid(ds=ds)
        x0, x1, y0, y1 = clidu.determine_lon_lat_range(
            g.ds, "lon", "lat", "lon_bnds", "lat_bnds", apply_fix=False
        )
        for i, j in [(x0, 0), (x1, 360), (y0, -79.2), (y1, 89.8)]:
            assert np.isclose(i, j, atol=0.1)


def test_grid_init_ds_tas_unstructured(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP6_UNSTR_ICON_A"], use_cftime=True) as ds:
        grid = Grid(ds=ds)

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == ds.latitude.name
        assert grid.lon == ds.longitude.name
        assert grid.type == "unstructured"
        assert grid.extent == "global"
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        assert grid.lat_bnds == "latitude_bnds"
        assert grid.lon_bnds == "longitude_bnds"
        assert grid.ncells == 20480
        assert grid.mask is False


def test_grid_init_ds_zonmean(mini_esgf_data):
    with (
        xr.open_dataset(mini_esgf_data["CMIP6_ZONMEAN_A"], use_cftime=True) as dsA,
        xr.open_dataset(mini_esgf_data["CMIP6_ZONMEAN_B"], use_cftime=True) as dsB,
        xr.open_dataset(
            mini_esgf_data["CMIP6_ATM_VERT_ONE_TIMESTEP_ZONMEAN"], use_cftime=True
        ) as dsC,
    ):

        # Zonal mean dataset without "lon" dimension
        with pytest.raises(
            Exception,
            match="The grid format is not supported.",
        ):
            Grid(ds=dsA)
        with pytest.raises(
            Exception,
            match="The grid format is not supported.",
        ):
            Grid(ds=dsB)
        # Zonal mean dataset with "lon" singleton dimension
        with pytest.raises(
            Exception,
            match="Remapping zonal mean datasets or generally datasets without meridional extent is not supported.",
        ):
            Grid(ds=dsC)


def test_grid_init_ds_erroneous_cf_units_cmip5(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP5_WRONG_CF_UNITS"], use_cftime=True) as ds:

        # Warnings will be raised due to erroneous CF units
        with pytest.warns(UserWarning) as issuedWarnings:
            grid = Grid(ds=ds)
            if not issuedWarnings:
                raise RuntimeError("Expected warnings.")

        issuedWarnings = [
            w
            for w in issuedWarnings
            if "Removing attribute" in str(w.message)
            or "Selecting the best fit." in str(w.message)
        ]
        assert len(issuedWarnings) == 8

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "lat"
        assert grid.lon == "lon"
        assert grid.type == "curvilinear"
        assert grid.extent == "global"
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        # Only the bounds for grid_latitude/longitude are specified
        assert grid.lat_bnds is None
        assert grid.lon_bnds is None
        assert grid.ncells == 83520
        assert grid.mask is False


def test_grid_init_ds_erroneous_cf_units_cmip6(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP6_WRONG_CF_UNITS"], use_cftime=True) as ds:
        # Warnings will be raised due to erroneous CF units
        with pytest.warns(UserWarning) as issuedWarnings:
            grid = Grid(ds=ds)
            if not issuedWarnings:
                raise RuntimeError("Expected warnings.")

        issuedWarnings = [
            w
            for w in issuedWarnings
            if "Removing attribute" in str(w.message)
            or "Selecting the best fit." in str(w.message)
        ]
        assert len(issuedWarnings) == 8

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "latitude"
        assert grid.lon == "longitude"
        assert grid.type == "curvilinear"
        assert grid.extent == "global"
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        # Only the bounds for grid_latitude/longitude are specified
        assert grid.lat_bnds is None
        assert grid.lon_bnds is None
        assert grid.ncells == 83520
        assert grid.mask is False

        # Test bounds creation
        grid = Grid(ds=ds.isel(lat=slice(0, 20), lon=slice(0, 20)), compute_bounds=True)
        assert grid.lat_bnds == "vertices_latitude"
        assert grid.lon_bnds == "vertices_longitude"
        assert grid.ncells == 400


def test_grid_init_ds_erroneous_cf_attrs_cmip6(mini_esgf_data):
    with xr.open_dataset(mini_esgf_data["CMIP6_WRONG_CF_ATTRS"], use_cftime=True) as ds:
        # Warnings will be raised due to erroneous CF units
        with pytest.warns(UserWarning) as issuedWarnings:
            grid = Grid(ds=ds)
            if not issuedWarnings:
                raise RuntimeError("Expected warnings.")

        issuedWarnings = [
            w
            for w in issuedWarnings
            if "Removing attribute" in str(w.message)
            or "Selecting the best fit." in str(w.message)
        ]
        assert len(issuedWarnings) == 8

        assert grid.format == "CF"
        assert grid.source == "Dataset"
        assert grid.lat == "latitude"
        assert grid.lon == "longitude"
        assert grid.type == "curvilinear"
        assert grid.extent == "global"
        assert not grid.contains_collapsed_cells
        assert not grid.contains_duplicated_cells
        # Only the bounds for grid_latitude/longitude are specified
        assert grid.lat_bnds is None
        assert grid.lon_bnds is None
        assert grid.ncells == 990720
        assert grid.mask is False


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_grid_instructor_global():
    grid_instructor = (1.5, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"
    assert not grid.contains_collapsed_cells
    assert not grid.contains_duplicated_cells

    # check that grid_from_instructor sets the format to xESMF
    grid._grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 120
    assert grid.nlon == 240
    assert grid.ncells == 28800
    assert grid.mask is False


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_grid_instructor_2d_regional_change_lon():
    grid_instructor = (50, 240, 1.5, -90, 90, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "regional"
    assert not grid.contains_collapsed_cells
    assert not grid.contains_duplicated_cells

    # check that grid_from_instructor sets the format to xESMF
    grid._grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 120
    assert grid.nlon == 127
    assert grid.ncells == 15240
    assert grid.mask is False


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_grid_instructor_2d_regional_change_lat():
    grid_instructor = (0, 360, 1.5, -60, 50, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"

    # Extent in y-direction ignored, as not of importance
    #  for xesmf.Regridder. Extent in x-direction should be
    #  detected as "global"
    assert grid.extent == "global"

    assert not grid.contains_collapsed_cells
    assert not grid.contains_duplicated_cells

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 73
    assert grid.nlon == 240
    assert grid.ncells == 17520
    assert grid.mask is False


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_grid_instructor_2d_regional_change_lon_and_lat():
    grid_instructor = (50, 240, 1.5, -60, 50, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "regional"
    assert not grid.contains_collapsed_cells
    assert not grid.contains_duplicated_cells

    # check that grid_from_instructor sets the format to xESMF
    grid._grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 73
    assert grid.nlon == 127
    assert grid.ncells == 9271
    assert grid.mask is False


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_grid_instructor_2d_global():
    grid_instructor = (0, 360, 1.5, -90, 90, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"
    assert not grid.contains_collapsed_cells
    assert not grid.contains_duplicated_cells

    # check that grid_from_instructor sets the format to xESMF
    grid._grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 120
    assert grid.nlon == 240
    assert grid.ncells == 28800
    assert grid.mask is False


def test_from_grid_id():
    """Test to create grid from grid_id"""
    grid = Grid(grid_id="ERA-40")

    assert grid.format == "CF"
    assert grid.source == "Predefined_ERA-40"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"
    assert not grid.contains_collapsed_cells
    assert not grid.contains_duplicated_cells
    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 145
    assert grid.nlon == 288
    assert grid.ncells == 41760
    assert grid.mask is False


@pytest.mark.parametrize(
    "grid_id",
    [
        "0pt5deg_lsm",
        "0pt25deg_era5_lsm",
        "1deg_lsm",
        "2deg_lsm",
        "0pt25deg_era5_lsm_binary",
        "1deg_lsm_binary",
        "2deg_lsm_binary",
        "T63_lsm_binary",
        "T127_lsm_binary",
    ],
)
def test_from_grid_id_mask(grid_id):
    """Test to create grid from grid_id"""

    grid = Grid(grid_id=grid_id, mask="ocean")
    assert grid.format == "CF"
    assert grid.mask is True
    np.all(grid.lsm.data == grid.ds["mask"].data)
    osum = grid.lsm.sum().item()
    # Make sure it is a binary mask
    mask = grid.ds["mask"].copy(deep=True)
    np.all(grid.lsm.astype("bool").astype(np.int32).data == mask.data)

    grid = Grid(grid_id=grid_id, mask="land")
    assert grid.format == "CF"
    assert grid.mask is True
    np.all(grid.lsm.data == grid.ds["mask"].data)
    lsum = grid.lsm.sum().item()
    # Make sure it is a binary mask
    mask = grid.ds["mask"].copy(deep=True).data
    np.all(grid.lsm.astype("bool").astype(np.int32).data == mask.data)

    # Make sure "land" and "ocean" are interpreted properly
    assert lsum > osum


@pytest.mark.slow
@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
class TestGridFromDS:

    def test_grid_from_ds_adaptive_extent(self, mini_esgf_data):
        """Test that the extent is evaluated as global for original and derived adaptive grid."""
        with (
            xr.open_dataset(
                mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"], use_cftime=True
            ) as dsA,
            xr.open_dataset(
                mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True
            ) as dsB,
            xr.open_dataset(
                mini_esgf_data["CMIP6_UNSTR_ICON_A"], use_cftime=True
            ) as dsC,
        ):

            gA = Grid(ds=dsA)
            gB = Grid(ds=dsB)
            gC = Grid(ds=dsC)
            gAa = Grid(ds=dsA, grid_id="adaptive")
            gBa = Grid(ds=dsB, grid_id="adaptive")
            gCa = Grid(ds=dsC, grid_id="auto")

            assert gA.extent == "global"
            assert gB.extent == "global"
            assert gC.extent == "global"
            assert gAa.extent == "global"
            assert gBa.extent == "global"
            assert gCa.extent == "global"

    def test_grid_from_ds_adaptive_reproducibility(self):
        """Test that the extent is evaluated as global for original and derived adaptive grid."""
        fpathA = get_grid_file("0pt25deg")
        dsA = xr.open_dataset(fpathA, use_cftime=True)
        fpathB = get_grid_file("1deg")
        dsB = xr.open_dataset(fpathB, use_cftime=True)

        gAa = Grid(ds=dsA, grid_id="adaptive")
        gA = Grid(grid_id="0pt25deg")
        gBa = Grid(ds=dsB, grid_id="adaptive")
        gB = Grid(grid_id="1deg")

        assert gA.extent == "global"
        assert gA.compare_grid(gAa)
        assert gB.extent == "global"
        assert gB.compare_grid(gBa)


@pytest.mark.slow
def test_compare_grid_same_resolution():
    """Test that two grids of same resolution from different sources evaluate as the same grid"""
    ds025 = xr.open_dataset(get_grid_file("0pt25deg_era5"))
    g025 = Grid(grid_id="0pt25deg_era5", compute_bounds=True)
    g025_lsm = Grid(grid_id="0pt25deg_era5_lsm", compute_bounds=True)

    assert g025.compare_grid(g025_lsm)
    assert g025.compare_grid(ds025)
    assert g025_lsm.compare_grid(ds025)


def test_compare_grid_diff_in_precision(mini_esgf_data):
    """Test that the same grid stored with different precision is evaluated as the same grid"""
    dsA = xr.open_dataset(mini_esgf_data["CMIP6_TAS_PRECISION_A"], use_cftime=True)
    dsB = xr.open_dataset(mini_esgf_data["CMIP6_TAS_PRECISION_B"], use_cftime=True)

    gA = Grid(ds=dsA)
    gB = Grid(ds=dsB)

    assert gA.compare_grid(gB)


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_compare_grid_hash_dict_and_verbose(capfd):
    """Test Grid.hash_dict keys and Grid.compare_grid verbose option"""
    gA = Grid(grid_instructor=(1.0, 0.5))
    gB = Grid(grid_instructor=(1.0,))
    is_equal = gA.compare_grid(gB, verbose=True)
    stdout, stderr = capfd.readouterr()

    assert stderr == ""
    assert stdout == "The two grids differ in their respective lat, lat_bnds.\n"
    assert not is_equal
    assert len(gA.hash_dict) == 5
    assert list(gA.hash_dict.keys()) == ["lat", "lon", "lat_bnds", "lon_bnds", "mask"]


def test_to_netcdf(tmp_path, mini_esgf_data):
    """Test if grid file is properly written to disk using to_netcdf method."""
    # Create Grid object
    dsA = xr.open_dataset(mini_esgf_data["CMIP6_TAS_PRECISION_A"])
    gA = Grid(ds=dsA)

    # Save to disk
    outdir = Path(tmp_path, "grids")
    outfile = "grid_test.nc"
    gA.to_netcdf(folder=outdir, filename=outfile)

    # Read from disk - ensure outfile has been created and lockfile deleted
    assert os.path.isfile(Path(outdir, outfile))
    assert len([os.path.basename(f) for f in glob(f"{outdir}/*")]) == 1
    dsB = xr.open_dataset(Path(outdir, outfile))
    gB = Grid(ds=dsB)

    # Ensure Grid attributes and ds attrs are the same
    assert gA.compare_grid(gB)
    assert gA.format == gB.format
    assert gA.type == gB.type
    assert gA.extent == gB.extent
    assert gA.source == gB.source
    assert gA.contains_collapsed_cells == gB.contains_collapsed_cells
    assert sorted(list(gA.ds.attrs.keys()) + ["clisops"]) == sorted(
        list(gB.ds.attrs.keys())
    )

    # Ensure all variables have been deleted from the dataset
    assert not list(gB.ds.data_vars)
    assert sorted(list(gB.ds.coords)) == [gA.lat, gA.lat_bnds, gA.lon, gA.lon_bnds]

    # Ensure the non-CF-compliant attributes xarray commonly defines are not present:
    assert "_FillValue" not in dsB[gB.lat_bnds].attrs.keys()
    assert "_FillValue" not in dsB[gB.lon_bnds].attrs.keys()
    assert "coordinates" not in dsB.attrs.keys()


@pytest.mark.slow
class TestDetect:

    def test_detect_extent_shifted_lon_frame(self, mini_esgf_data):
        """Test whether the extent can be correctly inferred for a dataset with shifted longitude frame."""
        # Load dataset with longitude ranging from (-300, 60)
        ds = xr.open_dataset(mini_esgf_data["CMIP6_GFDL_EXTENT"], use_cftime=True)

        # Convert the longitude frame to 0,360 (shall happen implicitly in the future)
        ds, ll, lu = clidu.cf_convert_between_lon_frames(ds, (0, 360))
        assert (ll, lu) == (0, 360)

        # Create Grid object and assert zonal extent
        g = Grid(ds=ds)
        assert g.extent == "global"

    def test_detect_collapsed_cells(self, mini_esgf_data, load_test_data):
        """Test that collapsed cells are properly identified."""

        dsA = xr.open_dataset(mini_esgf_data["CMIP6_OCE_HALO_CNRM"], use_cftime=True)
        dsB = xr.open_dataset(
            mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"], use_cftime=True
        )
        dsC = xr.open_dataset(
            mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True
        )

        gA = Grid(ds=dsA)
        gB = Grid(ds=dsB)
        gC = Grid(ds=dsC)

        assert gA.contains_collapsed_cells
        assert gB.contains_collapsed_cells
        assert not gC.contains_collapsed_cells

    def test_detect_duplicated_cells(self, mini_esgf_data):
        """Test that collapsed cells are properly identified"""
        dsA = xr.open_dataset(mini_esgf_data["CMIP6_OCE_HALO_CNRM"], use_cftime=True)
        dsB = xr.open_dataset(
            mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"], use_cftime=True
        )
        dsC = xr.open_dataset(
            mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True
        )

        gA = Grid(ds=dsA)
        gB = Grid(ds=dsB)
        gC = Grid(ds=dsC)

        assert gA.contains_duplicated_cells
        assert gB.contains_duplicated_cells
        assert not gC.contains_duplicated_cells


def test_subsetted_grid(mini_esgf_data):
    ds = xr.open_dataset(
        mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True
    ).load()

    area = (0.0, 10.0, 175.0, 90.0)

    ds = subset(
        ds=ds,
        area=area,
        output_type="xarray",
    )[0]

    grid = Grid(ds=ds)

    assert grid.format == "CF"
    assert grid.source == "Dataset"
    assert grid.lat == ds.lat.name
    assert grid.lon == ds.lon.name
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "regional"
    assert not grid.contains_collapsed_cells
    assert not grid.contains_duplicated_cells

    assert grid.lat_bnds == ds.lat_bnds.name
    assert grid.lon_bnds == ds.lon_bnds.name
    assert grid.nlat == 35
    assert grid.nlon == 88
    assert grid.ncells == 3080

    # not implemented yet
    # assert self.mask


@pytest.mark.slow
def test_drop_vars_transfer_coords(mini_esgf_data):
    """Test for Grid methods drop_vars and transfer_coords"""
    ds = xr.open_dataset(mini_esgf_data["CMIP6_ATM_VERT_ONE_TIMESTEP"])
    g = Grid(ds=ds)
    gt = Grid(grid_id="0pt25deg_era5_lsm", compute_bounds=True)
    assert sorted(list(g.ds.data_vars.keys())) == ["o3", "ps"]
    assert list(gt.ds.data_vars.keys()) != []

    gt._drop_vars()
    assert gt.ds.attrs == {}
    assert sorted(list(gt.ds.coords.keys())) == [
        "lat_bnds",
        "latitude",
        "lon_bnds",
        "longitude",
    ]

    gt._transfer_coords(g)
    assert gt.ds.attrs["institution"] == "Max Planck Institute for Meteorology"
    assert gt.ds.attrs["activity_id"] == "CMIP"
    assert sorted(list(gt.ds.coords.keys())) == [
        "ap",
        "ap_bnds",
        "b",
        "b_bnds",
        "lat_bnds",
        "latitude",
        "lev",
        "lev_bnds",
        "lon_bnds",
        "longitude",
        "time",
        "time_bnds",
    ]
    assert list(gt.ds.data_vars.keys()) == []


def test_calculate_bounds_curvilinear(mini_esgf_data):
    """Test for bounds calculation for curvilinear grid"""
    ds = xr.open_dataset(mini_esgf_data["CORDEX_TAS_NO_BOUNDS"]).isel(
        {"rlat": range(10), "rlon": range(10)}
    )
    g = Grid(ds=ds, compute_bounds=True)
    assert g.lat_bnds is not None
    assert g.lon_bnds is not None


def test_calculate_bounds_duplicated_cells(mini_esgf_data):
    """Test for bounds calculation for curvilinear grid"""
    ds = xr.open_dataset(mini_esgf_data["CORDEX_TAS_NO_BOUNDS"]).isel(
        {"rlat": range(10), "rlon": range(10)}
    )

    # create duplicated cells
    ds["lat"][:, 0] = ds["lat"][:, 1]
    ds["lon"][:, 0] = ds["lon"][:, 1]

    # assert warning
    with pytest.warns(
        UserWarning,
        match="This grid contains duplicated cell centers, which may affect the quality of the calculated bounds.",
    ):
        Grid(ds=ds, compute_bounds=True)


def test_centers_within_bounds_curvilinear(mini_esgf_data):
    """Test for bounds calculation for curvilinear grid"""
    ds = xr.open_dataset(mini_esgf_data["CORDEX_TAS_NO_BOUNDS"]).isel(
        {"rlat": range(10), "rlon": range(10)}
    )
    g = Grid(ds=ds, compute_bounds=True)
    assert g.lat_bnds is not None
    assert g.lon_bnds is not None
    assert g.contains_collapsed_cells is False

    # Check that there are bounds values smaller and greater than the cell center values
    ones = np.ones((g.nlat, g.nlon), dtype=int)
    assert np.all(
        ones
        == xr.where(
            np.sum(xr.where(g.ds[g.lat] >= g.ds[g.lat_bnds], 1, 0), -1) > 0, 1, 0
        )
    )
    assert np.all(
        ones
        == xr.where(
            np.sum(xr.where(g.ds[g.lat] <= g.ds[g.lat_bnds], 1, 0), -1) > 0, 1, 0
        )
    )
    assert np.all(
        ones
        == xr.where(
            np.sum(xr.where(g.ds[g.lon] >= g.ds[g.lon_bnds], 1, 0), -1) > 0, 1, 0
        )
    )
    assert np.all(
        ones
        == xr.where(
            np.sum(xr.where(g.ds[g.lon] <= g.ds[g.lon_bnds], 1, 0), -1) > 0, 1, 0
        )
    )


@pytest.mark.slow
def test_centers_within_bounds_regular_lat_lon():
    """Test for bounds calculation of regular lat lon grid"""
    g = Grid(grid_id="0pt25deg_era5_lsm", compute_bounds=True)
    assert g.lat_bnds is not None
    assert g.lon_bnds is not None
    assert bool(g.contains_collapsed_cells) is False

    # Check that there are bounds values smaller and greater than the cell center values
    ones_lat = np.ones((g.nlat,), dtype=int)
    ones_lon = np.ones((g.nlon,), dtype=int)
    assert np.all(
        ones_lat
        == xr.where(
            np.sum(xr.where(g.ds[g.lat] >= g.ds[g.lat_bnds], 1, 0), -1) > 0, 1, 0
        )
    )
    assert np.all(
        ones_lat
        == xr.where(
            np.sum(xr.where(g.ds[g.lat] <= g.ds[g.lat_bnds], 1, 0), -1) > 0, 1, 0
        )
    )
    assert np.all(
        ones_lon
        == xr.where(
            np.sum(xr.where(g.ds[g.lon] >= g.ds[g.lon_bnds], 1, 0), -1) > 0, 1, 0
        )
    )
    assert np.all(
        ones_lon
        == xr.where(
            np.sum(xr.where(g.ds[g.lon] <= g.ds[g.lon_bnds], 1, 0), -1) > 0, 1, 0
        )
    )


def test_data_vars_coords_reset_and_cfxr(mini_esgf_data):
    ds_a = xr.open_dataset(mini_esgf_data["CMIP6_ATM_VERT_ONE_TIMESTEP"])

    # generate dummy areacella
    areacella = xr.DataArray(
        {
            "dims": ("lat", "lon"),
            "attrs": {
                "standard_name": "cell_area",
                "cell_methods": "area: sum",
            },
            "data": np.ones(18432, dtype=np.float32).reshape((96, 192)),
        }
    )
    ds_a.update({"areacella": areacella})
    ds_b = xr.decode_cf(ds_a, decode_coords="all")

    # Grid._set_data_vars_and_coords should (re)set coords appropriately
    gA = Grid(ds=ds_a)
    gB = Grid(ds=ds_b)

    # cf_xarray should be able to identify important attributes and present both datasets equally
    assert gA.compare_grid(gB)
    assert gA.ds.cf.cell_measures == gB.ds.cf.cell_measures
    assert gA.ds.o3.cf.cell_measures == gB.ds.o3.cf.cell_measures
    assert gA.ds.cf.formula_terms == gB.ds.cf.formula_terms
    assert gA.ds.o3.cf.formula_terms == gB.ds.o3.cf.formula_terms
    assert gA.ds.cf.bounds == gB.ds.cf.bounds
    assert str(gA.ds.cf) == str(gB.ds.cf)


# test all methods
@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
class TestWeights:

    def test_grids_in_and_out_bilinear(self, tmp_path, mini_esgf_data):
        ds = xr.open_dataset(mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True)
        grid_in = Grid(ds=ds)

        assert grid_in.extent == "global"

        grid_instructor_out = (0, 360, 1.5, -90, 90, 1.5)
        grid_out = Grid(grid_instructor=grid_instructor_out)

        assert grid_out.extent == "global"

        weights_cache_init(Path(tmp_path, "weights"))
        w = Weights(grid_in=grid_in, grid_out=grid_out, method="bilinear")

        assert w.method == "bilinear"
        assert (
            w.id
            == "8edb4ee828dbebc2dc8e193281114093_bf73249f1725126ad3577727f3652019_peri_skip-degen_bilinear"
        )
        assert w.periodic
        assert w.id in w.filename
        assert "xESMF_v" in w.tool
        assert w.format == "xESMF"

        # default file_name = method_inputgrid_outputgrid_periodic"
        assert w.regridder.filename == "bilinear_80x180_120x240_peri.nc"

    def test_grids_in_and_out_conservative(self, tmp_path, mini_esgf_data):
        ds = xr.open_dataset(mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True)
        grid_in = Grid(ds=ds)

        assert grid_in.extent == "global"

        grid_instructor_out = (0, 360, 1.5, -90, 90, 1.5)
        grid_out = Grid(grid_instructor=grid_instructor_out)

        assert grid_out.extent == "global"

        weights_cache_init(Path(tmp_path, "weights"))
        w = Weights(grid_in=grid_in, grid_out=grid_out, method="conservative")

        assert w.method == "conservative"
        assert (
            w.id
            == "8edb4ee828dbebc2dc8e193281114093_bf73249f1725126ad3577727f3652019_peri_skip-degen_conservative"
        )
        assert (
            w.periodic != w.regridder.periodic
        )  # xESMF resets periodic to False for conservative weights
        assert w.id in w.filename
        assert "xESMF_v" in w.tool
        assert w.format == "xESMF"

        # default file_name = method_inputgrid_outputgrid_periodic"
        assert w.regridder.filename == "conservative_80x180_120x240.nc"

    # def test_from_id(self):
    #     """Test creating a Weights object by reading weights from disk, identified by the id."""
    #     pass
    #
    # def test_from_disk(self):
    #     """Test creating a Weights object by reading an xESMF or other weights file from disk."""
    #     pass

    def test_conservative_no_bnds(self, tmp_path, mini_esgf_data):
        """Test whether exception is raised when no bounds present for conservative remapping."""
        ds = xr.open_dataset(mini_esgf_data["CORDEX_TAS_NO_BOUNDS"])
        gi = Grid(ds=ds)
        go = Grid(grid_id="1deg", compute_bounds=True)

        assert gi.lat_bnds is None
        assert gi.lon_bnds is None
        assert go.lat_bnds is not None
        assert go.lon_bnds is not None

        with pytest.raises(
            Exception,
            match="For conservative remapping, horizontal grid bounds have to be defined for the source and target grids.",
        ):
            weights_cache_init(Path(tmp_path, "weights"))
            Weights(grid_in=gi, grid_out=go, method="conservative")


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_Weights_compute(tmp_path):
    """Test the generation of Weights with the _compute method."""
    g = Grid(grid_id="1deg")
    g_out = Grid(grid_id="2deg_lsm")

    weights_cache_init(Path(tmp_path, "weights"))

    # Exception should be raised if input and output grid are evaluated as equal
    with pytest.raises(
        Exception,
        match="The selected source and target grids are the same. No regridding operation required.",
    ):
        Weights(
            g,
            Grid(grid_instructor=(0.0, 360.0, 1.0, -90.0, 90.0, 1.0)),
            method="bilinear",
        )

    # Exception should be raised for conservative method if input or output grid do not contain bounds
    with pytest.raises(
        Exception,
        match="For conservative remapping, horizontal grid bounds have to be defined for the source and target grids.",
    ):
        Weights(g, g_out, method="conservative")

    # Test computation and cache storage
    g_out = Grid(grid_id="2deg_lsm", compute_bounds=True)
    w = Weights(g, g_out, method="nearest_s2d")
    assert w.id == w._read_info_from_cache("uid")
    assert w.tool == w._read_info_from_cache("tool")
    assert w.regridder.periodic == w.periodic
    assert w._read_info_from_cache("method") == "nearest_s2d"
    assert w.regridder.method == w.method
    assert w.format == "xESMF"
    assert w.regridder.filename == "nearest_s2d_180x360_90x180_peri.nc"
    assert not w.regridder.reuse_weights
    assert w.regridder.ignore_degenerate is True
    assert w.regridder.n_in == 180 * 360
    assert w.regridder.n_out == 90 * 180
    assert w.ignore_degenerate is True
    assert w.filename == "weights_" + w.id + ".nc"

    # Test weight reusage
    z = Weights(g, g_out, method="nearest_s2d")
    assert z.regridder.reuse_weights


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_Weights_compute_unstructured(tmp_path, mini_esgf_data):
    """Test the generation of Weights for unstructured grids with the _compute method."""
    ds = xr.open_dataset(mini_esgf_data["CMIP6_UNSTR_ICON_A"], use_cftime=True)
    g = Grid(ds=ds)
    g_out = Grid(grid_id="2deg_lsm", compute_bounds=True)

    weights_cache_init(Path(tmp_path, "weights"))

    # Exception should be raised for other than nearest_s2d remapping method
    with pytest.raises(
        Exception,
        match="For unstructured grids, the only supported remapping method that is currently supported is nearest neighbour.",
    ):
        Weights(g, g_out, method="conservative")

    # Check translated xesmf settings
    w = Weights(g, g_out, method="nearest_s2d")
    assert w.regridder.sequence_in
    assert not w.regridder.sequence_out
    assert w.regridder.ignore_degenerate is True
    assert w.regridder.n_in == g.ncells
    assert w.regridder.n_out == 90 * 180


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_Weights_generate_id(tmp_path):
    "Test the generation of Weight ids."
    g = Grid(grid_id="1deg")
    g_out = Grid(grid_id="2pt5deg")

    weights_cache_init(Path(tmp_path, "weights"))
    w = Weights(g, g_out, method="bilinear")

    assert w.id == w._generate_id()
    assert w.id == "_".join([g.hash, g_out.hash, "peri", "skip-degen", "bilinear"])


@pytest.mark.slow
@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_Weights_init_with_collapsed_cells(tmp_path, mini_esgf_data):
    "Test the creation of remapping weights for a grid containing collapsed cells"
    # ValueError: ESMC_FieldRegridStore failed with rc = 506. Please check the log files (named "*ESMF_LogFile").
    ds = xr.open_dataset(mini_esgf_data["CMIP6_OCE_HALO_CNRM"], use_cftime=True)

    g = Grid(ds=ds)
    g_out = Grid(grid_instructor=(10.0,))

    assert g.contains_collapsed_cells

    weights_cache_init(Path(tmp_path, "weights"))
    Weights(g, g_out, method="bilinear")


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_Regridder_filename(tmp_path):
    """Test that Regridder filename is reset properly."""
    g1 = Grid(grid_id="2pt5deg")
    g2 = Grid(grid_id="2deg_lsm")

    weights_cache_init(Path(tmp_path, "weights"))

    w = Weights(g1, g2, method="nearest_s2d")

    assert w.regridder.filename == w.regridder._get_default_filename()
    assert os.path.isfile(Path(tmp_path, "weights", w.filename))
    assert w.filename != w.regridder.filename


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_cache_init_and_flush(tmp_path):
    """Test of the cache init and flush functionalities"""

    weights_dir = Path(tmp_path, "clisops_weights")
    weights_cache_init(weights_dir)

    gi = Grid(grid_instructor=20)
    go = Grid(grid_instructor=10)
    Weights(grid_in=gi, grid_out=go, method="nearest_s2d")

    flist = sorted(os.path.basename(f) for f in glob(f"{weights_dir}/*"))
    flist_ref = [
        "grid_4d324aecaa8302ab0f2f212e9821b00f.nc",
        "grid_96395935b4e81f2a5b55970bd920d82c.nc",
        "weights_4d324aecaa8302ab0f2f212e9821b00f_96395935b4e81f2a5b55970bd920d82c_peri_skip-degen_nearest_s2d.json",
        "weights_4d324aecaa8302ab0f2f212e9821b00f_96395935b4e81f2a5b55970bd920d82c_peri_skip-degen_nearest_s2d.nc",
    ]
    assert flist == flist_ref

    weights_cache_flush()
    assert glob(f"{weights_dir}/*") == []


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_cache_lock_mechanism(tmp_path, mini_esgf_data):
    """Test lock mechanism of local regrid weights cache."""
    with xr.open_dataset(
        mini_esgf_data["CMIP6_TAS_ONE_TIME_STEP"], use_cftime=True
    ) as ds:

        grid_in = Grid(ds=ds)
        grid_out = Grid(grid_instructor=10)

        # First round - creating the weights should work without problems
        weights_cache_init(Path(tmp_path, "weights"))
        w = Weights(grid_in=grid_in, grid_out=grid_out, method="nearest_s2d")

        # Remove grid files to suppress related warnings of already existing files
        os.remove(Path(tmp_path, "weights", "grid_" + grid_in.hash + ".nc"))
        os.remove(Path(tmp_path, "weights", "grid_" + grid_out.hash + ".nc"))

        # Second round, but manually put lockfile in place
        LOCK_FILE = Path(tmp_path, "weights", w.filename + ".lock")
        lock = FileLock(LOCK_FILE)
        lock.acquire(timeout=10)

        # Fail test if lockfile is not recognized
        with pytest.warns(UserWarning, match=r"lockfile|xarray") as issuedWarnings:
            Weights(grid_in=grid_in, grid_out=grid_out, method="nearest_s2d")
            if not issuedWarnings:
                raise RuntimeError("Lockfile not recognized/ignored.")

        # Filter matches and assert number of warnings
        issuedWarningsLockfile = [
            w for w in issuedWarnings if "lockfile" in str(w.message)
        ]
        assert len(issuedWarningsLockfile) == 1
        lock.release()


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_cache_reinit_and_write_protection(tmp_path):
    """Test that Regridder does not write to cache if lockfile exists."""
    g1 = Grid(grid_id="2pt5deg")
    g2 = Grid(grid_id="2deg_lsm")

    orig_cache_dir = CONFIG["clisops:grid_weights"]["local_weights_dir"]
    weights_cache_init(Path(tmp_path, "weights"))

    # Create weights, get filename and flush cache
    w = Weights(g1, g2, method="nearest_s2d")
    fname = w.filename
    weights_cache_flush()

    # Create lockfile
    LOCK_FILE = Path(tmp_path, "weights", fname + ".lock")
    lock = FileLock(LOCK_FILE)
    lock.acquire(timeout=10)

    # recreate weights
    w = Weights(g1, g2, method="nearest_s2d")

    # ensure that cache does not contain weight file and metadata
    lock.release()
    flist = sorted(os.path.basename(f) for f in glob(f"{Path(tmp_path, 'weights')}/*"))
    assert all([f.startswith("grid_") for f in flist])
    assert len(flist) == 2
    assert Path(CONFIG["clisops:grid_weights"]["local_weights_dir"]) == Path(
        tmp_path, "weights"
    )
    assert CONFIG["clisops:grid_weights"]["local_weights_dir"] != orig_cache_dir


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_read_metadata(tmp_path):
    """Test Weights method _read_info_from_cache."""
    g1 = Grid(grid_instructor=10.0)
    g2 = Grid(grid_instructor=15.0)

    # Create weights and assert attributes written to cache
    weights_cache_init(Path(tmp_path, "weights"))
    w = Weights(g1, g2, method="nearest_s2d")

    assert w._read_info_from_cache("filename") == w.filename
    assert w._read_info_from_cache("method") == "nearest_s2d"
    assert w._read_info_from_cache("source_uid") == g1.hash
    assert w._read_info_from_cache("target_extent") == g2.extent
    assert w._read_info_from_cache("bla") is None


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
class TestRegrid:
    c6tots = "CMIP6_TAS_ONE_TIME_STEP"

    def _setup(self, ds):
        if hasattr(self, "setup_done"):
            return

        self.grid_in = Grid(ds=ds)

        self.grid_instructor_out = (0, 360, 1.5, -90, 90, 1.5)
        self.grid_out = Grid(grid_instructor=self.grid_instructor_out)
        self.setup_done = True

    def test_adaptive_masking(self, tmp_path, mini_esgf_data):
        with xr.open_dataset(mini_esgf_data[self.c6tots], use_cftime=True) as ds:
            self._setup(ds)

            weights_cache_init(Path(tmp_path, "weights"))
            w = Weights(
                grid_in=self.grid_in, grid_out=self.grid_out, method="conservative"
            )
            regrid(self.grid_in, self.grid_out, w, adaptive_masking_threshold=0.7)

    def test_no_adaptive_masking(self, tmp_path, mini_esgf_data):
        with xr.open_dataset(mini_esgf_data[self.c6tots], use_cftime=True) as ds:
            self._setup(ds)

            weights_cache_init(Path(tmp_path, "weights"))
            w = Weights(grid_in=self.grid_in, grid_out=self.grid_out, method="bilinear")
            regrid(self.grid_in, self.grid_out, w, adaptive_masking_threshold=-1.0)

    def test_duplicated_cells_warning_issued(self, tmp_path, mini_esgf_data):
        with xr.open_dataset(mini_esgf_data[self.c6tots], use_cftime=True) as ds:
            self._setup(ds)

            weights_cache_init(Path(tmp_path, "weights"))
            w = Weights(
                grid_in=self.grid_in, grid_out=self.grid_out, method="conservative"
            )

            # Cheat regrid into thinking, grid_in contains duplicated cells
            self.grid_in.contains_duplicated_cells = True

            with pytest.warns(
                UserWarning,
                match="The grid of the selected dataset contains duplicated cells. "
                "For the conservative remapping method, "
                "duplicated grid cells contribute to the resulting value, "
                "which is in most parts counter-acted by the applied re-normalization. "
                "However, please be wary with the results and consider removing / masking "
                "the duplicated cells before remapping.",
            ) as issuedWarnings:
                regrid(self.grid_in, self.grid_out, w, adaptive_masking_threshold=0.0)
                if not issuedWarnings:
                    raise RuntimeError(
                        "No warning issued regarding the duplicated cells in the grid."
                    )
                elif Version(xr.__version__) >= Version("2023.3.0"):
                    # Warning will also be issued for xarray being too new
                    assert len(issuedWarnings) == 2
                else:
                    assert len(issuedWarnings) == 1

    def test_regrid_dataarray(self, tmp_path, mini_esgf_data):
        with xr.open_dataset(mini_esgf_data[self.c6tots], use_cftime=True) as ds:
            self._setup(ds)

            weights_cache_init(Path(tmp_path, "weights"))
            w = Weights(
                grid_in=self.grid_in, grid_out=self.grid_out, method="nearest_s2d"
            )
            grid_da = Grid(self.grid_in.ds.tas)

            vattrs = (
                "regrid_method",
                "standard_name",
                "long_name",
                "comment",
                "units",
                "cell_methods",
                "cell_measures",
                "history",
            )
            gattrs = (
                "grid",
                "grid_label",
                "regrid_operation",
                "regrid_tool",
                "regrid_weights_uid",
            )

            r1 = regrid(grid_da, self.grid_out, w, keep_attrs=True)
            assert vattrs == tuple(r1["tas"].attrs.keys())
            assert gattrs == tuple(r1.attrs.keys())

            r2 = regrid(grid_da, self.grid_out, w, keep_attrs=False)
            assert ("regrid_method",) == tuple(r2["tas"].attrs.keys())
            assert gattrs == tuple(r2.attrs.keys())


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_duplicated_cells_renormalization(tmp_path, mini_esgf_data):
    # todo: Should probably be an xesmf test as well, will do PR there in the future
    with xr.open_dataset(
        mini_esgf_data["CMIP6_STAGGERED_UCOMP"], use_cftime=True
    ) as ds:

        # some internal xesmf code to create array of ones
        missing = np.isnan(ds.tauuo)
        ds["tauuo"] = (~missing).astype("d")

        grid_in = Grid(ds=ds)
        assert grid_in.contains_collapsed_cells is True
        assert grid_in.contains_duplicated_cells is True

        # Make sure all values that are not missing, are equal to one
        assert grid_in.ncells == ds["tauuo"].where(~missing, 1.0).sum()
        # Make sure all values that are missing are equal to 0
        assert 0.0 == ds["tauuo"].where(missing, 0.0).sum()

        grid_out = Grid(grid_instructor=(0, 360, 1.5, -90, 90, 1.5))
        weights_cache_init(Path(tmp_path, "weights"))
        w = Weights(grid_in=grid_in, grid_out=grid_out, method="conservative")

        # Remap using adaptive masking
        r1 = regrid(grid_in, grid_out, w, adaptive_masking_threshold=0.5)

        # Remap using default setting (na_thres = 0.5)
        r2 = regrid(grid_in, grid_out, w)

        # Make sure both options yield equal results
        xr.testing.assert_equal(r1, r2)

        # Remap without using adaptive masking - internally, then adaptive masking is used
        #   with threshold 0., to still re-normalize contributions from duplicated cells
        #   but not from masked cells or out-of-source-domain area
        r3 = regrid(grid_in, grid_out, w, adaptive_masking_threshold=-1.0)

        # Make sure, contributions from duplicated cells (i.e. values > 1) are re-normalized
        assert r2["tauuo"].where(r2["tauuo"] > 1.0, 0.0).sum() == 0.0
        assert r3["tauuo"].where(r2["tauuo"] > 1.0, 0.0).sum() == 0.0

        # Make sure xesmf behaves as expected:
        #   test that deactivated adaptive masking in xesmf will yield results > 1
        #   and else, contributions from duplicated cells will be re-normalized
        r4 = w.regridder(ds["tauuo"], skipna=False)
        r5 = w.regridder(ds["tauuo"], skipna=True, na_thres=0.0)
        assert r4.where(r4 > 1.0, 0.0).sum() > 0.0
        assert r5.where(r5 > 1.0, 0.0).sum() == 0.0
