import numpy as np
import pytest
import xarray as xr

from clisops.core.regrid import Grid
from clisops.ops.subset import subset

from .._common import CMIP6_TAS_ONE_TIME_STEP, CMIP6_TOS_ONE_TIME_STEP

# test from grid_id --predetermined
# test different types of grid e.g. irregular, not supported type
# test for errors e.g.
# no lat/lon in dataset
# more than one latitude/longitude
# grid instructor tuple not correct length


def test_grid_init_ds_tas_regular(load_esgf_test_data):
    ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
    grid = Grid(ds=ds)

    assert grid.format == "CF"
    assert grid.source == "Dataset"
    assert grid.lat == ds.lat.name
    assert grid.lon == ds.lon.name
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"
    assert grid.lat_bnds == ds.lat_bnds.name
    assert grid.lon_bnds == ds.lon_bnds.name
    assert grid.nlat == 80
    assert grid.nlon == 180
    assert grid.ncells == 14400

    # not implemented yet
    # assert self.mask


@pytest.mark.skip("Doesn't have bounds attribute")
def test_grid_init_da_tas_regular(load_esgf_test_data):
    ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
    da = ds.tas
    grid = Grid(ds=da)

    assert grid.format == "CF"
    assert grid.source == "Dataset"
    assert grid.lat == da.lat.name
    assert grid.lon == da.lon.name
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"
    assert grid.lat_bnds == ""
    assert grid.lon_bnds == ""
    assert grid.nlat == 80
    assert grid.nlon == 180
    assert grid.ncells == 14400

    # not implemented yet
    # assert self.mask


def test_grid_init_ds_tos_curvilinear(load_esgf_test_data):
    ds = xr.open_dataset(CMIP6_TOS_ONE_TIME_STEP, use_cftime=True)
    grid = Grid(ds=ds)

    assert grid.format == "CF"
    assert grid.source == "Dataset"
    assert grid.lat == ds.latitude.name
    assert grid.lon == ds.longitude.name
    assert grid.type == "curvilinear"
    assert grid.extent == "global"
    assert grid.lat_bnds == "vertices_latitude"
    assert grid.lon_bnds == "vertices_longitude"
    assert grid.nlat == 404  # this is number of 'j's
    assert grid.nlon == 802  # this is the number of 'i's
    assert grid.ncells == 324008

    # not implemented yet
    # assert self.mask


def test_grid_instructor_global():
    grid_instructor = (1.5, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"

    # check that grid_from_instructor sets the format to xESMF
    grid.grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 120
    assert grid.nlon == 240
    assert grid.ncells == 28800

    # not implemented yet
    # assert self.mask


def test_grid_instructor_2d_regional_change_lon():
    grid_instructor = (50, 240, 1.5, -90, 90, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "regional"

    # check that grid_from_instructor sets the format to xESMF
    grid.grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 120
    assert grid.nlon == 127
    assert grid.ncells == 15240

    # not implemented yet
    # assert self.mask


def test_grid_instructor_2d_regional_change_lat():
    grid_instructor = (0, 360, 1.5, -60, 50, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"

    # Extent in y-direction yet ignored, as not of importance
    #  for xesmf.Regridder. Extent in x-direction should be
    #  detected as "global"
    assert grid.extent == "global"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 74
    assert grid.nlon == 240
    assert grid.ncells == 17760

    # not implemented yet
    # assert self.mask


def test_grid_instructor_2d_regional_change_lon_and_lat():
    grid_instructor = (50, 240, 1.5, -60, 50, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "regional"

    # check that grid_from_instructor sets the format to xESMF
    grid.grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 74
    assert grid.nlon == 127
    assert grid.ncells == 9398

    # not implemented yet
    # assert self.mask


def test_grid_instructor_2d_global():
    grid_instructor = (0, 360, 1.5, -90, 90, 1.5)
    grid = Grid(grid_instructor=grid_instructor)

    assert grid.format == "CF"
    assert grid.source == "xESMF"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"

    # check that grid_from_instructor sets the format to xESMF
    grid.grid_from_instructor(grid_instructor)
    assert grid.format == "xESMF"

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 120
    assert grid.nlon == 240
    assert grid.ncells == 28800

    # not implemented yet
    # assert self.mask


def test_from_grid_id():
    # don't have the grid files
    # grid = Grid(grid_id="ERA-40")

    # assert grid.format == "CF"
    # assert grid.source == "Predefined_ERA-40"
    # assert grid.lat == "lat"
    # assert grid.lon == "lon"
    # assert grid.type == "regular_lat_lon"
    # assert grid.extent == "global"
    # assert grid.lat_bnds == "lat_bnds"
    # assert grid.lon_bnds == "lon_bnds"
    # assert grid.nlat = 145
    # assert grid.nlon = 288
    # assert grid.ncells = 41760

    # not implemented yet
    # assert self.mask0
    pass


def test_subsetted_grid():
    ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)

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

    assert grid.lat_bnds == ds.lat_bnds.name
    assert grid.lon_bnds == ds.lon_bnds.name
    assert grid.nlat == 35
    assert grid.nlon == 88
    assert grid.ncells == 3080

    # not implemented yet
    # assert self.mask
