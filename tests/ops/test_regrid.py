import os
import sys
from unittest.mock import Mock

import pytest
import xarray as xr
from roocs_grids import grid_dict

import clisops
from clisops import CONFIG
from clisops.ops.regrid import regrid

from .._common import CMIP5_MRSOS_ONE_TIME_STEP


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath)


def test_regrid_basic(tmpdir, load_esgf_test_data):
    "Test a basic regridding operation."
    fpath = CMIP5_MRSOS_ONE_TIME_STEP
    basename = os.path.splitext(os.path.basename(fpath))[0]
    method = "nearest_s2d"

    result = regrid(
        fpath,
        method=method,
        adaptive_masking_threshold=0.5,
        grid="1deg",
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result, fname=f"{basename}-20051201_regrid-{method}-180x360_cells_grid.nc"
    )


def test_regrid_grid_as_none(tmpdir, load_esgf_test_data):
    """
    Test behaviour when none passed as method and grid -
    should use the default regridding.
    """
    fpath = CMIP5_MRSOS_ONE_TIME_STEP

    with pytest.raises(
        Exception,
        match="xarray.Dataset, grid_id or grid_instructor have to be specified as input.",
    ):
        regrid(
            fpath,
            grid=None,
            output_dir=tmpdir,
            output_type="netcdf",
            file_namer="standard",
        )


@pytest.mark.parametrize("grid_id", sorted(grid_dict))
def test_regrid_regular_grid_to_all_roocs_grids(tmpdir, load_esgf_test_data, grid_id):
    "Test regridded a regular lat/lon field to all roocs grid types."
    fpath = CMIP5_MRSOS_ONE_TIME_STEP
    basename = os.path.splitext(os.path.basename(fpath))[0]
    method = "nearest_s2d"

    result = regrid(
        fpath,
        method=method,
        adaptive_masking_threshold=0.5,
        grid=grid_id,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    nc_file = result[0]
    assert os.path.basename(nc_file).startswith(f"{basename}-20051201_regrid-{method}-")

    # Can we read the output file
    ds = xr.open_dataset(nc_file)
    assert "mrsos" in ds
    assert ds.mrsos.size > 100
