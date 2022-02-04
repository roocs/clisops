import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import xarray as xr
from roocs_grids import get_grid_file, grid_dict

import clisops
from clisops import CONFIG
from clisops.core.regrid import XESMF_MINIMUM_VERSION, weights_cache_init, xe
from clisops.ops.regrid import regrid

from .._common import (
    CMIP5_MRSOS_ONE_TIME_STEP,
    CMIP6_OCE_HALO_CNRM,
    CMIP6_TOS_ONE_TIME_STEP,
)

XESMF_IMPORT_MSG = (
    f"xESMF >= {XESMF_MINIMUM_VERSION} is needed for regridding functionalities."
)


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


def _load_ds(fpath):
    return xr.open_mfdataset(fpath)


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_basic(tmpdir, load_esgf_test_data, tmp_path):
    "Test a basic regridding operation."
    fpath = CMIP5_MRSOS_ONE_TIME_STEP
    basename = os.path.splitext(os.path.basename(fpath))[0]
    method = "nearest_s2d"

    weights_cache_init(Path(tmp_path, "weights"))

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


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_grid_as_none(tmpdir, load_esgf_test_data, tmp_path):
    """
    Test behaviour when none passed as method and grid -
    should use the default regridding.
    """
    fpath = CMIP5_MRSOS_ONE_TIME_STEP

    weights_cache_init(Path(tmp_path, "weights"))

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


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
@pytest.mark.parametrize("grid_id", sorted(grid_dict))
def test_regrid_regular_grid_to_all_roocs_grids(
    tmpdir, load_esgf_test_data, grid_id, tmp_path
):
    "Test regridded a regular lat/lon field to all roocs grid types."
    fpath = CMIP5_MRSOS_ONE_TIME_STEP
    basename = os.path.splitext(os.path.basename(fpath))[0]
    method = "nearest_s2d"

    weights_cache_init(Path(tmp_path, "weights"))

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


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_keep_attrs(load_esgf_test_data, tmp_path):
    "Test if dataset and variable attributes are kept / removed as specified."
    fpath = CMIP6_TOS_ONE_TIME_STEP
    method = "nearest_s2d"

    weights_cache_init(Path(tmp_path, "weights"))

    ds = xr.open_dataset(fpath).isel(time=0)

    # regrid - keeping all attributes
    result = regrid(
        ds,
        method=method,
        adaptive_masking_threshold=-1,
        grid="2pt5deg",
        output_type="xarray",
    )

    # regrid - scrapping attributes
    result_na = regrid(
        ds,
        method=method,
        adaptive_masking_threshold=-1,
        grid="2pt5deg",
        output_type="xarray",
        keep_attrs=False,
    )

    ds_remap = result[0]
    ds_remap_na = result_na[0]

    assert "tos" in ds_remap and "tos" in ds_remap_na
    assert all([key not in ds_remap_na.tos.attrs.keys() for key in ds.tos.attrs.keys()])
    assert all([key in ds_remap.tos.attrs.keys() for key in ds.tos.attrs.keys()])
    assert all(
        [
            key not in ds_remap_na.attrs.keys()
            for key in ds.attrs.keys()
            if key not in ["grid", "grid_label"]
        ]
    )
    # todo: remove the restriction when nominal_resolution of the target grid is calculated in core/regrid.py
    assert all(
        [
            key in ds_remap.attrs.keys()
            for key in ds.attrs.keys()
            if key not in ["nominal_resolution"]
        ]
    )


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_halo_simple(load_esgf_test_data, tmp_path):
    "Test regridding with a simple halo."
    fpath = CMIP6_TOS_ONE_TIME_STEP
    ds = xr.open_dataset(fpath).isel(time=0)

    weights_cache_init(Path(tmp_path, "weights"))

    ds_out = regrid(
        ds,
        method="conservative",
        adaptive_masking_threshold=-1,
        grid=5,
        output_type="xarray",
    )[0]

    assert ds_out.attrs["regrid_operation"] == "conservative_402x800_36x72"


@pytest.mark.xfail
@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_halo_adv(load_esgf_test_data, tmp_path):
    "Test regridding of dataset with a more complex halo."
    fpath = CMIP6_OCE_HALO_CNRM
    ds = xr.open_dataset(fpath).isel(time=0)

    weights_cache_init(Path(tmp_path, "weights"))

    ds_out = regrid(
        ds,
        method="conservative",
        adaptive_masking_threshold=-1,
        grid=5,
        output_type="xarray",
    )[0]

    # After the halo can be properly removed (maybe 1049x1440), the test can be updated
    assert ds_out.attrs["regrid_operation"] == "conservative_1050x1442_36x72"


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_same_grid_exception(tmpdir, tmp_path):
    "Test that an exception is raised when source and target grid are the same."
    fpath = get_grid_file("0pt25deg_era5")

    weights_cache_init(Path(tmp_path, "weights"))

    with pytest.raises(
        Exception,
        match="The selected source and target grids are the same. No regridding operation required.",
    ):
        regrid(
            fpath,
            method="conservative",
            adaptive_masking_threshold=0.5,
            grid="0pt25deg_era5_lsm",
            output_dir=tmpdir,
            output_type="netcdf",
            file_namer="standard",
        )
