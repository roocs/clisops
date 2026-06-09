import os
from pathlib import Path

import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr
from roocs_grids import get_grid_file, grid_dict

from clisops.core.regrid import XESMF_MINIMUM_VERSION, weights_cache_init, xe
from clisops.exceptions import InvalidParameterValue
from clisops.ops.regrid import regrid
from clisops.ops.subset import subset

XESMF_IMPORT_MSG = f"xESMF >= {XESMF_MINIMUM_VERSION} is needed for regridding functionalities."


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [os.path.basename(_) for _ in result]


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_basic(tmpdir, tmp_path, mini_esgf_data):
    """Test a basic regridding operation."""
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]
    basename = Path(fpath).stem
    method = "nearest_s2d"

    weights_cache_init(Path(tmp_path, "weights"))

    result = regrid(
        fpath,
        method=method,
        adaptive_masking_threshold=0.5,
        grid="auto",
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname=f"{basename}-20051201_regrid-{method}-144x192_cells_grid.nc")


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_to_ds(tmpdir, tmp_path, mini_esgf_data):
    """Test a basic regridding operation to another dataset."""
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]
    basename = Path(fpath).stem
    ds_tgt = xe.util.grid_2d(10, 20, 2, 10, 20, 2)
    method = "nearest_s2d"

    weights_cache_init(Path(tmp_path, "weights"))

    result = regrid(
        fpath,
        method=method,
        grid=ds_tgt,
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(result, fname=f"{basename}-20051201_regrid-{method}-5x5_cells_grid.nc")


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_grid_as_none(tmpdir, tmp_path, mini_esgf_data):
    """
    Test behaviour when none passed as method and grid.

    Should use the default regridding.
    """
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]

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
def test_regrid_invalid_grid_id(tmpdir, tmp_path, mini_esgf_data):
    """Test behaviour when an invalid grid_id is passed as target grid."""
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]

    weights_cache_init(Path(tmp_path, "weights"))

    with pytest.raises(
        KeyError,
        match="The grid_id 'invalid' you specified does not exist.",
    ):
        regrid(
            fpath,
            grid="invalid",
            output_dir=tmpdir,
            output_type="netcdf",
            file_namer="standard",
        )


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_invalid_grid_instructor(tmpdir, tmp_path, mini_esgf_data):
    """Test behaviour when an invalid grid_instructor is passed as target grid."""
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]

    weights_cache_init(Path(tmp_path, "weights"))

    with pytest.raises(
        InvalidParameterValue,
        match="The grid_instructor has to be a tuple of length 1, 2, 3 or 6.",
    ):
        regrid(
            fpath,
            grid=(1, 2, 3, 4, 5),
            output_dir=tmpdir,
            output_type="netcdf",
            file_namer="standard",
        )


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_no_ds_da(tmpdir, tmp_path):
    """Test behaviour when no Dataset or DataArray is passed."""
    weights_cache_init(Path(tmp_path, "weights"))

    with pytest.raises(
        InvalidParameterValue,
        match="An xarray.Dataset or xarray.DataArray has to be provided as input for the source grid.",
    ):
        regrid(
            np.array([1, 2, 3]),
            grid="1deg",
            output_dir=tmpdir,
            output_type="netcdf",
            file_namer="standard",
        )


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_invalid_mask(tmpdir, tmp_path, mini_esgf_data):
    """Test behaviour when an invalid masking option is passed."""
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]

    weights_cache_init(Path(tmp_path, "weights"))

    with pytest.raises(
        ValueError,
        match="mask must be one of 'land', 'ocean' or None, not 'invalid'",
    ):
        regrid(
            fpath,
            grid="1deg_lsm",
            mask="invalid",
            output_dir=tmpdir,
            output_type="netcdf",
            file_namer="standard",
        )


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_invalid_method(tmpdir, tmp_path, mini_esgf_data):
    """Test behaviour when an invalid regridding method is passed."""
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]

    weights_cache_init(Path(tmp_path, "weights"))

    with pytest.raises(
        Exception,
        match="selected regridding method is not supported. Please choose one of",
    ):
        regrid(
            fpath,
            grid="1deg_lsm",
            method="invalid",
            output_dir=tmpdir,
            output_type="netcdf",
            file_namer="standard",
        )


@pytest.mark.slow
@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
@pytest.mark.parametrize("grid_id", sorted(grid_dict))
def test_regrid_regular_grid_to_all_roocs_grids(tmpdir, tmp_path, grid_id, mini_esgf_data):
    """Test for regridding a regular lat/lon field to all roocs grid types."""
    fpath = mini_esgf_data["CMIP5_MRSOS_ONE_TIME_STEP"]
    basename = Path(fpath).stem
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


@pytest.mark.slow
@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_subset_and_regrid_erroneous_cf_units_cmip5(tmpdir, mini_esgf_data, tmp_path):
    """Test subset and regrid ds with erroneous cf units."""
    fpath = mini_esgf_data["CMIP5_WRONG_CF_UNITS"]
    basename = Path(fpath).stem
    method = "conservative"
    weights_cache_init(Path(tmp_path, "weights"))

    # subset
    result = subset(
        ds=fpath,
        area=(0, -10.0, 20.0, 10.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)

    # regrid
    result = regrid(
        result[0],
        method=method,
        adaptive_masking_threshold=0.5,
        grid="1deg",
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    nc_file = result[0]
    assert (
        os.path.basename(nc_file)
        == f"{basename.replace('Omon', 'mon')}16-20060116_regrid-{method}-180x360_cells_grid.nc"
    )

    # Can we read the output file
    ds = xr.open_dataset(nc_file)
    assert "zos" in ds
    assert ds.zos.size == 360 * 180
    assert ds.zos.count() == 163


@pytest.mark.slow
@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
@pytest.mark.parametrize("dset", ["ATLAS_v1_CORDEX", "ATLAS_v1_EOBS_GRID", "ATLAS_v0_CORDEX_ANT"])
def test_regrid_ATLAS_datasets(tmpdir, dset, mini_esgf_data):
    """Test regridding for several ATLAS datasets."""
    result = regrid(
        ds=mini_esgf_data[dset],
        method="bilinear",
        adaptive_masking_threshold=0.5,
        grid="0pt5deg_lsm",
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )
    assert os.path.basename(result[0]).endswith("_regrid-bilinear-360x720_cells_grid.nc")


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_ATLAS_CORDEX(tmpdir, caplog, mini_esgf_data):
    """Test regridding for ATLAS CORDEX dataset."""
    ds = xr.open_dataset(
        mini_esgf_data["ATLAS_v0_CORDEX_ANT"],
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    # Might trigger KeyError in future netcdf-c versions
    # PR: https://github.com/Unidata/netcdf-c/pull/2716
    for cvar in [
        "member_id",
        "rcm_variant",
        "rcm_model",
        "rcm_institution",
        "gcm_variant",
        "gcm_model",
        "gcm_institution",
    ]:
        assert cvar in ds

    result = regrid(
        ds=ds,
        method="nearest_s2d",
        adaptive_masking_threshold=0.5,
        grid="1deg",
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )
    assert (
        os.path.basename(result[0])
        == "tnn_CORDEX-ANT_rcp45_mon_20060101-20060101_regrid-nearest_s2d-180x360_cells_grid.nc"
    )


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_keep_attrs(tmp_path, mini_esgf_data):
    """Test if dataset and variable attributes are kept / removed as specified."""
    fpath = mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"]
    method = "nearest_s2d"

    weights_cache_init(Path(tmp_path, "weights"))

    ds = xr.open_dataset(fpath).isel(time=0)

    # regrid - preserve input attrs
    result = regrid(
        ds,
        method=method,
        adaptive_masking_threshold=-1,
        grid="2deg_lsm",
        output_type="xarray",
    )

    # regrid - scrapping attrs
    result_na = regrid(
        ds,
        method=method,
        adaptive_masking_threshold=-1,
        grid="2deg_lsm",
        output_type="xarray",
        keep_attrs=False,
    )

    # regrid - keep target attrs
    result_ta = regrid(
        ds,
        method=method,
        adaptive_masking_threshold=-1,
        grid="2deg_lsm",
        output_type="xarray",
        keep_attrs="target",
    )

    ds_remap = result[0]
    ds_remap_na = result_na[0]
    ds_remap_ta = result_ta[0]

    assert "tos" in ds_remap and "tos" in ds_remap_na and "tos" in ds_remap_ta
    assert all([key in ds_remap.tos.attrs.keys() for key in ds.tos.attrs.keys()])
    assert all([key in ds_remap.attrs.keys() for key in ds.attrs.keys() if key not in ["nominal_resolution"]])
    # todo: remove the restriction when nominal_resolution of the target grid is calculated in core/regrid.py
    assert all([key not in ds_remap_na.tos.attrs.keys() for key in ds.tos.attrs.keys()])
    assert all([key not in ds_remap_na.attrs.keys() for key in ds.attrs.keys() if key not in ["grid", "grid_label"]])
    assert all([key in ds_remap_ta.tos.attrs.keys() for key in ds.tos.attrs.keys()])
    assert all(
        [
            key not in ds_remap_ta.attrs.keys()
            for key in ds.attrs.keys()
            if key not in ["source", "Conventions", "history", "NCO", "grid", "grid_label"]
        ]
    )


@pytest.mark.slow
class TestRegridHalo:
    @pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
    def test_regrid_halo_simple(self, tmp_path, mini_esgf_data):
        """Test regridding with a simple halo."""
        fpath = mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"]
        ds = xr.open_dataset(fpath).isel(time=0)

        weights_cache_init(Path(tmp_path, "weights"))

        ds_out = regrid(
            ds,
            method="conservative",
            adaptive_masking_threshold=-1,
            grid=5,
            output_type="xarray",
        )[0]

        assert ds_out.attrs["regrid_operation"] == "conservative_404x802_36x72"

    @pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
    def test_regrid_halo_adv(self, tmp_path, mini_esgf_data):
        """Test regridding of dataset with a more complex halo."""
        fpath = mini_esgf_data["CMIP6_OCE_HALO_CNRM"]
        ds = xr.open_dataset(fpath).isel(time=0)

        weights_cache_init(Path(tmp_path, "weights"))

        ds_out = regrid(
            ds,
            method="conservative",
            adaptive_masking_threshold=-1,
            grid=5,
            output_type="xarray",
        )[0]

        assert ds_out.attrs["regrid_operation"] == "conservative_1050x1442_36x72"


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_shifted_lon_frame(tmp_path, mini_esgf_data):
    """Test regridding of dataset with shifted longitude frame."""
    fpath = mini_esgf_data["CMIP6_IITM_EXTENT"]
    ds = xr.open_dataset(fpath).isel(time=0)

    weights_cache_init(Path(tmp_path, "weights"))

    ds_out = regrid(
        ds,
        method="bilinear",
        adaptive_masking_threshold=-1,
        grid=5,
        output_type="xarray",
    )[0]

    assert ds_out.attrs["regrid_operation"] == "bilinear_200x360_36x72_peri"


@pytest.mark.slow
@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_same_grid_exception(tmpdir, tmp_path):
    """Test that a warning is issued when source and target grid are the same."""
    fpath = get_grid_file("0pt25deg_era5")
    ds = xr.open_dataset(fpath)

    weights_cache_init(Path(tmp_path, "weights"))

    with pytest.warns(
        UserWarning,
        match="The selected source and target grids are the same.",
    ):
        ds_regrid = regrid(
            fpath,
            method="conservative",
            adaptive_masking_threshold=0.5,
            grid="0pt25deg_era5_lsm",
            output_dir=tmpdir,
            output_type="xarray",
            file_namer="standard",
        )[0]
    # It is expected that the input ds is simply passed through
    xr.testing.assert_identical(ds, ds_regrid)


@pytest.mark.skipif(xe is None, reason=XESMF_IMPORT_MSG)
def test_regrid_cmip6_nc_consistent_bounds_and_coords(tmpdir, mini_esgf_data):
    """Tests clisops regrid function and check metadata added by xarray"""
    result = regrid(
        ds=mini_esgf_data["CMIP6_ATM_VERT_ONE_TIMESTEP"],
        method="nearest_s2d",
        grid=10.0,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="standard",
    )
    res = xr.open_mfdataset(result)
    # check fill value in bounds
    assert "_FillValue" not in res.lat_bnds.encoding
    assert "_FillValue" not in res.lon_bnds.encoding
    assert "_FillValue" not in res.time_bnds.encoding
    assert "_FillValue" not in res.lev_bnds.encoding
    assert "_FillValue" not in res.ap_bnds.encoding
    assert "_FillValue" not in res.b_bnds.encoding
    # check fill value in coordinates
    assert "_FillValue" not in res.time.encoding
    assert "_FillValue" not in res.lat.encoding
    assert "_FillValue" not in res.lon.encoding
    assert "_FillValue" not in res.lev.encoding
    assert "_FillValue" not in res.ap.encoding
    assert "_FillValue" not in res.b.encoding
    # check coordinates in bounds
    assert "coordinates" not in res.lat_bnds.encoding
    assert "coordinates" not in res.lon_bnds.encoding
    assert "coordinates" not in res.time_bnds.encoding
    assert "coordinates" not in res.lev_bnds.encoding
    assert "coordinates" not in res.ap_bnds.encoding
    assert "coordinates" not in res.b_bnds.encoding
    # Check coordinates not in variable attributes
    assert "coordinates" not in res.o3.encoding
    assert "coordinates" not in res.ps.encoding
