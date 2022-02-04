import os
import warnings
from glob import glob
from pathlib import Path

import cf_xarray as cfxr
import numpy as np
import pytest
import xarray as xr
from pkg_resources import parse_version
from roocs_grids import get_grid_file

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

from .._common import (
    CMIP6_ATM_VERT_ONE_TIMESTEP,
    CMIP6_OCE_HALO_CNRM,
    CMIP6_TAS_ONE_TIME_STEP,
    CMIP6_TAS_PRECISION_A,
    CMIP6_TAS_PRECISION_B,
    CMIP6_TOS_ONE_TIME_STEP,
    CMIP6_UNSTR_ICON_A,
    CORDEX_TAS_NO_BOUNDS,
)

try:
    import xesmf

    if parse_version(xesmf.__version__) < parse_version(XESMF_MINIMUM_VERSION):
        raise ImportError
except ImportError:
    xesmf = None


# test from grid_id --predetermined
# test different types of grid e.g. irregular, not supported type
# test for errors e.g.
# no lat/lon in dataset
# more than one latitude/longitude
# grid instructor tuple not correct length


XESMF_IMPORT_MSG = (
    f"xESMF >= {XESMF_MINIMUM_VERSION} is needed for regridding functionalities."
)


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
    assert grid.lat_bnds is None
    assert grid.lon_bnds is None
    assert grid.nlat == 80
    assert grid.nlon == 180
    assert grid.ncells == 14400


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
    assert grid.nlat == 402  # 404 incl halo  # this is number of 'j's
    assert grid.nlon == 800  # 802 incl halo  # this is the number of 'i's
    assert grid.ncells == 321600  # 324008 incl halo

    # not implemented yet
    # assert self.mask


def test_grid_init_ds_tas_irregular(load_esgf_test_data):
    ds = xr.open_dataset(CMIP6_UNSTR_ICON_A, use_cftime=True)
    grid = Grid(ds=ds)

    assert grid.format == "CF"
    assert grid.source == "Dataset"
    assert grid.lat == ds.latitude.name
    assert grid.lon == ds.longitude.name
    assert grid.type == "irregular"
    assert grid.extent == "global"
    assert grid.lat_bnds == "latitude_bnds"
    assert grid.lon_bnds == "longitude_bnds"
    assert grid.ncells == 20480
    print(grid.contains_collapsing_cells)

    # not implemented yet
    # assert self.mask


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

    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 74
    assert grid.nlon == 240
    assert grid.ncells == 17760

    # not implemented yet
    # assert self.mask


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
    "Test to create grid from grid_id"
    grid = Grid(grid_id="ERA-40")

    assert grid.format == "CF"
    assert grid.source == "Predefined_ERA-40"
    assert grid.lat == "lat"
    assert grid.lon == "lon"
    assert grid.type == "regular_lat_lon"
    assert grid.extent == "global"
    assert grid.lat_bnds == "lat_bnds"
    assert grid.lon_bnds == "lon_bnds"
    assert grid.nlat == 145
    assert grid.nlon == 288
    assert grid.ncells == 41760

    # not implemented yet
    # assert self.mask0


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_grid_from_ds_adaptive_extent(load_esgf_test_data):
    "Test that the extent is evaluated as global for original and derived adaptive grid."
    dsA = xr.open_dataset(CMIP6_TOS_ONE_TIME_STEP, use_cftime=True)
    dsB = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
    dsC = xr.open_dataset(CMIP6_UNSTR_ICON_A, use_cftime=True)

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


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_grid_from_ds_adaptive_reproducibility():
    "Test that the extent is evaluated as global for original and derived adaptive grid."
    fpathA = get_grid_file("0pt25deg")
    dsA = xr.open_dataset(fpathA, use_cftime=True)
    fpathB = get_grid_file("1deg")
    dsB = xr.open_dataset(fpathB, use_cftime=True)

    gAa = Grid(ds=dsA, grid_id="adaptive")
    gA = Grid(grid_id="0pt25deg")
    print(repr(gAa))
    print(repr(gA))
    print(gAa.ds.lon[715:735])
    print(gA.ds.lon[715:735])

    gBa = Grid(ds=dsB, grid_id="adaptive")
    gB = Grid(grid_id="1deg")
    print(gBa.ds.lon[170:190])
    print(gB.ds.lon[170:190])
    print(repr(gBa))
    print(repr(gB))

    assert gA.extent == "global"
    assert gA.compare_grid(gAa)
    assert gB.extent == "global"
    assert gB.compare_grid(gBa)


def test_compare_grid_same_resolution():
    "Test that two grids of same resolution from different sources evaluate as the same grid"
    ds025 = xr.open_dataset(get_grid_file("0pt25deg_era5"))
    g025 = Grid(grid_id="0pt25deg_era5", compute_bounds=True)
    g025_lsm = Grid(grid_id="0pt25deg_era5_lsm", compute_bounds=True)

    assert g025.compare_grid(g025_lsm)
    assert g025.compare_grid(ds025)
    assert g025_lsm.compare_grid(ds025)


def test_compare_grid_diff_in_precision(load_esgf_test_data):
    "Test that the same grid stored with different precision is evaluated as the same grid"
    dsA = xr.open_dataset(CMIP6_TAS_PRECISION_A, use_cftime=True)
    dsB = xr.open_dataset(CMIP6_TAS_PRECISION_B, use_cftime=True)

    gA = Grid(ds=dsA)
    gB = Grid(ds=dsB)

    assert gA.compare_grid(gB)


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_compare_grid_hash_dict_and_verbose(capfd):
    "Test Grid.hash_dict keys and Grid.compare_grid verbose option"
    gA = Grid(grid_instructor=(1.0, 0.5))
    gB = Grid(grid_instructor=(1.0,))
    is_equal = gA.compare_grid(gB, verbose=True)
    stdout, stderr = capfd.readouterr()

    assert stderr == ""
    assert stdout == "The two grids differ in their respective lat, lat_bnds.\n"
    assert not is_equal
    assert len(gA.hash_dict) == 5
    assert list(gA.hash_dict.keys()) == ["lat", "lon", "lat_bnds", "lon_bnds", "mask"]


def test_detect_collapsing_cells(load_esgf_test_data):
    "Test that collapsing cells are properly identified"
    # todo: the used datasets might not be appropriate when the halo gets more properly removed
    dsA = xr.open_dataset(CMIP6_OCE_HALO_CNRM, use_cftime=True)
    dsB = xr.open_dataset(CMIP6_TOS_ONE_TIME_STEP, use_cftime=True)

    gA = Grid(ds=dsA)
    gB = Grid(ds=dsB)

    assert gA.contains_collapsing_cells
    assert not gB.contains_collapsing_cells


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_Weights_init_with_collapsing_cells(tmp_path, load_esgf_test_data):
    "Test the creation of remapping weights for a grid containing collapsing cells"
    # todo: the used dataset might not be appropriate if the halo gets more properly removed
    # ValueError: ESMC_FieldRegridStore failed with rc = 506. Please check the log files (named "*ESMF_LogFile").
    ds = xr.open_dataset(CMIP6_OCE_HALO_CNRM, use_cftime=True)

    g = Grid(ds=ds)
    g_out = Grid(grid_instructor=(10.0,))

    weights_cache_init(Path(tmp_path, "weights"))
    Weights(g, g_out, method="bilinear")


def test_subsetted_grid(load_esgf_test_data):
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


def test_drop_vars_transfer_coords(load_esgf_test_data):
    "Test for Grid methods _drop_vars ad _transfer_coords"
    ds = xr.open_dataset(CMIP6_ATM_VERT_ONE_TIMESTEP)
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


def test_calculate_bounds_curvilinear(load_esgf_test_data):
    "Test for bounds calculation for curvilinear grid"
    ds = xr.open_dataset(CORDEX_TAS_NO_BOUNDS).isel(
        {"rlat": range(10), "rlon": range(10)}
    )
    g = Grid(ds=ds, compute_bounds=True)
    assert g.lat_bnds is not None
    assert g.lon_bnds is not None


def test_centers_within_bounds_curvilinear(load_esgf_test_data):
    "Test for bounds calculation for curvilinear grid"
    ds = xr.open_dataset(CORDEX_TAS_NO_BOUNDS).isel(
        {"rlat": range(10), "rlon": range(10)}
    )
    g = Grid(ds=ds, compute_bounds=True)
    assert g.lat_bnds is not None
    assert g.lon_bnds is not None
    assert g.contains_collapsing_cells is False

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


def test_centers_within_bounds_regular_lat_lon():
    "Test for bounds calculation of regular lat lon grid"
    g = Grid(grid_id="0pt25deg_era5_lsm", compute_bounds=True)
    assert g.lat_bnds is not None
    assert g.lon_bnds is not None
    assert bool(g.contains_collapsing_cells) is False

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


# test all methods
@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
class TestWeights:
    def test_grids_in_and_out_bilinear(self, tmp_path):
        ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
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
            == "8edb4ee828dbebc2dc8e193281114093_bf73249f1725126ad3577727f3652019_peri_bilinear"
        )
        assert w.periodic
        assert w.id in w.filename
        assert "xESMF_v" in w.tool
        assert w.format == "xESMF"

        # default file_name = method_inputgrid_outputgrid_periodic"
        assert w.regridder.filename == "bilinear_80x180_120x240_peri.nc"

    def test_grids_in_and_out_conservative(self, tmp_path):
        ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
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
            == "8edb4ee828dbebc2dc8e193281114093_bf73249f1725126ad3577727f3652019_peri_conservative"
        )
        assert (
            w.periodic != w.regridder.periodic
        )  # xESMF resets periodic to False for conservative weights
        assert w.id in w.filename
        assert "xESMF_v" in w.tool
        assert w.format == "xESMF"

        # default file_name = method_inputgrid_outputgrid_periodic"
        assert w.regridder.filename == "conservative_80x180_120x240.nc"

    def test_from_id(self):
        pass

    def test_from_disk(self):
        pass

    def test_conservative_no_bnds(self, load_esgf_test_data, tmp_path):
        "Test whether exception is raised when no bounds present for conservative remapping."
        ds = xr.open_dataset(CORDEX_TAS_NO_BOUNDS)
        gi = Grid(ds=ds)
        go = Grid(grid_id="1deg", compute_bounds=True)

        assert gi.lat_bnds is None
        assert gi.lon_bnds is None
        assert go.lat_bnds is not None
        assert go.lon_bnds is not None

        with pytest.raises(
            Exception,
            match="For conservative remapping, horizontal grid bounds have to be defined for the input and output grid.",
        ):
            weights_cache_init(Path(tmp_path, "weights"))
            Weights(grid_in=gi, grid_out=go, method="conservative")


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
class TestRegrid:
    def _setup(self):
        if hasattr(self, "setup_done"):
            return

        self.ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
        self.grid_in = Grid(ds=self.ds)

        self.grid_instructor_out = (0, 360, 1.5, -90, 90, 1.5)
        self.grid_out = Grid(grid_instructor=self.grid_instructor_out)
        self.setup_done = True

    def test_adaptive_masking(self, load_esgf_test_data, tmp_path):
        self._setup()
        weights_cache_init(Path(tmp_path, "weights"))
        w = Weights(grid_in=self.grid_in, grid_out=self.grid_out, method="conservative")
        r = regrid(self.grid_in, self.grid_out, w, adaptive_masking_threshold=0.7)
        print(r)

    def test_no_adaptive_masking(self, load_esgf_test_data, tmp_path):
        self._setup()
        weights_cache_init(Path(tmp_path, "weights"))
        w = Weights(grid_in=self.grid_in, grid_out=self.grid_out, method="bilinear")
        r = regrid(self.grid_in, self.grid_out, w, adaptive_masking_threshold=-1.0)
        print(r)


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_cache_init_and_flush(tmp_path):
    "Test of the cache init and flush functionalities"

    weights_dir = Path(tmp_path, "clisops_weights")
    weights_cache_init(weights_dir)

    gi = Grid(grid_instructor=20)
    go = Grid(grid_instructor=10)
    Weights(grid_in=gi, grid_out=go, method="nearest_s2d")

    flist = sorted(os.path.basename(f) for f in glob(f"{weights_dir}/*"))
    flist_ref = [
        "grid_4d324aecaa8302ab0f2f212e9821b00f.nc",
        "grid_96395935b4e81f2a5b55970bd920d82c.nc",
        "weights_4d324aecaa8302ab0f2f212e9821b00f_96395935b4e81f2a5b55970bd920d82c_peri_nearest_s2d.json",
        "weights_4d324aecaa8302ab0f2f212e9821b00f_96395935b4e81f2a5b55970bd920d82c_peri_nearest_s2d.nc",
    ]
    assert flist == flist_ref

    weights_cache_flush()
    assert glob(f"{weights_dir}/*") == []


def test_data_vars_coords_reset_and_cfxr(load_esgf_test_data):
    dsA = xr.open_dataset(CMIP6_ATM_VERT_ONE_TIMESTEP)

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
    dsA.update({"areacella": areacella})
    dsB = xr.decode_cf(dsA, decode_coords="all")

    # Grid._set_data_vars_and_coords should (re)set coords appropriately
    gA = Grid(ds=dsA)
    gB = Grid(ds=dsB)

    # cf_xarray should be able to identify important attributes and present both datasets equally
    assert gA.compare_grid(gB)
    assert gA.ds.cf.cell_measures == gB.ds.cf.cell_measures
    assert gA.ds.o3.cf.cell_measures == gB.ds.o3.cf.cell_measures
    assert gA.ds.cf.formula_terms == gB.ds.cf.formula_terms
    assert gA.ds.o3.cf.formula_terms == gB.ds.o3.cf.formula_terms
    assert gA.ds.cf.bounds == gB.ds.cf.bounds
    assert str(gA.ds.cf) == str(gB.ds.cf)


@pytest.mark.skipif(xesmf is None, reason=XESMF_IMPORT_MSG)
def test_cache_lock_mechanism(load_esgf_test_data, tmp_path):
    """Test lock mechanism of local regrid weights cache."""
    ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)

    grid_in = Grid(ds=ds)
    grid_out = Grid(grid_instructor=10)

    # First round - creating the weights should work without problems
    weights_cache_init(Path(tmp_path, "weights"))
    w = Weights(grid_in=grid_in, grid_out=grid_out, method="nearest_s2d")

    # Second round, but manually put lockfile in place
    LOCK_FILE = Path(tmp_path, "weights", w.filename + ".lock")
    lock = FileLock(LOCK_FILE)
    lock.acquire(timeout=10)

    # Fail test if lockfile is not recognized
    with pytest.warns(UserWarning, match="lockfile") as issuedWarnings:
        Weights(grid_in=grid_in, grid_out=grid_out, method="nearest_s2d")
        if not issuedWarnings:
            raise Exception("Lockfile not recognized/ignored.")
        else:
            assert len(issuedWarnings) == 3
            # todo: assert 2 issued warnings if core.regrid.Weights.loadFromCache is getting removed
            # for issuedWarning in issuedWarnings:
            #    print(issuedWarning.message)

    lock.release()
