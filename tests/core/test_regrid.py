from pkg_resources import parse_version

try:
    import xesmf

    if parse_version(xesmf.__version__) < parse_version("0.6.0"):
        raise ImportError
except ImportError:
    xesmf = None

import cf_xarray as cfxr
import numpy as np
import pytest
import xarray as xr
from roocs_grids import get_grid_file

from clisops.core.regrid import Grid, Weights, regrid
from clisops.ops.subset import subset

from .._common import (
    CMIP6_ATM_VERT_ONE_TIMESTEP,
    CMIP6_OCE_HALO_CNRM,
    CMIP6_TAS_ONE_TIME_STEP,
    CMIP6_TAS_PRECISION_A,
    CMIP6_TAS_PRECISION_B,
    CMIP6_TOS_ONE_TIME_STEP,
)

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


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
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


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
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


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
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


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
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


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
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


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
def test_grid_from_ds_adaptive_extent(load_esgf_test_data):
    "Test that the extent is evaluated as global for original and derived adaptive grid."
    dsA = xr.open_dataset(CMIP6_TOS_ONE_TIME_STEP, use_cftime=True)
    dsB = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)

    gA = Grid(ds=dsA)
    gB = Grid(ds=dsB)
    gAa = Grid(ds=dsA, grid_id="adaptive")
    gBa = Grid(ds=dsB, grid_id="adaptive")

    assert gA.extent == "global"
    assert gB.extent == "global"
    assert gAa.extent == "global"
    assert gBa.extent == "global"


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
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
    g025 = Grid(grid_id="0pt25deg_era5")
    g025_lsm = Grid(grid_id="0pt25deg_era5_lsm")

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


def test_detect_collapsing_weights(load_esgf_test_data):
    "Test that collapsing cells are properly identified"
    # todo: the used datasets might not be appropriate when the halo gets more properly removed
    dsA = xr.open_dataset(CMIP6_OCE_HALO_CNRM, use_cftime=True)
    dsB = xr.open_dataset(CMIP6_TOS_ONE_TIME_STEP, use_cftime=True)

    gA = Grid(ds=dsA)
    gB = Grid(ds=dsB)

    assert gA.contains_collapsing_cells
    assert not gB.contains_collapsing_cells


def test_Weights_init_with_collapsing_cells(load_esgf_test_data):
    "Test the creation of remapping weights for a grid containing collapsing cells"
    # todo: the used dataset might not be appropriate if the halo gets more properly removed
    # ValueError: ESMC_FieldRegridStore failed with rc = 506. Please check the log files (named "*ESMF_LogFile").
    ds = xr.open_dataset(CMIP6_OCE_HALO_CNRM, use_cftime=True)

    g = Grid(ds=ds)
    g_out = Grid(grid_instructor=(10.0,))
    Weights(g, g_out, method="bilinear")


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


# test all methods
@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
class TestWeights:
    def test_grids_in_and_out_bilinear(self):
        ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
        grid_in = Grid(ds=ds)

        assert grid_in.extent == "global"

        grid_instructor_out = (0, 360, 1.5, -90, 90, 1.5)
        grid_out = Grid(grid_instructor=grid_instructor_out)

        assert grid_out.extent == "global"

        w = Weights(grid_in=grid_in, grid_out=grid_out, method="bilinear")

        assert w.method == "bilinear"
        assert (
            w.id
            == "fd3943b6c22982be62e2bb731ec5a760_a2a2ea3333c5c49211f770b494a27ae5_True_None_bilinear"
        )
        assert w.periodic
        assert w.id in w.filename
        assert "xESMF_v" in w.tool
        assert w.format == "xESMF"

        # default file_name = method_inputgrid_outputgrid_periodic"
        assert w.regridder.filename == "bilinear_80x180_120x240_peri.nc"

    def test_grids_in_and_out_conservative(self):
        ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
        grid_in = Grid(ds=ds)

        assert grid_in.extent == "global"

        grid_instructor_out = (0, 360, 1.5, -90, 90, 1.5)
        grid_out = Grid(grid_instructor=grid_instructor_out)

        assert grid_out.extent == "global"

        w = Weights(grid_in=grid_in, grid_out=grid_out, method="conservative")

        assert w.method == "conservative"
        assert (
            w.id
            == "fd3943b6c22982be62e2bb731ec5a760_a2a2ea3333c5c49211f770b494a27ae5_True_None_conservative"
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


@pytest.mark.skipif(
    xesmf is None, reason="xESMF >= 0.6.0 is needed for regridding functionalities."
)
class TestRegrid:
    def _setup(self):
        if hasattr(self, "setup_done"):
            return

        self.ds = xr.open_dataset(CMIP6_TAS_ONE_TIME_STEP, use_cftime=True)
        self.grid_in = Grid(ds=self.ds)

        self.grid_instructor_out = (0, 360, 1.5, -90, 90, 1.5)
        self.grid_out = Grid(grid_instructor=self.grid_instructor_out)
        self.setup_done = True

    def test_adaptive_masking(self, load_esgf_test_data):
        self._setup()
        w = Weights(grid_in=self.grid_in, grid_out=self.grid_out, method="conservative")
        r = regrid(self.grid_in, self.grid_out, w, adaptive_masking_threshold=0.7)
        print(r)

    def test_no_adaptive_masking(self, load_esgf_test_data):
        self._setup()
        w = Weights(grid_in=self.grid_in, grid_out=self.grid_out, method="bilinear")
        r = regrid(self.grid_in, self.grid_out, w, adaptive_masking_threshold=-1.0)
        print(r)


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
    dsA["areacella"] = areacella
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
