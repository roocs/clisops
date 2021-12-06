import os
import tempfile
from pathlib import Path

import xarray as xr
from jinja2 import Template

ROOCS_CFG = Path(tempfile.gettempdir(), "roocs.ini").as_posix()
TESTS_HOME = Path(__file__).parent.absolute().as_posix()
DEFAULT_CMIP5_ARCHIVE_BASE = Path(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip5/data"
).as_posix()
REAL_C3S_CMIP5_ARCHIVE_BASE = "/gws/nopw/j04/cp4cds1_vol1/data/"
DEFAULT_CMIP6_ARCHIVE_BASE = Path(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip6/data"
).as_posix()


def assert_vars_equal(var_id, *ds_list, extras=None):
    """Extract variable/DataArray `var_id` from each Dataset in the `ds_list`.
    Check they are all the same by comparing the arrays and common attributes.
    `extras` is an optional list of extra attributes to check.
    """
    if not extras:
        extras = []

    if len(ds_list) == 1:
        raise Exception("Only one Dataset passed to: _ds_var_check()")

    das = [ds[var_id] for ds in ds_list]
    ref_da = das[0]

    # Create a list of attributes to compare
    attrs = ["standard_name", "long_name", "units", "cell_methods"] + extras

    for da in das[1:]:
        assert da.values.tolist() == ref_da.values.tolist()

        for attr in attrs:
            if attr in ref_da.attrs:
                assert ref_da.attrs[attr] == da.attrs.get(attr)


# This is now only required for json files
XCLIM_TESTS_DATA = Path(TESTS_HOME, "xclim-testdata/testdata").as_posix()
MINI_ESGF_CACHE_DIR = Path.home() / ".mini-esgf-data"
MINI_ESGF_MASTER_DIR = os.path.join(MINI_ESGF_CACHE_DIR, "master")


def _check_output_nc(result, fname="output_001.nc", time=None):
    assert fname in [Path(_).name for _ in result]
    if time:
        ds = xr.open_mfdataset(result, use_cftime=True, decode_timedelta=False)
        time_ = f"{ds.time.values.min().isoformat()}/{ds.time.values.max().isoformat()}"
        assert time == time_


def write_roocs_cfg():
    cfg_templ = """
    [project:cmip5]
    base_dir = {{ base_dir }}/test_data/badc/cmip5/data/cmip5

    [project:cmip6]
    base_dir = {{ base_dir }}/test_data/badc/cmip6/data/CMIP6

    [project:cordex]
    base_dir = {{ base_dir }}/test_data/badc/cordex/data/cordex

    [project:c3s-cmip5]
    base_dir = {{ base_dir }}/test_data/gws/nopw/j04/cp4cds1_vol1/data/c3s-cmip5

    [project:c3s-cmip6]
    base_dir = {{ base_dir }}/test_data/badc/cmip6/data/CMIP6

    [project:c3s-cordex]
    base_dir = {{ base_dir }}/test_data/pool/data/CORDEX/data/cordex
    """
    cfg = Template(cfg_templ).render(base_dir=MINI_ESGF_MASTER_DIR)
    with open(ROOCS_CFG, "w") as fp:
        fp.write(cfg)

    # point to roocs cfg in environment
    os.environ["ROOCS_CONFIG"] = ROOCS_CFG


def cmip5_archive_base():
    if "CMIP5_ARCHIVE_BASE" in os.environ:
        return Path(os.environ["CMIP5_ARCHIVE_BASE"]).as_posix()
    return DEFAULT_CMIP5_ARCHIVE_BASE


def cmip6_archive_base():
    if "CMIP6_ARCHIVE_BASE" in os.environ:
        return Path(os.environ["CMIP6_ARCHIVE_BASE"]).as_posix()
    return DEFAULT_CMIP6_ARCHIVE_BASE


CMIP5_ARCHIVE_BASE = cmip5_archive_base()

CMIP5_ZOSTOGA = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip5/data/cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/zostoga_Omon_inmcm4_rcp45_r1i1p1_200601-210012.nc",
).as_posix()

CMIP5_TAS = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
).as_posix()

CMIP5_RH = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc",
).as_posix()

CMIP6_ARCHIVE_BASE = cmip6_archive_base()

CMIP6_RLDS = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc",
).as_posix()

CMIP6_RLDS_ONE_TIME_STEP = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001.nc",
).as_posix()

CMIP6_MRSOFC = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp119/r1i1p1f1/fx/mrsofc/gr/v20190410"
    "/mrsofc_fx_IPSL-CM6A-LR_ssp119_r1i1p1f1_gr.nc",
).as_posix()

CMIP6_SICONC = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc",
).as_posix()

CMIP6_SICONC_DAY = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/SIday/siconc/gn/v20190429/siconc_SIday_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc",
).as_posix()

CMIP6_TA = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp119/r1i1p1f1/Amon/ta/gn/files/d20190807/ta_Amon_MIROC6_ssp119_r1i1p1f1_gn_201501-202412.nc",
).as_posix()

C3S_CORDEX_AFR_TAS = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/pool/data/CORDEX/data/cordex/output/AFR-22/GERICS/MPI-M-MPI-ESM-LR/historical/r1i1p1/GERICS-REMO2015/v1/day/tas/v20201015/*.nc",
).as_posix()

C3S_CORDEX_NAM_PR = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/pool/data/CORDEX/data/cordex/output/NAM-22/OURANOS/NOAA-GFDL-GFDL-ESM2M/rcp45/r1i1p1/OURANOS-CRCM5/v1/day/pr/v20200831/*.nc",
).as_posix()

C3S_CORDEX_EUR_ZG500 = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/pool/data/CORDEX/data/cordex/output/EUR-11/IPSL/IPSL-IPSL-CM5A-MR/rcp85/r1i1p1/IPSL-WRF381P/v1/day/zg500/v20190919/*.nc",
).as_posix()

C3S_CORDEX_ANT_SFC_WIND = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/pool/data/CORDEX/data/cordex/output/ANT-44/KNMI/ECMWF-ERAINT/evaluation/r1i1p1/KNMI-RACMO21P/v1/day/sfcWind/v20201001/*.nc",
).as_posix()

C3S_CMIP5_TSICE = Path(
    REAL_C3S_CMIP5_ARCHIVE_BASE,
    "c3s-cmip5/output1/NCC/NorESM1-ME/rcp60/mon/seaIce/OImon/r1i1p1/tsice/v20120614/*.nc",
).as_posix()

C3S_CMIP5_TOS = Path(
    REAL_C3S_CMIP5_ARCHIVE_BASE,
    "c3s-cmip5/output1/BCC/bcc-csm1-1-m/historical/mon/ocean/Omon/r1i1p1/tos/v20120709/*.nc",
).as_posix()


CMIP6_TOS = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tos/gn/v20190710/tos_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001-186912.nc",
).as_posix()


CMIP6_TOS_ONE_TIME_STEP = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/tos/gn/v20190710/tos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_185001.nc",
).as_posix()
