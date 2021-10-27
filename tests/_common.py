import os
import tempfile
from pathlib import Path

import pytest
from jinja2 import Template

from clisops.utils import get_file

ROOCS_CFG = Path(tempfile.gettempdir(), "roocs.ini").as_posix()
TESTS_HOME = Path(__file__).parent.absolute().as_posix()
DEFAULT_CMIP5_ARCHIVE_BASE = Path(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip5/data"
).as_posix()
REAL_C3S_CMIP5_ARCHIVE_BASE = "/gws/nopw/j04/cp4cds1_vol1/data/"
DEFAULT_CMIP6_ARCHIVE_BASE = Path(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip6/data"
).as_posix()

# This is now only required for json files
XCLIM_TESTS_DATA = Path(TESTS_HOME, "xclim-testdata/testdata").as_posix()
MINI_ESGF_CACHE_DIR = Path.home() / ".mini-esgf-data"
MINI_ESGF_MASTER_DIR = os.path.join(MINI_ESGF_CACHE_DIR, "master")


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
    base_dir = {{ base_dir }}/test_data/gws/nopw/j04/cp4cds1_vol1/data/c3s-cordex
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

C3S_CORDEX_PSL = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cordex/output/EUR-11/IPSL/MOHC-HadGEM2-ES/rcp85/r1i1p1/IPSL-WRF381P/v1/day/psl/v20190212/*.nc",
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


# test daatsets used for regridding tests - one time step, full lat/lon
# cmip6 atmosphere grid
# x-res = 2.0
CMIP6_TAS_ONE_TIME_STEP = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/CAS/FGOALS-g3/historical/r1i1p1f1/Amon/tas/gn/v20190818/tas_Amon_FGOALS-g3_historical_r1i1p1f1_gn_185001.nc",
).as_posix()

# cmip6 ocean grids
# x-res = 0.90
CMIP6_TOS_ONE_TIME_STEP = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/tos/gn/v20190710/tos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_185001.nc",
).as_posix()

# cmip5 regular gridded one time step
CMIP5_MRSOS_ONE_TIME_STEP = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/day/land/day/r1i1p1/latest/mrsos/mrsos_day_HadGEM2-ES_rcp85_r1i1p1_20051201.nc",
).as_posix()

CMIP6_TAS_PRECISION_A = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/1pctCO2/r1i1p1f1/Amon/tas/gn/v20200212/tas_Amon_AWI-ESM-1-1-LR_1pctCO2_r1i1p1f1_gn_185501.nc",
).as_posix()

CMIP6_TAS_PRECISION_B = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/1pctCO2/r1i1p1f1/Amon/tas/gn/v20200212/tas_Amon_AWI-ESM-1-1-LR_1pctCO2_r1i1p1f1_gn_209901.nc",
).as_posix()

CMIP6_ATM_VERT_ONE_TIMESTEP = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/AERmon/o3/gn/v20190710/o3_AERmon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
).as_posix()

CMIP6_OCE_HALO_CNRM = Path(
    MINI_ESGF_CACHE_DIR,
    "master/test_data/badc/cmip6/data/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1-HR/historical/r1i1p1f2/Omon/tos/gn/v20191021/tos_Omon_CNRM-CM6-1-HR_historical_r1i1p1f2_gn_185001.nc",
).as_posix()

CMIP6_UNSTR_FESOM_LR = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/Omon/tos/gn/v20200212/tos_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001.nc",
).as_posix()

CMIP6_UNSTR_ICON_A = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/ICON-ESM-LR/historical/r1i1p1f1/Amon/tas/gn/v20210215/tas_Amon_ICON-ESM-LR_historical_r1i1p1f1_gn_185001.nc",
).as_posix()

CMIP6_UNSTR_VERT_ICON_O = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/ICON-ESM-LR/historical/r1i1p1f1/Omon/thetao/gn/v20210215/thetao_Omon_ICON-ESM-LR_historical_r1i1p1f1_gn_185001.nc",
).as_posix()

CMIP6_UNTAGGED_MISSVALS = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2-FV2/historical/r1i1p1f1/Omon/tos/gn/v20191120/tos_Omon_CESM2-FV2_historical_r1i1p1f1_gn_200001.nc",
).as_posix()

CMIP6_STAGGERED_UCOMP = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tauvo/gn/v20200909/tauuo_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
).as_posix()

CMIP6_STAGGERED_VCOMP = Path(
     MINI_ESGF_CACHE_DIR,
     "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tauvo/gn/v20190710/tauvo_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc ",
).as_posix()







