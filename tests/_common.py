import os
import tempfile
from pathlib import Path

import pytest
from jinja2 import Template

from clisops.utils import get_file

ROOCS_CFG = os.path.join(tempfile.gettempdir(), "roocs.ini")
TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CMIP5_ARCHIVE_BASE = os.path.join(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip5/data"
)
REAL_C3S_CMIP5_ARCHIVE_BASE = "/gws/nopw/j04/cp4cds1_vol1/data/"
DEFAULT_CMIP6_ARCHIVE_BASE = os.path.join(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip6/data"
)

# This is now only required for json files
XCLIM_TESTS_DATA = os.path.join(TESTS_HOME, "xclim-testdata/testdata")
MINI_ESGF_CACHE_DIR = Path.home() / ".mini-esgf-data"
ESGF_TEST_DATA_REPO_URL = 'https://github.com/roocs/mini-esgf-data'


def write_roocs_cfg():
    cfg_templ = """
    [project:cmip5]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/badc/cmip5/data/cmip5

    [project:cmip6]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/badc/cmip6/data/CMIP6

    [project:cordex]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/badc/cordex/data/cordex

    [project:c3s-cmip5]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/gws/nopw/j04/cp4cds1_vol1/data/c3s-cmip5

    [project:c3s-cmip6]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/badc/cmip6/data/CMIP6

    [project:c3s-cordex]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/gws/nopw/j04/cp4cds1_vol1/data/c3s-cordex
    """
    cfg = Template(cfg_templ).render(base_dir=TESTS_HOME)
    with open(ROOCS_CFG, "w") as fp:
        fp.write(cfg)
    # point to roocs cfg in environment
    os.environ["ROOCS_CONFIG"] = ROOCS_CFG


def cmip5_archive_base():
    if "CMIP5_ARCHIVE_BASE" in os.environ:
        return os.environ["CMIP5_ARCHIVE_BASE"]
    return DEFAULT_CMIP5_ARCHIVE_BASE


def cmip6_archive_base():
    if "CMIP6_ARCHIVE_BASE" in os.environ:
        return os.environ["CMIP6_ARCHIVE_BASE"]
    return DEFAULT_CMIP6_ARCHIVE_BASE


CMIP5_ARCHIVE_BASE = cmip5_archive_base()

CMIP5_ZOSTOGA = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/badc/cmip5/data/cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/zostoga_Omon_inmcm4_rcp45_r1i1p1_200601-210012.nc'
)

CMIP5_TAS = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc'
)

CMIP5_RH = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc'
)


CMIP6_ARCHIVE_BASE = cmip6_archive_base()


CMIP6_RLDS = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc'
)

CMIP6_SICONC = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc'
)

C3S_CORDEX_PSL = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cordex/output/EUR-11/IPSL/MOHC-HadGEM2-ES/rcp85/r1i1p1/IPSL-WRF381P/v1/day/psl/v20190212/*.nc'
)

C3S_CMIP5_TSICE = os.path.join(
    REAL_C3S_CMIP5_ARCHIVE_BASE,
    "c3s-cmip5/output1/NCC/NorESM1-ME/rcp60/mon/seaIce/OImon/r1i1p1/tsice/v20120614/*.nc",
)

C3S_CMIP5_TOS = os.path.join(
    REAL_C3S_CMIP5_ARCHIVE_BASE,
    "c3s-cmip5/output1/BCC/bcc-csm1-1-m/historical/mon/ocean/Omon/r1i1p1/tos/v20120709/*.nc",
)
