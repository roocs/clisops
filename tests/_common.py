import os
import tempfile
from pathlib import Path
import xarray as xr

from jinja2 import Template

from clisops.utils import get_file

ROOCS_CFG = os.path.join(tempfile.gettempdir(), "roocs.ini")
TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CMIP5_ARCHIVE_BASE = os.path.join(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip5/data"
)
REAL_C3S_CMIP5_ARCHIVE_BASE = "/group_workspaces/jasmin2/cp4cds1/vol1/data/"
DEFAULT_CMIP6_ARCHIVE_BASE = os.path.join(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip6/data"
)

# This is now only required for json files
XCLIM_TESTS_DATA = os.path.join(TESTS_HOME, "xclim-testdata/testdata")
MINI_ESGF_CACHE_DIR = Path.home() / ".mini-esgf-data"


def write_roocs_cfg():
    cfg_templ = """
    [project:cmip5]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/badc/cmip5/data

    [project:cmip6]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/badc/cmip6/data

    [project:cordex]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/badc/cordex/data

    [project:c3s-cmip5]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/group_workspaces/jasmin2/cp4cds1/vol1/data

    [project:c3s-cmip6]
    base_dir = NOT DEFINED YET

    [project:c3s-cordex]
    base_dir = {{ base_dir }}/mini-esgf-data/test_data/group_workspaces/jasmin2/cp4cds1/vol1/data
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

MINI_ESGF_KWARGS = dict(
    github_url = 'https://github.com/roocs/mini-esgf-data',
    branch = 'master',
    cache_dir = MINI_ESGF_CACHE_DIR
)

#CMIP5_ZOSTOGA = os.path.join(
#    CMIP5_ARCHIVE_BASE,
#    "cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/*.nc",
#)

CMIP5_ZOSTOGA = str(
    get_file(
        'test_data/badc/cmip5/data/cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/zostoga_Omon_inmcm4_rcp45_r1i1p1_200601-210012.nc',
        **MINI_ESGF_KWARGS
    )
)

#CMIP5_TAS = os.path.join(
#    CMIP5_ARCHIVE_BASE,
#    "cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
#)

CMIP5_TAS_FILES = [
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_209912-212411.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_219912-222411.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_229912-229912.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_203012-205511.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_212412-214911.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_222412-224911.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_205512-208011.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_214912-217411.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_224912-227411.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_208012-209912.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_217412-219911.nc',
    'tas_Amon_HadGEM2-ES_rcp85_r1i1p1_227412-229911.nc'
]

_ = get_file(
    [os.path.join('test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas', nc_path)
        for nc_path in CMIP5_TAS_FILES
    ],
    **MINI_ESGF_KWARGS
)

CMIP5_TAS = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc'
)

#CMIP5_RH = os.path.join(
#    CMIP5_ARCHIVE_BASE,
#    "cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc",
#)

CMIP5_RH_FILES = [
    'rh_Lmon_HadGEM2-ES_historical_r1i1p1_185912-188411.nc',
    'rh_Lmon_HadGEM2-ES_historical_r1i1p1_190912-193411.nc',
    'rh_Lmon_HadGEM2-ES_historical_r1i1p1_195912-198411.nc',
    'rh_Lmon_HadGEM2-ES_historical_r1i1p1_188412-190911.nc',
    'rh_Lmon_HadGEM2-ES_historical_r1i1p1_193412-195911.nc',
    'rh_Lmon_HadGEM2-ES_historical_r1i1p1_198412-200511.nc'
]

_ = get_file(
    [os.path.join('test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh', nc_path)
        for nc_path in CMIP5_RH_FILES
    ],
    **MINI_ESGF_KWARGS
)

CMIP5_RH = os.path.join(
    MINI_ESGF_CACHE_DIR,
    'master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc'
)


CMIP5_TAS_FILE = str(
    get_file(
        "cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc",
        branch="add_cmip5_hadgem",  # This will be removed once the branch is merged into "main"
    )
)

CMIP6_ARCHIVE_BASE = cmip6_archive_base()

CMIP6_O3 = str(
    get_file(
        "cmip6/o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc",
    )
)

#CMIP6_RLDS = os.path.join(
#    CMIP6_ARCHIVE_BASE,
#    "CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803",
#    "rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc",
#)

CMIP6_RLDS = str(
    get_file(
        'test_data/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc',
        **MINI_ESGF_KWARGS
    )
)

C3S_CMIP5_TSICE = os.path.join(
    REAL_C3S_CMIP5_ARCHIVE_BASE,
    "c3s-cmip5/output1/NCC/NorESM1-ME/rcp60/mon/seaIce/OImon/r1i1p1/tsice/v20120614/*.nc",
)

C3S_CMIP5_TOS = os.path.join(
    REAL_C3S_CMIP5_ARCHIVE_BASE,
    "c3s-cmip5/output1/BCC/bcc-csm1-1-m/historical/mon/ocean/Omon/r1i1p1/tos/v20120709/*.nc",
)
