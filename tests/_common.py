import os

TESTS_HOME = os.path.abspath(os.path.dirname(__file__))
XCLIM_TESTS_DATA = os.path.join(TESTS_HOME, "xclim-testdata/testdata")
DEFAULT_CMIP5_ARCHIVE_BASE = os.path.join(
    TESTS_HOME, "mini-esgf-data/test_data/badc/cmip5/data"
)


def cmip5_archive_base():
    if "CMIP5_ARCHIVE_BASE" in os.environ:
        return os.environ["CMIP5_ARCHIVE_BASE"]
    return DEFAULT_CMIP5_ARCHIVE_BASE


CMIP5_ARCHIVE_BASE = cmip5_archive_base()


CMIP5_ZOSTOGA = os.path.join(
    CMIP5_ARCHIVE_BASE,
    "cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/*.nc",
)

CMIP5_TAS = os.path.join(
    CMIP5_ARCHIVE_BASE,
    "cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
)

CMIP5_RH = os.path.join(
    CMIP5_ARCHIVE_BASE,
    "cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc",
)

CMIP5_TAS_FILE = os.path.join(
    CMIP5_ARCHIVE_BASE,
    "cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc",  # noqa
)
