import importlib.resources as ilr
import os
import warnings
from pathlib import Path
from shutil import copytree
from sys import platform
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlretrieve

from filelock import FileLock
from jinja2 import Template
from loguru import logger

try:
    import pooch
except ImportError:
    warnings.warn(
        "The `pooch` library is not installed. "
        "The default cache directory for testing data will not be set."
    )
    pooch = None

__all__ = [
    "ContextLogger",
    "ESGF_TEST_DATA_CACHE_DIR",
    "ESGF_TEST_DATA_REPO_URL",
    "ESGF_TEST_DATA_VERSION",
    "XCLIM_TEST_DATA_CACHE_DIR",
    "XCLIM_TEST_DATA_REPO_URL",
    "XCLIM_TEST_DATA_VERSION",
    "default_esgf_test_data_cache",
    "default_xclim_test_data_cache",
    "get_esgf_file_paths",
    "get_esgf_glob_paths",
    "load_registry",
    "stratus",
    "write_roocs_cfg",
]

try:
    default_esgf_test_data_cache = pooch.os_cache("mini-esgf-data")
    default_xclim_test_data_cache = pooch.os_cache("xclim-testdata")
except AttributeError:
    default_esgf_test_data_cache = None
    default_xclim_test_data_cache = None

ESGF_TEST_DATA_REPO_URL = os.getenv(
    "ESGF_TEST_DATA_REPO_UR", "https://raw.githubusercontent.com/roocs/mini-esgf-data"
)
default_esgf_test_data_version = "v1"
ESGF_TEST_DATA_VERSION = os.getenv(
    "ESGF_TEST_DATA_VERSION", default_esgf_test_data_version
)
ESGF_TEST_DATA_CACHE_DIR = os.getenv(
    "ESGF_TEST_DATA_CACHE_DIR", default_esgf_test_data_cache
)

XCLIM_TEST_DATA_REPO_URL = os.getenv(
    "XCLIM_TEST_DATA_REPO_URL",
    "https://raw.githubusercontent.com/Ouranosinc/xclim-testdata",
)
default_xclim_test_data_version = "v2024.8.23"
XCLIM_TEST_DATA_VERSION = os.getenv(
    "XCLIM_TEST_DATA_VERSION", default_xclim_test_data_version
)
XCLIM_TEST_DATA_CACHE_DIR = os.getenv(
    "XCLIM_TEST_DATA_CACHE_DIR", default_xclim_test_data_cache
)


def write_roocs_cfg(
    template: Optional[str] = None,
    cache_dir: Union[str, Path] = default_esgf_test_data_cache,
) -> str:
    default_template = """
    [project:cmip5]
    base_dir = {{ base_dir }}/badc/cmip5/data/cmip5

    [project:cmip6]
    base_dir = {{ base_dir }}/badc/cmip6/data/CMIP6

    [project:cordex]
    base_dir = {{ base_dir }}/badc/cordex/data/cordex

    [project:c3s-cmip5]
    base_dir = {{ base_dir }}/gws/nopw/j04/cp4cds1_vol1/data/c3s-cmip5

    [project:c3s-cmip6]
    base_dir = {{ base_dir }}/badc/cmip6/data/CMIP6

    [project:c3s-cordex]
    base_dir = {{ base_dir }}/pool/data/CORDEX/data/cordex

    [project:proj_test]
    base_dir = /projects/test/proj
    fixed_path_modifiers =
        variable:rain sun cloud
    fixed_path_mappings =
        proj_test.my.first.test:first/test/something.nc
        proj_test.my.second.test:second/test/data_*.txt
        proj_test.another.{variable}.test:good/test/{variable}.nc
    """

    cfg_template = template or default_template
    roocs_config = Path(cache_dir, "roocs.ini")
    cfg = Template(cfg_template).render(
        base_dir=Path(ESGF_TEST_DATA_CACHE_DIR)
        .joinpath(ESGF_TEST_DATA_VERSION)
        .as_posix()
    )
    with open(roocs_config, "w") as fp:
        fp.write(cfg)

    return roocs_config.as_posix()


def get_esgf_file_paths(esgf_cache_dir: Union[str, os.PathLike[str]]):
    return {
        "CMIP5_ZOSTOGA": Path(
            esgf_cache_dir,
            "badc/cmip5/data/cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/zostoga_Omon_inmcm4_rcp45_r1i1p1_200601-210012.nc",
        ).as_posix(),
        "CMIP6_RLDS": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc",
        ).as_posix(),
        "CMIP6_RLDS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001.nc",
        ).as_posix(),
        "CMIP6_RLUS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlus/gr/v20180803/rlus_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001.nc",
        ).as_posix(),
        "CMIP6_MRSOFC": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp119/r1i1p1f1/fx/mrsofc/gr/v20190410/mrsofc_fx_IPSL-CM6A-LR_ssp119_r1i1p1f1_gr.nc",
        ).as_posix(),
        "CMIP6_SICONC": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc",
        ).as_posix(),
        "CMIP6_SICONC_DAY": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/SIday/siconc/gn/v20190429/siconc_SIday_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc",
        ).as_posix(),
        "CMIP6_TA": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp119/r1i1p1f1/Amon/ta/gn/files/d20190807/ta_Amon_MIROC6_ssp119_r1i1p1f1_gn_201501-202412.nc",
        ).as_posix(),
        "CMIP6_TASMIN": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Amon/tasmin/gn/v20190710/tasmin_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_201001-201412.nc",
        ).as_posix(),
        "CMIP6_JULIAN": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/CCCR-IITM/IITM-ESM/1pctCO2/r1i1p1f1/Omon/tos/gn/v20191204/tos_Omon_IITM-ESM_1pctCO2_r1i1p1f1_gn_193001-193412.nc",
        ).as_posix(),
        "CMIP6_TOS": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tos/gn/v20190710/tos_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001-186912.nc",
        ).as_posix(),
        "CMIP6_AREACELLO": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Ofx/areacello/gn/v20190726/areacello_Ofx_GFDL-ESM4_historical_r1i1p1f1_gn.nc",
        ).as_posix(),
        "CMIP6_TOS_CNRM": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-CM6-1/ssp245/r1i1p1f2/Omon/tos/gn/v20190219/tos_Omon_CNRM-CM6-1_ssp245_r1i1p1f2_gn_201501.nc",
        ).as_posix(),
        "CMIP6_TAS_DAY": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp119/r1i1p1f1/day/tas/gn/v20191016/tas_day_MIROC6_ssp119_r1i1p1f1_gn_20150101.nc",
        ).as_posix(),
        "CMIP6_SFTOF": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/NCC/NorESM2-MM/ssp126/r1i1p1f1/Ofx/sftof/gn/v20191108/sftof_Ofx_NorESM2-MM_ssp126_r1i1p1f1_gn.nc",
        ).as_posix(),
        "CMIP6_TAS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/CAS/FGOALS-g3/historical/r1i1p1f1/Amon/tas/gn/v20190818/tas_Amon_FGOALS-g3_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_TOS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/tos/gn/v20190710/tos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        # CMIP6 ocean with collapsing cells
        "CMIP6_TOS_LR_DEGEN": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/HAMMOZ-Consortium/MPI-ESM-1-2-HAM/ssp370/r1i1p1f1/Omon/tos/gn/v20190628/tos_Omon_MPI-ESM-1-2-HAM_ssp370_r1i1p1f1_gn_201501.nc",
        ).as_posix(),
        # 2nd dataset CMIP6 ocean with collapsing cells
        "CMIP6_FX_DEGEN": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/EC-Earth-Consortium/EC-Earth3-Veg/ssp245/r5i1p1f1/Ofx/deptho/gn/v20200312/deptho_Ofx_EC-Earth3-Veg_ssp245_r5i1p1f1_gn.nc",
        ).as_posix(),
        # CMIP6 ocean with collapsing cells, cells extending over 50 degrees, missing_values in lat/lon
        "CMIP6_SIMASS_DEGEN": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/NCC/NorESM2-MM/ssp126/r1i1p1f1/SImon/simass/gn/v20191108/simass_SImon_NorESM2-MM_ssp126_r1i1p1f1_gn_201501.nc",
        ).as_posix(),
        # CMIP5 rlat,rlon uncompliant CF units
        "CMIP5_WRONG_CF_UNITS": Path(
            esgf_cache_dir,
            "pool/data/C3SCMIP5/BCC/bcc-csm1-1/rcp85/mon/ocean/Omon/r1i1p1/zos/v20120705/zos_Omon_bcc-csm1-1_rcp85_r1i1p1_200601.nc",
        ).as_posix(),
        # CMIP6 rlat,rlon uncompliant CF units
        "CMIP6_WRONG_CF_UNITS": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/AerChemMIP/BCC/BCC-ESM1/ssp370/r1i1p1f1/Omon/pbo/gn/v20190624/pbo_Omon_BCC-ESM1_ssp370_r1i1p1f1_gn_201501.nc",
        ).as_posix(),
        # CMIP6 lat, lon with uncompliant CF units and standard_name
        "CMIP6_WRONG_CF_ATTRS": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/HighResMIP/BCC/BCC-CSM2-HR/hist-1950/r1i1p1f1/Omon/tos/gn/v20200922/tos_Omon_BCC-CSM2-HR_hist-1950_r1i1p1f1_gn_198001.nc",
        ).as_posix(),
        "CMIP5_MRSOS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/day/land/day/r1i1p1/latest/mrsos/mrsos_day_HadGEM2-ES_rcp85_r1i1p1_20051201.nc",
        ).as_posix(),
        "CMIP6_GFDL_EXTENT": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/historical/r1i1p1f1/Omon/sos/gn/v20180701/sos_Omon_GFDL-CM4_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_TAS_PRECISION_A": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/1pctCO2/r1i1p1f1/Amon/tas/gn/v20200212/tas_Amon_AWI-ESM-1-1-LR_1pctCO2_r1i1p1f1_gn_185501.nc",
        ).as_posix(),
        "CMIP6_TAS_PRECISION_B": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/1pctCO2/r1i1p1f1/Amon/tas/gn/v20200212/tas_Amon_AWI-ESM-1-1-LR_1pctCO2_r1i1p1f1_gn_209901.nc",
        ).as_posix(),
        "CMIP6_ATM_VERT_ONE_TIMESTEP": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/AERmon/o3/gn/v20190710/o3_AERmon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_ATM_VERT_ONE_TIMESTEP_ZONMEAN": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/AERmon/o3/gn/v20190710/o3_AERmon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001_zm.nc",
        ).as_posix(),
        "CMIP6_IITM_EXTENT": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/CCCR-IITM/IITM-ESM/1pctCO2/r1i1p1f1/Omon/tos/gn/v20191204/tos_Omon_IITM-ESM_1pctCO2_r1i1p1f1_gn_193001.nc",
        ).as_posix(),
        # CMIP6 dataset with weird range in its longitude coordinate (-300, 60)
        #   and unmasked missing values in the latitude and longitude coordinates
        "CMIP6_EXTENT_UNMASKED": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/OMIP/NOAA-GFDL/GFDL-OM4p5B/omip1/r1i1p1f1/Omon/volcello/gn/v20180701/volcello_Omon_GFDL-OM4p5B_omip1_r1i1p1f1_gn_176801.nc",
        ).as_posix(),
        "CMIP6_OCE_HALO_CNRM": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1-HR/historical/r1i1p1f2/Omon/tos/gn/v20191021/tos_Omon_CNRM-CM6-1-HR_historical_r1i1p1f2_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNSTR_FESOM_LR": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/Omon/tos/gn/v20200212/tos_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNSTR_ICON_A": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/ICON-ESM-LR/historical/r1i1p1f1/Amon/tas/gn/v20210215/tas_Amon_ICON-ESM-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNSTR_VERT_ICON_O": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/ICON-ESM-LR/historical/r1i1p1f1/Omon/thetao/gn/v20210215/thetao_Omon_ICON-ESM-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNTAGGED_MISSVALS": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2-FV2/historical/r1i1p1f1/Omon/tos/gn/v20191120/tos_Omon_CESM2-FV2_historical_r1i1p1f1_gn_200001.nc",
        ).as_posix(),
        "CMIP6_STAGGERED_UCOMP": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tauuo/gn/v20200909/tauuo_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_STAGGERED_VCOMP": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tauvo/gn/v20190710/tauvo_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_FILLVALUE": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2-WACCM/historical/r1i1p1f1/day/tas/gn/v20190227/tas_day_CESM2-WACCM_historical_r1i1p1f1_gn_20000101-20091231.nc",
        ).as_posix(),
        "CMIP6_ZONMEAN_A": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/msftmz/gn/v20190710/msftmz_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_191001.nc",
        ).as_posix(),
        "CMIP6_ZONMEAN_B": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/CMIP/NCC/NorCPM1/historical/r22i1p1f1/Omon/msftmz/grz/v20200724/msftmz_Omon_NorCPM1_historical_r22i1p1f1_grz_185001.nc",
        ).as_posix(),
        # CMIP6 dataset without defined bounds on curvilinear grid
        "CMIP6_NO_BOUNDS": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/CAS/FGOALS-f3-L/ssp126/r1i1p1f1/Omon/tos/gn/v20191008/tos_Omon_FGOALS-f3-L_ssp126_r1i1p1f1_gn_201501.nc",
        ).as_posix(),
        # CMIP6 dataset with character dimension 'sector'
        "CMIP6_CHAR_DIM": Path(
            esgf_cache_dir,
            "badc/cmip6/data/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp245/r1i1p1f1/Lmon/landCoverFrac/gr/v20190119/landCoverFrac_Lmon_IPSL-CM6A-LR_ssp245_r1i1p1f1_gr_201501.nc",
        ).as_posix(),
        # CORDEX dataset with maldefined bounds
        "CORDEX_ERRONEOUS_BOUNDS": Path(
            esgf_cache_dir,
            "pool/data/C3SCORDEX/data/c3s-cordex/output/ARC-44/BCCR/ECMWF-ERAINT/evaluation/r1i1p1/BCCR-WRF331/v1/day/tas/v20200915/tas_ARC-44_ECMWF-ERAINT_evaluation_r1i1p1_BCCR-WRF331_v1_day_20010101.nc",
        ).as_posix(),
        "CORDEX_TAS_ONE_TIMESTEP": Path(
            esgf_cache_dir,
            "pool/data/CORDEX/data/cordex/output/EUR-22/GERICS/MPI-M-MPI-ESM-LR/rcp85/r1i1p1/GERICS-REMO2015/v1/mon/tas/v20191029/tas_EUR-22_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_GERICS-REMO2015_v1_mon_202101.nc",
        ).as_posix(),
        "CORDEX_TAS_ONE_TIMESTEP_ANT": Path(
            esgf_cache_dir,
            "pool/data/CORDEX/data/cordex/output/ANT-44/KNMI/ECMWF-ERAINT/evaluation/r1i1p1/DMI-HIRHAM5/v1/day/tas/v20201001/tas_ANT-44_ECMWF-ERAINT_evaluation_r1i1p1_DMI-HIRHAM5_v1_day_20060101.nc",
        ).as_posix(),
        "CORDEX_TAS_NO_BOUNDS": Path(
            esgf_cache_dir,
            "pool/data/CORDEX/data/cordex/output/EUR-11/KNMI/MPI-M-MPI-ESM-LR/rcp85/r1i1p1/KNMI-RACMO22E/v1/mon/tas/v20190625/tas_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_KNMI-RACMO22E_v1_mon_209101.nc",
        ).as_posix(),
        "ATLAS_v1_CMIP5": Path(
            esgf_cache_dir,
            "pool/data/c3s-cica-atlas/CMIP5/rcp26/pr_CMIP5_rcp26_mon_200601-210012.nc",
        ).as_posix(),
        "ATLAS_v1_EOBS": Path(
            esgf_cache_dir,
            "pool/data/c3s-cica-atlas/E-OBS/sfcwind_E-OBS_mon_195001-202112.nc",
        ).as_posix(),
        "ATLAS_v1_ERA5": Path(
            esgf_cache_dir,
            "pool/data/c3s-cica-atlas/ERA5/psl_ERA5_mon_194001-202212.nc",
        ).as_posix(),
        "ATLAS_v1_CORDEX": Path(
            esgf_cache_dir,
            "pool/data/c3s-cica-atlas/CORDEX-CORE/historical/huss_CORDEX-CORE_historical_mon_197001.nc",
        ).as_posix(),
        "ATLAS_v1_EOBS_GRID": Path(
            esgf_cache_dir,
            "pool/data/c3s-cica-atlas/E-OBS/t_E-OBS_mon_195001.nc",
        ).as_posix(),
        "ATLAS_v0_CORDEX_NAM": Path(
            esgf_cache_dir,
            "pool/data/c3s-ipcc-ar6-atlas/CORDEX-NAM/historical/rx1day_CORDEX-NAM_historical_mon_197001-200512.nc",
        ).as_posix(),
        "ATLAS_v0_CMIP6": Path(
            esgf_cache_dir,
            "pool/data/c3s-ipcc-ar6-atlas/CMIP6/ssp245/sst_CMIP6_ssp245_mon_201501-210012.nc",
        ).as_posix(),
        "ATLAS_v0_CORDEX_ANT": Path(
            esgf_cache_dir,
            "pool/data/c3s-ipcc-ar6-atlas/CORDEX-ANT/rcp45/tnn_CORDEX-ANT_rcp45_mon_200601.nc",
        ).as_posix(),
    }


def get_kerchunk_datasets():
    kerchunk = {
        # Kerchunk datasets
        "CMIP6_KERCHUNK_HTTPS_OPEN_JSON": (
            "https://gws-access.jasmin.ac.uk/public/cmip6_prep/eodh-eocis/kc-indexes-cmip6-http-v1/"
            "CMIP6.CMIP.MOHC.UKESM1-1-LL.1pctCO2.r1i1p1f2.Amon.tasmax.gn.v20220513.json"
        )
    }
    kerchunk["CMIP6_KERCHUNK_HTTPS_OPEN_ZST"] = (
        f"{kerchunk['CMIP6_KERCHUNK_HTTPS_OPEN_JSON']}.zst"
    )
    return kerchunk


def get_esgf_glob_paths(esgf_cache_dir: Union[str, os.PathLike[str]]):
    return {
        "CMIP5_TAS": Path(
            esgf_cache_dir,
            "badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
        ).as_posix(),
        "CMIP5_TAS_EC_EARTH": Path(
            esgf_cache_dir,
            "badc/cmip5/data/cmip5/output1/ICHEC/EC-EARTH/historical/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
        ).as_posix(),
        "CMIP5_RH": Path(
            esgf_cache_dir,
            "badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc",
        ).as_posix(),
        "C3S_CMIP5_TSICE": Path(
            esgf_cache_dir,
            "gws/nopw/j04/cp4cds1_vol1/data/c3s-cmip5/output1/NCC/NorESM1-ME/rcp60/mon/seaIce/OImon/r1i1p1/tsice/v20120614/*.nc",
        ).as_posix(),
        "C3S_CORDEX_AFR_TAS": Path(
            esgf_cache_dir,
            "pool/data/CORDEX/data/cordex/output/AFR-22/GERICS/MPI-M-MPI-ESM-LR/historical/r1i1p1/GERICS-REMO2015/v1/day/tas/v20201015/*.nc",
        ).as_posix(),
        "C3S_CORDEX_NAM_PR": Path(
            esgf_cache_dir,
            "pool/data/CORDEX/data/cordex/output/NAM-22/OURANOS/NOAA-GFDL-GFDL-ESM2M/rcp45/r1i1p1/OURANOS-CRCM5/v1/day/pr/v20200831/*.nc",
        ).as_posix(),
        "C3S_CORDEX_EUR_ZG500": Path(
            esgf_cache_dir,
            "pool/data/CORDEX/data/cordex/output/EUR-11/IPSL/IPSL-IPSL-CM5A-MR/rcp85/r1i1p1/IPSL-WRF381P/v1/day/zg500/v20190919/*.nc",
        ).as_posix(),
        "C3S_CORDEX_ANT_SFC_WIND": Path(
            esgf_cache_dir,
            "pool/data/CORDEX/data/cordex/output/ANT-44/KNMI/ECMWF-ERAINT/evaluation/r1i1p1/KNMI-RACMO21P/v1/day/sfcWind/v20201001/*.nc",
        ).as_posix(),
        "CMIP5_MRSOS_MULTIPLE_TIME_STEPS": Path(
            esgf_cache_dir,
            "badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp45/day/land/day/r1i1p1/latest/mrsos/*.nc",
        ).as_posix(),
        "C3S_CMIP5_TAS": Path(
            esgf_cache_dir,
            "gws/nopw/j04/cp4cds1_vol1/data/c3s-cmip5/output1/ICHEC/EC-EARTH/historical/day/atmos/day/r1i1p1/tas/v20131231/*.nc",
        ).as_posix(),
    }


class ContextLogger:
    """Helper function for safe logging management in pytests"""

    def __init__(self, caplog=False):
        from loguru import logger

        self.logger = logger
        self.using_caplog = False
        if caplog:
            self.using_caplog = True

    def __enter__(self, package_name: str = "clisops"):
        self.logger.enable(package_name)
        self._package = package_name
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """If test is supplying caplog, pytest will manage teardown."""

        self.logger.disable(self._package)
        if not self.using_caplog:
            try:
                self.logger.remove()
            except ValueError:
                pass


def load_registry(branch: str, repo: str):
    """Load the registry file for the test data.

    Returns
    -------
    dict
        Dictionary of filenames and hashes.
    """
    if repo == ESGF_TEST_DATA_REPO_URL:
        project = "mini-esgf-data"
        default_testdata_version = ESGF_TEST_DATA_VERSION
        default_testdata_repo_url = ESGF_TEST_DATA_REPO_URL
    elif repo == XCLIM_TEST_DATA_REPO_URL:
        project = "xclim-testdata"
        default_testdata_version = XCLIM_TEST_DATA_VERSION
        default_testdata_repo_url = XCLIM_TEST_DATA_REPO_URL
    else:
        raise ValueError(
            f"Repository URL {repo} not recognized. "
            f"Please use one of {ESGF_TEST_DATA_REPO_URL} or {XCLIM_TEST_DATA_REPO_URL}"
        )

    remote_registry = audit_url(f"{repo}{branch}/data/{project}_registry.txt")
    if branch != default_testdata_version:
        custom_registry_folder = Path(
            str(ilr.files("clisops").joinpath(f"utils/registries/{branch}"))
        )
        custom_registry_folder.mkdir(parents=True, exist_ok=True)
        registry_file = custom_registry_folder.joinpath(f"{project}_registry.txt")
        urlretrieve(remote_registry, registry_file)  # noqa: S310
    elif repo != default_testdata_repo_url:
        registry_file = Path(
            str(ilr.files("clisops").joinpath(f"utils/{project}_registry.txt"))
        )
        urlretrieve(remote_registry, registry_file)  # noqa: S310

    registry_file = Path(
        str(ilr.files("clisops").joinpath(f"utils/{project}_registry.txt"))
    )
    if not registry_file.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_file}")

    # Load the registry file
    with registry_file.open() as f:
        registry = {line.split()[0]: line.split()[1] for line in f}
    return registry


def stratus(  # noqa: PR01
    repo: str,
    branch: str,
    cache_dir: Union[str, Path],
    data_updates: bool = True,
):
    """Pooch registry instance for xclim test data.

    Parameters
    ----------
    repo : str
        URL of the repository to use when fetching testing datasets.
    branch : str
        Branch of repository to use when fetching testing datasets.
    cache_dir : str or Path
        The path to the directory where the data files are stored.
    data_updates : bool
        If True, allow updates to the data files. Default is True.

    Returns
    -------
    pooch.Pooch
        The Pooch instance for accessing the testing data.

    Examples
    --------
    Using the registry to download a file:

    .. code-block:: python

        import xarray as xr
        from clisops.utils.testing import stratus

        s = stratus(data_dir=..., repo=..., branch=...)
        example_file = s.fetch("example.nc")
        data = xr.open_dataset(example_file)
    """
    if pooch is None:
        raise ImportError(
            "The `pooch` package is required to fetch the remote testing data. "
            "You can install it with `pip install pooch` or `pip install roocs-utils[dev]`."
        )

    if repo.endswith("xclim-testdata"):
        _version = XCLIM_TEST_DATA_VERSION
        _default_version = default_xclim_test_data_version
    elif repo.endswith("mini-esgf-data"):
        _version = ESGF_TEST_DATA_VERSION
        _default_version = default_esgf_test_data_version
    else:
        raise ValueError(
            f"Repository URL {repo} not recognized. "
            f"Please use one of {ESGF_TEST_DATA_REPO_URL} or {XCLIM_TEST_DATA_REPO_URL}"
        )

    remote = audit_url(f"{repo}/{branch}/data")
    return pooch.create(
        path=cache_dir,
        base_url=remote,
        version=_default_version,
        version_dev=_version,
        allow_updates=data_updates,
        registry=load_registry(branch=branch, repo=repo),
    )


def populate_testing_data(
    repo: str,
    branch: str,
    cache_dir: Path,
) -> None:
    """Populate the local cache with the testing data.

    Parameters
    ----------
    repo : str, optional
        URL of the repository to use when fetching testing datasets.
    branch : str, optional
        Branch of xclim-testdata to use when fetching testing datasets.
    cache_dir : Path
        The path to the local cache. Defaults to the location set by the platformdirs library.
        The testing data will be downloaded to this local cache.

    Returns
    -------
    None
    """
    # Create the Pooch instance
    n = stratus(cache_dir=cache_dir, repo=repo, branch=branch)

    # Download the files
    errored_files = []
    for file in load_registry(branch=branch, repo=repo):
        try:
            n.fetch(file)
        except HTTPError:
            msg = f"File `{file}` not accessible in remote repository."
            logger.error(msg)
            errored_files.append(file)
        else:
            logger.info("Files were downloaded successfully.")

    if errored_files:
        logger.error(
            "The following files were unable to be downloaded: %s",
            errored_files,
        )


def gather_testing_data(
    worker_cache_dir: Union[str, os.PathLike[str], Path],
    worker_id: str,
    branch: str,
    repo: str,
    cache_dir: Union[str, os.PathLike[str], Path],
):
    """Gather testing data across workers."""
    cache_dir = Path(cache_dir)
    if repo.endswith("xclim-testdata"):
        version = default_xclim_test_data_version
    elif repo.endswith("mini-esgf-data"):
        version = default_esgf_test_data_version
    else:
        raise ValueError(
            f"Repository URL {repo} not recognized. "
            f"Please use one of {ESGF_TEST_DATA_REPO_URL} or {XCLIM_TEST_DATA_REPO_URL}"
        )

    if worker_id == "master":
        populate_testing_data(branch=branch, repo=repo, cache_dir=cache_dir)
    else:
        if platform == "win32":
            if not cache_dir.joinpath(branch).exists():
                raise FileNotFoundError(
                    "Testing data not found and UNIX-style file-locking is not supported on Windows. "
                    "Consider running `populate_testing_data()` to download testing data beforehand."
                )
        else:
            cache_dir.mkdir(exist_ok=True, parents=True)
            lockfile = cache_dir.joinpath(".lock")
            test_data_being_written = FileLock(lockfile)
            with test_data_being_written:
                # This flag prevents multiple calls from re-attempting to download testing data in the same pytest run
                populate_testing_data(branch=branch, repo=repo, cache_dir=cache_dir)
                cache_dir.joinpath(".data_written").touch()
            with test_data_being_written.acquire():
                if lockfile.exists():
                    lockfile.unlink()
        copytree(cache_dir.joinpath(version), worker_cache_dir)


def audit_url(url: str, context: Optional[str] = None) -> str:
    """Check if the URL is well-formed.

    Raises
    ------
    URLError
        If the URL is not well-formed.
    """
    msg = ""
    result = urlparse(url)
    if result.scheme == "http":
        msg = f"{context if context else ''} URL is not using secure HTTP: '{url}'".strip()
    if not all([result.scheme, result.netloc]):
        msg = f"{context if context else ''} URL is not well-formed: '{url}'".strip()

    if msg:
        logger.error(msg)
        raise URLError(msg)
    return url
