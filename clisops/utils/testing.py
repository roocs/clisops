import os
from pathlib import Path
from typing import Union

from jinja2 import Template
from platformdirs import user_cache_dir

__all__ = [
    "MINI_ESGF_CACHE_DIR",
    "ContextLogger",
    "write_roocs_cfg",
    "get_esgf_file_paths",
]

MINI_ESGF_CACHE_DIR = user_cache_dir(".mini-esgf-data")


def write_roocs_cfg(cache_dir: Union[str, Path]):

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
    roocs_config = Path(cache_dir, "roocs.ini").as_posix()
    cfg = Template(cfg_templ).render(
        base_dir=Path(MINI_ESGF_CACHE_DIR).joinpath("master")
    )
    with open(roocs_config, "w") as fp:
        fp.write(cfg)

    # point to roocs cfg in environment
    os.environ["ROOCS_CONFIG"] = roocs_config


def get_esgf_file_paths(esgf_cache_dir) -> dict[str, str]:
    return {
        "CMIP5_ZOSTOGA": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip5/data/cmip5/output1/INM/inmcm4/rcp45/mon/ocean/Omon/r1i1p1/latest/zostoga/zostoga_Omon_inmcm4_rcp45_r1i1p1_200601-210012.nc",
        ).as_posix(),
        "CMIP5_TAS": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
        ).as_posix(),
        "CMIP5_RH": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/historical/mon/land/Lmon/r1i1p1/latest/rh/*.nc",
        ).as_posix(),
        "CMIP6_RLDS": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc",
        ).as_posix(),
        "CMIP6_RLDS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001.nc",
        ).as_posix(),
        "CMIP6_MRSOFC": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp119/r1i1p1f1/fx/mrsofc/gr/v20190410/mrsofc_fx_IPSL-CM6A-LR_ssp119_r1i1p1f1_gr.nc",
        ).as_posix(),
        "CMIP6_SICONC": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/SImon/siconc/gn/latest/siconc_SImon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc",
        ).as_posix(),
        "CMIP6_SICONC_DAY": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p1f1/SIday/siconc/gn/v20190429/siconc_SIday_CanESM5_historical_r1i1p1f1_gn_18500101-20141231.nc",
        ).as_posix(),
        "CMIP6_TA": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp119/r1i1p1f1/Amon/ta/gn/files/d20190807/ta_Amon_MIROC6_ssp119_r1i1p1f1_gn_201501-202412.nc",
        ).as_posix(),
        "CMIP6_TASMIN": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Amon/tasmin/gn/v20190710/tasmin_Amon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_201001-201412.nc",
        ).as_posix(),
        "CMIP6_JULIAN": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/CCCR-IITM/IITM-ESM/1pctCO2/r1i1p1f1/Omon/tos/gn/v20191204/tos_Omon_IITM-ESM_1pctCO2_r1i1p1f1_gn_193001-193412.nc",
        ).as_posix(),
        "C3S_CORDEX_AFR_TAS": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CORDEX/data/cordex/output/AFR-22/GERICS/MPI-M-MPI-ESM-LR/historical/r1i1p1/GERICS-REMO2015/v1/day/tas/v20201015/*.nc",
        ).as_posix(),
        "C3S_CORDEX_NAM_PR": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CORDEX/data/cordex/output/NAM-22/OURANOS/NOAA-GFDL-GFDL-ESM2M/rcp45/r1i1p1/OURANOS-CRCM5/v1/day/pr/v20200831/*.nc",
        ).as_posix(),
        "C3S_CORDEX_EUR_ZG500": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CORDEX/data/cordex/output/EUR-11/IPSL/IPSL-IPSL-CM5A-MR/rcp85/r1i1p1/IPSL-WRF381P/v1/day/zg500/v20190919/*.nc",
        ).as_posix(),
        "C3S_CORDEX_ANT_SFC_WIND": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CORDEX/data/cordex/output/ANT-44/KNMI/ECMWF-ERAINT/evaluation/r1i1p1/KNMI-RACMO21P/v1/day/sfcWind/v20201001/*.nc",
        ).as_posix(),
        "CMIP6_TOS": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tos/gn/v20190710/tos_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001-186912.nc",
        ).as_posix(),
        "CMIP6_TAS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/CAS/FGOALS-g3/historical/r1i1p1f1/Amon/tas/gn/v20190818/tas_Amon_FGOALS-g3_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_TOS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/tos/gn/v20190710/tos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP5_MRSOS_ONE_TIME_STEP": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/day/land/day/r1i1p1/latest/mrsos/mrsos_day_HadGEM2-ES_rcp85_r1i1p1_20051201.nc",
        ).as_posix(),
        "CMIP5_MRSOS_MULTIPLE_TIME_STEPS": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp45/day/land/day/r1i1p1/latest/mrsos/*.nc",
        ).as_posix(),
        "CMIP6_GFDL_EXTENT": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-CM4/historical/r1i1p1f1/Omon/sos/gn/v20180701/sos_Omon_GFDL-CM4_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_TAS_PRECISION_A": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/1pctCO2/r1i1p1f1/Amon/tas/gn/v20200212/tas_Amon_AWI-ESM-1-1-LR_1pctCO2_r1i1p1f1_gn_185501.nc",
        ).as_posix(),
        "CMIP6_TAS_PRECISION_B": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/1pctCO2/r1i1p1f1/Amon/tas/gn/v20200212/tas_Amon_AWI-ESM-1-1-LR_1pctCO2_r1i1p1f1_gn_209901.nc",
        ).as_posix(),
        "CMIP6_ATM_VERT_ONE_TIMESTEP": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/AERmon/o3/gn/v20190710/o3_AERmon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_ATM_VERT_ONE_TIMESTEP_ZONMEAN": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/AERmon/o3/gn/v20190710/o3_AERmon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001_zm.nc",
        ).as_posix(),
        "CMIP6_IITM_EXTENT": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/CCCR-IITM/IITM-ESM/1pctCO2/r1i1p1f1/Omon/tos/gn/v20191204/tos_Omon_IITM-ESM_1pctCO2_r1i1p1f1_gn_193001.nc",
        ).as_posix(),
        "CMIP6_OCE_HALO_CNRM": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1-HR/historical/r1i1p1f2/Omon/tos/gn/v20191021/tos_Omon_CNRM-CM6-1-HR_historical_r1i1p1f2_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNSTR_FESOM_LR": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-ESM-1-1-LR/historical/r1i1p1f1/Omon/tos/gn/v20200212/tos_Omon_AWI-ESM-1-1-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNSTR_ICON_A": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/ICON-ESM-LR/historical/r1i1p1f1/Amon/tas/gn/v20210215/tas_Amon_ICON-ESM-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNSTR_VERT_ICON_O": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/ICON-ESM-LR/historical/r1i1p1f1/Omon/thetao/gn/v20210215/thetao_Omon_ICON-ESM-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_UNTAGGED_MISSVALS": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2-FV2/historical/r1i1p1f1/Omon/tos/gn/v20191120/tos_Omon_CESM2-FV2_historical_r1i1p1f1_gn_200001.nc",
        ).as_posix(),
        "CMIP6_STAGGERED_UCOMP": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tauuo/gn/v20200909/tauuo_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_STAGGERED_VCOMP": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/tauvo/gn/v20190710/tauvo_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001.nc",
        ).as_posix(),
        "CMIP6_FILLVALUE": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CMIP6/data/CMIP6/CMIP/NCAR/CESM2-WACCM/historical/r1i1p1f1/day/tas/gn/v20190227/tas_day_CESM2-WACCM_historical_r1i1p1f1_gn_20000101-20091231.nc",
        ).as_posix(),
        "CMIP6_ZONMEAN_A": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/msftmz/gn/v20190710/msftmz_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_191001.nc",
        ).as_posix(),
        "CMIP6_ZONMEAN_B": Path(
            esgf_cache_dir,
            "master/test_data/badc/cmip6/data/CMIP6/CMIP/NCC/NorCPM1/historical/r22i1p1f1/Omon/msftmz/grz/v20200724/msftmz_Omon_NorCPM1_historical_r22i1p1f1_grz_185001.nc",
        ).as_posix(),
        "CORDEX_TAS_ONE_TIMESTEP": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CORDEX/data/cordex/output/EUR-22/GERICS/MPI-M-MPI-ESM-LR/rcp85/r1i1p1/GERICS-REMO2015/v1/mon/tas/v20191029/tas_EUR-22_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_GERICS-REMO2015_v1_mon_202101.nc",
        ).as_posix(),
        "CORDEX_TAS_ONE_TIMESTEP_ANT": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CORDEX/data/cordex/output/ANT-44/KNMI/ECMWF-ERAINT/evaluation/r1i1p1/DMI-HIRHAM5/v1/day/tas/v20201001/tas_ANT-44_ECMWF-ERAINT_evaluation_r1i1p1_DMI-HIRHAM5_v1_day_20060101.nc",
        ).as_posix(),
        "CORDEX_TAS_NO_BOUNDS": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/CORDEX/data/cordex/output/EUR-11/KNMI/MPI-M-MPI-ESM-LR/rcp85/r1i1p1/KNMI-RACMO22E/v1/mon/tas/v20190625/tas_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_KNMI-RACMO22E_v1_mon_209101.nc",
        ).as_posix(),
        "ATLAS_v1_CMIP5": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-cica-atlas/CMIP5/rcp26/pr_CMIP5_rcp26_mon_200601-210012.nc",
        ).as_posix(),
        "ATLAS_v1_EOBS": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-cica-atlas/E-OBS/sfcwind_E-OBS_mon_195001-202112.nc",
        ).as_posix(),
        "ATLAS_v1_ERA5": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-cica-atlas/ERA5/psl_ERA5_mon_194001-202212.nc",
        ).as_posix(),
        "ATLAS_v1_CORDEX": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-cica-atlas/CORDEX-CORE/historical/huss_CORDEX-CORE_historical_mon_197001.nc",
        ).as_posix(),
        "ATLAS_v1_EOBS_GRID": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-cica-atlas/E-OBS/t_E-OBS_mon_195001.nc",
        ).as_posix(),
        "ATLAS_v0_CORDEX_NAM": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-ipcc-ar6-atlas/CORDEX-NAM/historical/rx1day_CORDEX-NAM_historical_mon_197001-200512.nc",
        ).as_posix(),
        "ATLAS_v0_CMIP6": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-ipcc-ar6-atlas/CMIP6/ssp245/sst_CMIP6_ssp245_mon_201501-210012.nc",
        ).as_posix(),
        "ATLAS_v0_CORDEX_ANT": Path(
            esgf_cache_dir,
            "master/test_data/pool/data/c3s-ipcc-ar6-atlas/CORDEX-ANT/rcp45/tnn_CORDEX-ANT_rcp45_mon_200601.nc",
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

    def __enter__(self):
        self.logger.enable("clisops")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """If test is supplying caplog, pytest will manage teardown."""

        self.logger.disable("clisops")
        if not self.using_caplog:
            try:
                self.logger.remove()
            except ValueError:
                pass
