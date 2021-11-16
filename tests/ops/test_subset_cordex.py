from roocs_utils.parameter.param_utils import time_interval

from clisops.ops.subset import subset

from .._common import C3S_CORDEX_AFR_TAS, C3S_CORDEX_NAM_PR, _check_output_nc


def test_subset_cordex_afr(load_esgf_test_data, tmpdir):
    """Test subset on cordex data AFR domain"""

    result = subset(
        ds=C3S_CORDEX_AFR_TAS,
        time=time_interval("2000-01-01", "2001-12-31"),
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result,
        fname="tas_AFR-22_MPI-M-MPI-ESM-LR_historical_r1i1p1_GERICS-REMO2015_v1_day_20000101-20011231.nc",
    )


def test_subset_cordex_nam(load_esgf_test_data, tmpdir):
    """Test subset on cordex data NAM domain"""

    result = subset(
        ds=C3S_CORDEX_NAM_PR,
        time=time_interval("2051-01-01", "2052-12-31"),
        output_dir=tmpdir,
        output_type="netcdf",
        file_namer="standard",
    )

    _check_output_nc(
        result,
        fname="pr_NAM-22_NOAA-GFDL-GFDL-ESM2M_rcp45_r1i1p1_OURANOS-CRCM5_v1_day_20510101-20521231.nc",
    )
