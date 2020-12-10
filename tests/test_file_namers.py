import platform

import pytest
import xarray as xr

from clisops import CONFIG
from clisops.ops.subset import subset
from clisops.utils.file_namers import get_file_namer

from ._common import CMIP5_TAS


def test_SimpleFileNamer():
    s = get_file_namer("simple")()

    checks = [
        (("my.stuff", "netcdf"), "output_001.nc"),
        (("other", "netcdf"), "output_002.nc"),
    ]

    for args, expected in checks:
        resp = s.get_file_name(*args)
        assert resp == expected


def test_SimpleFileNamer_no_fmt():
    s = get_file_namer("simple")()

    checks = (("my.stuff", None),)

    for args in checks:
        with pytest.raises(KeyError):
            s.get_file_name(*args)


def test_SimpleFileNamer_with_chunking(tmpdir):
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
    area = (0.0, 10.0, 175.0, 90.0)

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size
    outputs = subset(
        ds=CMIP5_TAS,
        time=(start_time, end_time),
        area=area,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    count = 0
    for output in outputs:
        count += 1
        assert f"output_00{count}.nc" in output


def test_StandardFileNamer_no_project_match():
    s = get_file_namer("standard")()

    class Thing(object):
        pass

    mock_ds = Thing()
    mock_ds.attrs = {}

    with pytest.raises(KeyError):
        s.get_file_name(mock_ds)


def test_StandardFileNamer_cmip5():
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        "tests/mini-esgf-data/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES"
        "/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
        use_cftime=True,
        combine="by_coords",
    )

    checks = [(_ds, "tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-22991216.nc")]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_cmip5_use_default_attr_names():
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        "tests/mini-esgf-data/test_data/badc/cmip5/data/cmip5/output1/MOHC"
        "/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc",
        use_cftime=True,
        combine="by_coords",
    )

    checks = [(_ds, "tas_mon_no-model_rcp85_r1i1p1_20051216-22991216.nc")]
    del _ds.attrs["model_id"]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_cmip6():
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        "tests/mini-esgf-data/test_data/badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/SImon"
        "/siconc/gn/latest/*.nc",
        use_cftime=True,
        combine="by_coords",
    )

    checks = [(_ds, "siconc_SImon_CESM2_historical_r1i1p1f1_gn_18500115-20141215.nc")]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_cmip6_use_default_attr_names():
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        "tests/mini-esgf-data/test_data/badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/SImon"
        "/siconc/gn/latest/*.nc",
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (_ds, "siconc_SImon_no-model_historical_r1i1p1f1_no-grid_18500115-20141215.nc")
    ]
    del _ds.attrs["source_id"]
    del _ds.attrs["grid_label"]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


@pytest.mark.skipif(
    condition="platform.system() == 'Windows'",
    reason="Git modules not working on Windows",
)
def test_StandardFileNamer_c3s_cordex():
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        "tests/mini-esgf-data/test_data/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cordex/output/EUR-11"
        "/IPSL/MOHC-HadGEM2-ES/rcp85/r1i1p1/IPSL-WRF381P/v1/day/psl/v20190212/*.nc",
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (
            _ds,
            "psl_EUR-11_MOHC-HadGEM2-ES_rcp85_r1i1p1_IPSL-WRF381P_v1_day_20060101-20991201.nc",
        )
    ]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


@pytest.mark.skipif(
    condition="platform.system() == 'Windows'",
    reason="Git modules not working on Windows",
)
def test_StandardFileNamer_c3s_cordex_use_default_attr_names():
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        "tests/mini-esgf-data/test_data/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cordex/output/EUR-11"
        "/IPSL/MOHC-HadGEM2-ES/rcp85/r1i1p1/IPSL-WRF381P/v1/day/psl/v20190212/*.nc",
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (
            _ds,
            "psl_no-domain_MOHC-HadGEM2-ES_rcp85_rXiXpX_IPSL-WRF381P_v1_day_20060101-20991201.nc",
        )
    ]
    del _ds.attrs["CORDEX_domain"]
    del _ds.attrs["driving_model_ensemble_member"]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected
