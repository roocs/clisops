import pytest
import xarray as xr

from clisops import CONFIG
from clisops.exceptions import InvalidProject
from clisops.ops.subset import subset
from clisops.parameter import time_interval
from clisops.utils.file_namers import get_file_namer


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


def test_SimpleFileNamer_with_chunking(tmpdir, mini_esgf_data):
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
    area = (0.0, 10.0, 175.0, 90.0)

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size
    outputs = subset(
        ds=mini_esgf_data["CMIP5_TAS"],
        time=time_interval(start_time, end_time),
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

    class Thing:
        pass

    mock_ds = Thing()
    mock_ds.attrs = {}

    with pytest.raises(InvalidProject):
        s.get_file_name(mock_ds)


def test_StandardFileNamer_cmip5(mini_esgf_data):
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["CMIP5_TAS"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [(_ds, "tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-22991216.nc")]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_cmip5_use_default_attr_names(mini_esgf_data):
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["CMIP5_TAS"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [(_ds, "tas_mon_no-model_rcp85_r1i1p1_20051216-22991216.nc")]
    del _ds.attrs["model_id"]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_cmip6(mini_esgf_data):
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["CMIP6_SICONC"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [(_ds, "siconc_SImon_CanESM5_historical_r1i1p1f1_gn_18500116-20141216.nc")]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_cmip6_use_default_attr_names(mini_esgf_data):
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["CMIP6_SICONC"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (_ds, "siconc_SImon_no-model_historical_r1i1p1f1_no-grid_18500116-20141216.nc")
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
def test_StandardFileNamer_c3s_cordex(mini_esgf_data):
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["C3S_CORDEX_NAM_PR"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (
            _ds,
            "pr_NAM-22_NOAA-GFDL-GFDL-ESM2M_rcp45_r1i1p1_OURANOS-CRCM5_v1_day_20510101-20601231.nc",
        )
    ]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


@pytest.mark.skipif(
    condition="platform.system() == 'Windows'",
    reason="Git modules not working on Windows",
)
def test_StandardFileNamer_c3s_cordex_use_default_attr_names(mini_esgf_data):
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["C3S_CORDEX_NAM_PR"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (
            _ds,
            "pr_no-domain_NOAA-GFDL-GFDL-ESM2M_rcp45_rXiXpX_OURANOS-CRCM5_v1_day_20510101-20601231.nc",
        )
    ]
    del _ds.attrs["CORDEX_domain"]
    del _ds.attrs["driving_model_ensemble_member"]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_c3s_atlas_v0(mini_esgf_data):
    """Test C3S ATLAS v0 (c3s-ipcc-ar6-atlas) filenamer"""
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["ATLAS_v0_CORDEX_NAM"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (
            _ds,
            "rx1day_CORDEX-NAM_historical_mon_19700101-20051201.nc",
        )
    ]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected


def test_StandardFileNamer_c3s_atlas_v1(mini_esgf_data):
    """Test C3S ATLAS v1 (c3s-cica-atlas) filenamer"""
    s = get_file_namer("standard")()

    _ds = xr.open_mfdataset(
        mini_esgf_data["ATLAS_v1_ERA5"],
        use_cftime=True,
        combine="by_coords",
    )

    checks = [
        (
            _ds,
            "psl_ERA5_no-expt_mon_19400101-20221201.nc",
        )
    ]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert resp == expected
