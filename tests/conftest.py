import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _pytest.logging import caplog as _caplog  # noqa
from packaging.version import Version

from clisops.core.regrid import XARRAY_INCOMPATIBLE_VERSION
from clisops.utils import testing
from clisops.utils.testing import stratus as _stratus
from clisops.utils.testing import write_roocs_cfg as _write_roocs_cfg


@pytest.fixture(autouse=True)
def roocs_cfg(tmp_path):
    cfg_path = _write_roocs_cfg(None, tmp_path)
    # point to roocs cfg in environment
    os.environ["ROOCS_CONFIG"] = cfg_path


@pytest.fixture
def tmp_netcdf_filename(tmp_path):
    return tmp_path.joinpath("testfile.nc")


@pytest.fixture
def tas_series():
    def _tas_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tas",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: mean within days",
                "units": "K",
            },
        )

    return _tas_series


@pytest.fixture
def tasmax_series():
    def _tasmax_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tasmax",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: maximum within days",
                "units": "K",
            },
        )

    return _tasmax_series


@pytest.fixture
def tasmin_series():
    def _tasmin_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tasmin",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: minimum within days",
                "units": "K",
            },
        )

    return _tasmin_series


@pytest.fixture
def pr_series():
    def _pr_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: sum over day",
                "units": "kg m-2 s-1",
            },
        )

    return _pr_series


@pytest.fixture
def pr_ndseries():
    def _pr_series(values, start="1/1/2000"):
        nt, nx, ny = np.atleast_3d(values).shape
        time = pd.date_range(start, periods=nt, freq=pd.DateOffset(days=1))
        x = np.arange(nx)
        y = np.arange(ny)
        return xr.DataArray(
            values,
            coords=[time, x, y],
            dims=("time", "x", "y"),
            name="pr",
            attrs={
                "standard_name": "precipitation_flux",
                "cell_methods": "time: sum over day",
                "units": "kg m-2 s-1",
            },
        )

    return _pr_series


@pytest.fixture
def q_series():
    def _q_series(values, start="1/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="q",
            attrs={"standard_name": "dis", "units": "m3 s-1"},
        )

    return _q_series


@pytest.fixture
def ndq_series():
    nx, ny, nt = 2, 3, 5000
    x = np.arange(0, nx)
    y = np.arange(0, ny)

    cx = xr.IndexVariable("x", x)
    cy = xr.IndexVariable("y", y)
    dates = pd.date_range("1900-01-01", periods=nt, freq=pd.DateOffset(days=1))

    time = xr.IndexVariable(
        "time", dates, attrs={"units": "days since 1900-01-01", "calendar": "standard"}
    )
    rs = np.random.RandomState()

    return xr.DataArray(
        rs.lognormal(10, 1, (nt, nx, ny)),
        dims=("time", "x", "y"),
        coords={"time": time, "x": cx, "y": cy},
        attrs={"units": "m^3 s-1", "standard_name": "streamflow"},
    )


@pytest.fixture
def areacella():
    """Return a rectangular grid of grid cell area."""
    r = 6100000
    lon_bnds = np.arange(-180, 181, 1)
    lat_bnds = np.arange(-90, 91, 1)
    dlon = np.diff(lon_bnds)
    dlat = np.diff(lat_bnds)
    lon = np.convolve(lon_bnds, [0.5, 0.5], "valid")
    lat = np.convolve(lat_bnds, [0.5, 0.5], "valid")
    area = (
        r
        * np.radians(dlat)[:, np.newaxis]
        * r
        * np.cos(np.radians(lat)[:, np.newaxis])
        * np.radians(dlon)
    )
    return xr.DataArray(
        data=area,
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        attrs={"r": r, "units": "m^2"},
    )


@pytest.fixture
def rh_series():
    def _rh_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="rh",
            attrs={
                "standard_name": "relative humidity",
                "units": "%",
            },
        )

    return _rh_series


@pytest.fixture
def ws_series():
    def _ws_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="ws",
            attrs={
                "standard_name": "wind speed",
                "units": "km h-1",
            },
        )

    return _ws_series


@pytest.fixture
def huss_series():
    def _huss_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="huss",
            attrs={
                "standard_name": "specific_humidity",
                "units": "",
            },
        )

    return _huss_series


@pytest.fixture
def ps_series():
    def _ps_series(values, start="7/1/2000"):
        coords = pd.date_range(start, periods=len(values), freq=pd.DateOffset(days=1))
        return xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="ps",
            attrs={
                "standard_name": "air_pressure",
                "units": "Pa",
            },
        )

    return _ps_series


@pytest.fixture(scope="session", autouse=True)
def threadsafe_data_dir(tmp_path_factory):
    return tmp_path_factory.getbasetemp().joinpath("data")


@pytest.fixture(scope="session")
def stratus(threadsafe_data_dir, worker_id):
    return _stratus(
        repo=testing.ESGF_TEST_DATA_REPO_URL,
        branch=testing.ESGF_TEST_DATA_VERSION,
        cache_dir=(
            testing.ESGF_TEST_DATA_CACHE_DIR
            if worker_id == "master"
            else threadsafe_data_dir
        ),
    )


@pytest.fixture(scope="session")
def nimbus(threadsafe_data_dir, worker_id):
    return _stratus(
        repo=testing.XCLIM_TEST_DATA_REPO_URL,
        branch=testing.XCLIM_TEST_DATA_VERSION,
        cache_dir=(
            testing.XCLIM_TEST_DATA_CACHE_DIR
            if worker_id == "master"
            else threadsafe_data_dir
        ),
    )


@pytest.fixture
def check_output_nc():
    def _check_output_nc(result, fname="output_001.nc", time=None):
        assert fname in [Path(_).name for _ in result]
        if time:
            ds = xr.open_mfdataset(result, use_cftime=True, decode_timedelta=False)
            time_ = (
                f"{ds.time.values.min().isoformat()}/{ds.time.values.max().isoformat()}"
            )
            assert time == time_

    return _check_output_nc


@pytest.fixture(scope="session", autouse=True)
def load_test_data(worker_id, stratus, nimbus):
    """
    This fixture ensures that the required test data repository
    has been cloned to the cache directory within the home directory.
    """
    repositories = {
        "stratus": {
            "worker_cache_dir": stratus.path,
            "repo": testing.ESGF_TEST_DATA_REPO_URL,
            "branch": testing.ESGF_TEST_DATA_VERSION,
            "cache_dir": testing.ESGF_TEST_DATA_CACHE_DIR,
        },
        "nimbus": {
            "worker_cache_dir": nimbus.path,
            "repo": testing.XCLIM_TEST_DATA_REPO_URL,
            "branch": testing.XCLIM_TEST_DATA_VERSION,
            "cache_dir": testing.XCLIM_TEST_DATA_CACHE_DIR,
        },
    }

    for name, repo in repositories.items():
        testing.gather_testing_data(worker_id=worker_id, **repo)


@pytest.fixture
def c3s_cmip5_tsice():
    return Path(
        # This is now only required for json files
        Path(__file__).parent.absolute(),
        "data",
        "c3s-cmip5/output1/NCC/NorESM1-ME/rcp60/mon/seaIce/OImon/r1i1p1/tsice/v20120614/*.nc",
    ).as_posix()


@pytest.fixture
def c3s_cmip5_tos():
    return Path(
        Path(__file__).parent.absolute(),
        "data",
        "c3s-cmip5/output1/BCC/bcc-csm1-1-m/historical/mon/ocean/Omon/r1i1p1/tos/v20120709/*.nc",
    ).as_posix()


@pytest.fixture
def cmip5_archive_base():
    if "CMIP5_ARCHIVE_BASE" in os.environ:
        return os.environ["CMIP5_ARCHIVE_BASE"]
    return (
        Path(__file__)
        .parent.absolute()
        .joinpath("mini-esgf-data/test_data/badc/cmip5/data")
        .as_posix()
    )


@pytest.fixture
def cmip6_archive_base():
    if "CMIP6_ARCHIVE_BASE" in os.environ:
        return os.environ["CMIP6_ARCHIVE_BASE"]
    return (
        Path(__file__)
        .parent.absolute()
        .joinpath("mini-esgf-data/test_data/badc/cmip6/data")
        .as_posix()
    )


@pytest.fixture(scope="session", autouse=True)
def mini_esgf_data(stratus):
    return (
        testing.get_esgf_file_paths(stratus.path)
        | testing.get_esgf_glob_paths(stratus.path)
        | testing.get_kerchunk_datasets()
    )


@pytest.fixture(scope="session")
def esgf_kerchunk_urls():
    kerchunk = (
        "https://gws-access.jasmin.ac.uk/public/cmip6_prep/eodh-eocis/kc-indexes-cmip6-http-v1/"
        "CMIP6.CMIP.MOHC.UKESM1-1-LL.1pctCO2.r1i1p1f2.Amon.tasmax.gn.v20220513.json"
    )
    return {"JSON": kerchunk, "ZST": f"{kerchunk}.zst"}


@pytest.fixture(scope="session", autouse=True)
def clisops_test_data():
    test_data = Path(__file__).parent.absolute().joinpath("data")

    return {
        "meridian_geojson": test_data.joinpath("meridian.json").as_posix(),
        "meridian_multi_geojson": test_data.joinpath("meridian_multi.json").as_posix(),
        "poslons_geojson": test_data.joinpath("poslons.json").as_posix(),
        "eastern_canada_geojson": test_data.joinpath("eastern_canada.json").as_posix(),
        "southern_qc_geojson": test_data.joinpath(
            "southern_qc_geojson.json"
        ).as_posix(),
        "small_geojson": test_data.joinpath("small_geojson.json").as_posix(),
        "multi_regions_geojson": test_data.joinpath("multi_regions.json").as_posix(),
    }


# Temporarily required until https://github.com/pydata/xarray/issues/7794 is addressed
@pytest.fixture(scope="session")
def xfail_if_xarray_incompatible():
    if Version(xr.__version__) >= Version(XARRAY_INCOMPATIBLE_VERSION):
        pytest.xfail(
            f"xarray version >= {XARRAY_INCOMPATIBLE_VERSION} "
            f"is not supported for several operations with cf-time indexed arrays. "
            "For more information, see: https://github.com/pydata/xarray/issues/7794."
        )
