import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _pytest.logging import caplog as _caplog  # noqa
from git import Repo

from clisops.utils import get_file
from tests._common import MINI_ESGF_CACHE_DIR, write_roocs_cfg

write_roocs_cfg()

ESGF_TEST_DATA_REPO_URL = "https://github.com/roocs/mini-esgf-data"


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

    return xr.DataArray(
        np.random.lognormal(10, 1, (nt, nx, ny)),
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


@pytest.fixture
def cmip5_tas_file():
    return str(
        get_file(
            "cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc",
        )
    )


@pytest.fixture
def cmip6_o3():
    return str(
        get_file(
            "cmip6/o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc",
        )
    )


# Fixture to load mini-esgf-data repository used by roocs tests
@pytest.fixture(scope="session", autouse=True)
def load_esgf_test_data():
    """
    This fixture ensures that the required test data repository
    has been cloned to the cache directory within the home directory.
    """
    branch = "master"
    target = os.path.join(MINI_ESGF_CACHE_DIR, branch)

    if not os.path.isdir(MINI_ESGF_CACHE_DIR):
        os.makedirs(MINI_ESGF_CACHE_DIR)

    if not os.path.isdir(target):
        repo = Repo.clone_from(ESGF_TEST_DATA_REPO_URL, target)
        repo.git.checkout(branch)

    elif os.environ.get("ROOCS_AUTO_UPDATE_TEST_DATA", "true").lower() != "false":
        repo = Repo(target)
        repo.git.checkout(branch)
        repo.remotes[0].pull()
