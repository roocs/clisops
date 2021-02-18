import os

import pytest
from roocs_utils.exceptions import InvalidParameterValue

from clisops import CONFIG, utils


def test_local_config_loads():
    assert "clisops:read" in CONFIG
    assert "file_size_limit" in CONFIG["clisops:write"]


def test_dask_env_variables():
    assert os.getenv("MKL_NUM_THREADS") == "1"
    assert os.getenv("OPENBLAS_NUM_THREADS") == "1"
    assert os.getenv("OMP_NUM_THREADS") == "1"
