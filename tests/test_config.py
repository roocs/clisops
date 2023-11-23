import os

from clisops import CONFIG


def test_local_config_loads():
    assert "clisops:read" in CONFIG
    assert "file_size_limit" in CONFIG["clisops:write"]
    assert "clisops:grid_weights" in CONFIG
    assert "local_weights_dir" in CONFIG["clisops:grid_weights"]
    assert "clisops:coordinate_precision" in CONFIG
    assert "hor_coord_decimals" in CONFIG["clisops:coordinate_precision"]


def test_dask_env_variables():
    assert os.getenv("MKL_NUM_THREADS") == "1"
    assert os.getenv("OPENBLAS_NUM_THREADS") == "1"
    assert os.getenv("OMP_NUM_THREADS") == "1"
