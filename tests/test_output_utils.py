import os
import sys
import tempfile
import time
from pathlib import Path

import xarray as xr

from clisops import CONFIG
from clisops.utils.common import expand_wildcards
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import (
    FileLock,
    get_chunk_length,
    get_da,
    get_output,
    get_time_slices,
)
from clisops.utils.testing import ContextLogger


def _open(coll):
    if isinstance(coll, (str, Path)):
        coll = expand_wildcards(coll)
    if len(coll) > 1:
        # issues with dask and cftime
        ds = xr.open_mfdataset(coll, use_cftime=True, combine="by_coords").load()
    else:
        ds = xr.open_dataset(coll[0], use_cftime=True)
    return ds


def test_get_time_slices_single_slice(mini_esgf_data):
    tas = _open(mini_esgf_data["CMIP5_TAS"])

    test_data = [
        (
            tas,
            1000000,
            1,  # Setting file limit to 1000000 bytes
            ("2005-12-16", "2299-12-16"),
        ),
        (tas, None, 1, ("2005-12-16", "2299-12-16")),  # Using size limit from CONFIG
    ]

    split_method = "time:auto"
    for (
        ds,
        limit,
        n_times,
        slices,
    ) in test_data:
        resp = get_time_slices(ds, split_method, file_size_limit=limit)
        assert resp[0] == slices


def test_get_time_slices_multiple_slices(mini_esgf_data):
    tas = _open(mini_esgf_data["CMIP5_TAS"])

    test_data = [
        (
            tas,
            16000,
            4,
            ("2005-12-16", "2089-03-16"),
            ("2089-04-16", "2172-06-16"),
            ("2255-11-16", "2299-12-16"),
        ),
        (
            tas,
            32000,
            2,
            ("2005-12-16", "2172-06-16"),
            None,
            ("2172-07-16", "2299-12-16"),
        ),
    ]

    split_method = "time:auto"
    for ds, limit, n_times, first, second, last in test_data:
        resp = get_time_slices(ds, split_method, file_size_limit=limit)
        assert resp[0] == first
        assert resp[-1] == last

        if second:
            assert resp[1] == second


def test_tmp_dir_created_with_staging_dir(tmpdir):
    with ContextLogger() as _logger:
        _logger.add(sys.stdout, level="INFO")

        staging = Path(tmpdir).joinpath("tests")
        staging.mkdir(exist_ok=True)

        # copy part of function that creates tmp dir to check that it is created
        CONFIG["clisops:write"]["output_staging_dir"] = staging
        staging_dir = CONFIG["clisops:write"].get("output_staging_dir", "")

        output_path = "./output_001.nc"

        if os.path.isdir(staging_dir):
            tmp_dir = tempfile.TemporaryDirectory(dir=staging_dir)
            fname = os.path.basename(output_path)
            target_path = os.path.join(tmp_dir.name, fname)
            _logger.info(f"Writing to temporary path: {target_path}")
        else:
            target_path = output_path

        assert target_path != "output_001.nc"
        temp_test_folders = [f for f in staging.glob("tmp*")]
        assert len(temp_test_folders) == 1
        assert "tests/tmp" in temp_test_folders[0].as_posix()


def test_tmp_dir_not_created_with_no_staging_dir():
    # with ContextLogger() as _logger:
    #     _logger.add(sys.stdout, level="INFO")

    # copy part of function that creates tmp dir to check that it is not created when no staging dir
    CONFIG["clisops:write"]["output_staging_dir"] = ""
    staging_dir = CONFIG["clisops:write"].get("output_staging_dir", "")

    output_path = "./output_001.nc"

    if os.path.isdir(staging_dir):
        tmp_dir = tempfile.TemporaryDirectory(dir=staging_dir)
        fname = os.path.basename(output_path)
        target_path = os.path.join(tmp_dir.name, fname)
    else:
        target_path = output_path

    assert target_path == "./output_001.nc"


def test_no_staging_dir(caplog, mini_esgf_data):
    with ContextLogger(caplog) as _logger:
        _logger.add(sys.stdout, level="INFO")
        caplog.set_level("INFO", logger="clisops")

        CONFIG["clisops:write"]["output_staging_dir"] = ""
        ds = _open(mini_esgf_data["CMIP5_TAS"])
        output_path = get_output(
            ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")()
        )

        assert "Writing to temporary path: " not in caplog.text
        assert output_path == "output_001.nc"

        os.remove("output_001.nc")


def test_invalid_staging_dir(caplog, mini_esgf_data):
    with ContextLogger(caplog) as _logger:
        _logger.add(sys.stdout, level="INFO")
        caplog.set_level("INFO", logger="clisops")

        # check staging dir not used with invalid directory
        CONFIG["clisops:write"]["output_staging_dir"] = "test/not/real/dir/"

        ds = _open(mini_esgf_data["CMIP5_TAS"])
        output_path = get_output(
            ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")()
        )
        assert "Writing to temporary path: " not in caplog.text
        assert output_path == "output_001.nc"

        os.remove("output_001.nc")


def test_staging_dir_used(caplog, tmpdir, mini_esgf_data):
    with ContextLogger(caplog) as _logger:
        _logger.add(sys.stdout, level="INFO")
        caplog.set_level("INFO", logger="clisops")

        # check staging dir used when valid directory
        staging = Path(tmpdir).joinpath("tests")
        staging.mkdir(exist_ok=True)
        CONFIG["clisops:write"]["output_staging_dir"] = str(staging)
        ds = _open(mini_esgf_data["CMIP5_TAS"])

        output_path = get_output(
            ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")()
        )

        assert f"Writing to temporary path: {staging}" in caplog.text
        assert output_path == "output_001.nc"

        Path("output_001.nc").unlink()


def test_final_output_path_staging_dir(mini_esgf_data):
    # check final output file in correct location with a staging directory used
    CONFIG["clisops:write"]["output_staging_dir"] = "tests/"

    ds = _open(mini_esgf_data["CMIP5_TAS"])
    get_output(ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")())

    assert os.path.isfile("./output_001.nc")

    os.remove("output_001.nc")


def test_final_output_path_no_staging_dir(mini_esgf_data):
    # check final output file in correct location with a staging directory is not used
    ds = _open(mini_esgf_data["CMIP5_TAS"])
    get_output(ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")())

    assert os.path.isfile("./output_001.nc")

    os.remove("output_001.nc")


def test_tmp_dir_deleted(tmpdir, mini_esgf_data):
    # check temporary directory under staging dir gets deleted after data has bee staged
    staging = Path(tmpdir).joinpath("tests")
    staging.mkdir(exist_ok=True)
    CONFIG["clisops:write"]["output_staging_dir"] = staging

    # CONFIG["clisops:write"]["output_staging_dir"] = "tests/"

    ds = _open(mini_esgf_data["CMIP5_TAS"])
    get_output(ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")())

    # check that no tmpdir directories exist
    assert len([f for f in staging.glob("tmp*")]) == 0

    os.remove("output_001.nc")


def test_unify_chunks_cmip5(mini_esgf_data):
    """test unify chunks with a cmip5 example.

    da.unify_chunks() doesn't change da.chunks
    ds = ds.unify_chunks() doesn't appear to change ds.chunks in our case
    """
    # DataArray unify chunks method
    ds1 = _open(mini_esgf_data["CMIP5_TAS"])
    da = get_da(ds1)
    chunk_length = get_chunk_length(da)
    chunked_ds1 = ds1.chunk({"time": chunk_length})
    da.unify_chunks()

    # Dataset unify chunks method
    chunk_length = get_chunk_length(da)
    chunked_ds2 = ds1.chunk({"time": chunk_length})
    chunked_ds2_unified = chunked_ds2.unify_chunks()

    # test that da.unify_chunks hasn't changed ds.chunks
    assert chunked_ds1.chunks == chunked_ds2.chunks
    # test that ds = ds.unify_chunks hasn't changed ds.chunks
    assert chunked_ds2.chunks == chunked_ds2_unified.chunks


def test_unify_chunks_cmip6(mini_esgf_data):
    """Test unify chunks with a cmip6 example.

    da.unify_chunks() doesn't change da.chunks
    ds = ds.unify_chunks() doesn't appear to change ds.chunks in our case
    """
    # DataArray unify chunks method
    ds1 = _open(mini_esgf_data["CMIP6_TOS"])
    da = get_da(ds1)
    chunk_length = get_chunk_length(da)
    chunked_ds1 = ds1.chunk({"time": chunk_length})
    da.unify_chunks()

    # Dataset unify chunks method
    chunk_length = get_chunk_length(da)
    chunked_ds2 = ds1.chunk({"time": chunk_length})
    chunked_ds2_unified = chunked_ds2.unify_chunks()

    # test that da.unify_chunks hasn't changed ds.chunks
    assert chunked_ds1.chunks == chunked_ds2.chunks
    # test that ds = ds.unify_chunks hasn't changed ds.chunks
    assert chunked_ds2.chunks == chunked_ds2_unified.chunks


def test_filelock_simple(tmp_path):
    LOCK_FILE = Path(tmp_path, "test.lock")
    DATA_FILE = Path(tmp_path, "test.dat")

    lock = FileLock(LOCK_FILE)

    lock.acquire()
    try:
        time.sleep(1)
        assert os.path.isfile(LOCK_FILE)
        assert lock.state == "LOCKED"
        open(DATA_FILE, "a").write("1")
    finally:
        lock.release()

    time.sleep(1)
    assert not os.path.isfile(LOCK_FILE)


def test_filelock_already_locked(tmp_path):
    LOCK_FILE = Path(tmp_path, "test.lock")

    lock1 = FileLock(LOCK_FILE)
    lock2 = FileLock(LOCK_FILE)

    lock1.acquire(timeout=10)

    try:
        lock2.acquire(timeout=5)
    except Exception as exc:
        assert str(exc) == f"Could not obtain file lock on {LOCK_FILE}"
