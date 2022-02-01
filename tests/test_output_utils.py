import os
import tempfile
from glob import glob
from pathlib import Path
from unittest import mock

import xarray as xr

from clisops import CONFIG, logging
from clisops.utils.common import expand_wildcards
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import (
    get_chunk_length,
    get_da,
    get_output,
    get_time_slices,
)
from tests._common import CMIP5_TAS, CMIP6_TOS

LOGGER = logging.getLogger(__file__)


def _open(coll):
    if isinstance(coll, (str, Path)):
        coll = expand_wildcards(coll)
    if len(coll) > 1:
        ds = xr.open_mfdataset(coll, use_cftime=True, combine="by_coords")
    else:
        ds = xr.open_dataset(coll[0], use_cftime=True)
    return ds


def test_get_time_slices_single_slice(load_esgf_test_data):

    tas = _open(CMIP5_TAS)

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


def test_get_time_slices_multiple_slices(load_esgf_test_data):

    tas = _open(CMIP5_TAS)

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


def test_tmp_dir_created_with_staging_dir():
    # copy part of function that creates tmp dir to check that it is created
    CONFIG["clisops:write"]["output_staging_dir"] = "tests/"
    staging_dir = CONFIG["clisops:write"].get("output_staging_dir", "")

    output_path = "./output_001.nc"

    if os.path.isdir(staging_dir):
        tmp_dir = tempfile.TemporaryDirectory(dir=staging_dir)
        fname = os.path.basename(output_path)
        target_path = os.path.join(tmp_dir.name, fname)
        LOGGER.info(f"Writing to temporary path: {target_path}")
    else:
        target_path = output_path

    assert target_path != "output_001.nc"
    assert len(glob("tests/tmp*")) == 1
    assert "tests/tmp" in glob("tests/tmp*")[0]

    # delete the temporary directory
    tmp_dir.cleanup()


def test_tmp_dir_not_created_with_no_staging_dir():
    # copy part of function that creates tmp dir to check that it is not created when no staging dir
    CONFIG["clisops:write"]["output_staging_dir"] = ""
    staging_dir = CONFIG["clisops:write"].get("output_staging_dir", "")

    output_path = "./output_001.nc"

    if os.path.isdir(staging_dir):
        tmp_dir = tempfile.TemporaryDirectory(dir=staging_dir)
        fname = os.path.basename(output_path)
        target_path = os.path.join(tmp_dir.name, fname)
        LOGGER.info(f"Writing to temporary path: {target_path}")
    else:
        target_path = output_path

    assert target_path == "./output_001.nc"


def test_no_staging_dir(caplog):

    CONFIG["clisops:write"]["output_staging_dir"] = ""
    ds = _open(CMIP5_TAS)
    output_path = get_output(
        ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")()
    )

    assert "Writing to temporary path: " not in caplog.text
    assert output_path == "output_001.nc"

    os.remove("output_001.nc")


def test_invalid_staging_dir(caplog):
    # check stagin dir not used with invalid directory
    CONFIG["clisops:write"]["output_staging_dir"] = "test/not/real/dir/"

    ds = _open(CMIP5_TAS)
    output_path = get_output(
        ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")()
    )
    assert "Writing to temporary path: " not in caplog.text

    assert output_path == "output_001.nc"

    os.remove("output_001.nc")


def test_staging_dir_used(caplog):
    # check staging dir used when valid directory
    CONFIG["clisops:write"]["output_staging_dir"] = "tests/"

    ds = _open(CMIP5_TAS)

    output_path = get_output(
        ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")()
    )

    assert "Writing to temporary path: tests/" in caplog.text
    assert output_path == "output_001.nc"

    os.remove("output_001.nc")


def test_final_output_path_staging_dir():
    # check final output file in correct location with a staging directory used
    CONFIG["clisops:write"]["output_staging_dir"] = "tests/"

    ds = _open(CMIP5_TAS)
    get_output(ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")())

    assert os.path.isfile("./output_001.nc")

    os.remove("output_001.nc")


def test_final_output_path_no_staging_dir():
    # check final output file in correct location with a staging directory is not used
    ds = _open(CMIP5_TAS)
    get_output(ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")())

    assert os.path.isfile("./output_001.nc")

    os.remove("output_001.nc")


def test_tmp_dir_deleted():
    # check temporary directory under stagin dir gets deleted after data has bee staged
    CONFIG["clisops:write"]["output_staging_dir"] = "tests/"

    ds = _open(CMIP5_TAS)
    get_output(ds, output_type="nc", output_dir=".", namer=get_file_namer("simple")())

    # check that no tmpdir directories exist
    assert glob("tests/tmp*") == []

    os.remove("output_001.nc")


def test_unify_chunks_cmip5():
    """
    testing unify chunks with a cmip5 example:
    da.unify_chunks() doesn't change da.chunks
    ds = ds.unify_chunks() doesn't appear to change ds.chunks in our case
    """
    # DataArray unify chunks method
    ds1 = _open(CMIP5_TAS)
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


def test_unify_chunks_cmip6():
    """
    testing unify chunks with a cmip6 example:
    da.unify_chunks() doesn't change da.chunks
    ds = ds.unify_chunks() doesn't appear to change ds.chunks in our case
    """
    # DataArray unify chunks method
    ds1 = _open(CMIP6_TOS)
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
