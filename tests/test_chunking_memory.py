import os
import sys
from unittest.mock import Mock

import pytest
import xarray as xr
from memory_profiler import memory_usage

from clisops.ops.subset import subset
from clisops.utils import output_utils

data = [
    "/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cmip5/output1/MOHC/HadGEM2-ES/rcp45/mon/atmos/Amon"
    "/r1i1p1/vas/v20111128/*.nc",
    "/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cmip5/output1/MOHC/HadGEM2-ES/rcp45/mon/atmos/Amon"
    "/r3i1p1/vas/v20111205/*.nc",
]


def subset_for_test(real_data, start_time, end_time, output_dir):
    subset(
        ds=real_data,
        time=(start_time, end_time),
        output_type="netcdf",
        output_dir=output_dir,
        file_namer="simple",
    )


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_small_limit(tmpdir):
    # 6 chunks
    mem_limit = "45MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[0], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 45

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_smallest_limit(tmpdir):
    # 9 chunks
    mem_limit = "30MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[0], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 30

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_large_limit(tmpdir):
    # 1 chunk
    mem_limit = "570MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[0], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 570

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_medium_limit(tmpdir):
    # 3 chunks
    mem_limit = "100MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[0], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 100

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_smallest_limit_repeat(tmpdir):
    # 9 chunks
    mem_limit = "30MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[0], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 30

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_small_limit_data1(tmpdir):
    # 3 chunks
    mem_limit = "45MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[1], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 45

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_smallest_limit_data1(tmpdir):
    # 5 chunks
    mem_limit = "30MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[1], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 30

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_large_limit_data1(tmpdir):
    # 1 chunk
    mem_limit = "570MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[1], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 570

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_medium_limit_data1(tmpdir):
    # 2 chunks
    mem_limit = "100MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[1], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 100

    assert max(memory) <= upper_limit


@pytest.mark.skipif(
    os.path.isdir("/group_workspaces") is False, reason="data not available"
)
def test_memory_limit_subset_smallest_limit_data1_repeat(tmpdir):
    # 5 chunks
    mem_limit = "30MiB"
    output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    memory = memory_usage((subset_for_test, (data[1], start_time, end_time, tmpdir)))
    print(max(memory))
    upper_limit = 30

    assert max(memory) <= upper_limit


# def _subset_for_test(ds, time, area, output_dir):
#     level = None
#     args = map_params(ds, time, area, level)
#
#     result_ds = _subset(
#         ds=ds,
#         args=args
#     )
#     namer = get_file_namer("simple")()
#     get_output(result_ds, "netcdf", output_dir, namer)
#
#
# @pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
# def test_memory_limit():
#     """ check memory does not greatly exceed dask chunk limit """
#
#     real_data = (
#         "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES"
#         "/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc"
#     )
#
#     start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
#
#     config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
#     config_mem_limit = CONFIG["clisops:read"]["chunk_memory_limit"]
#
#     CONFIG["clisops:write"]["file_size_limit"] = "95MiB"
#     CONFIG["clisops:read"]["chunk_memory_limit"] = "80MiB"
#
#     memory = memory_usage((subset_for_test, (real_data, start_time, end_time)))
#
#     upper_limit = 80 * 1.1
#
#     assert max(memory) <= upper_limit
#
#     CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size
#     CONFIG["clisops:read"]["chunk_memory_limit"] = config_mem_limit
#
#
# @pytest.mark.skipif(
#     os.path.isdir("/group_workspaces") is False, reason="data not available"
# )
# def test_memory_limit_bigger_file():
#     real_data = (
#         "/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cordex/output/EUR-11/IPSL/MOHC-HadGEM2-ES/rcp85"
#         "/r1i1p1/IPSL-WRF381P/v1/day/psl/v20190212/*.nc"
#     )
#
#     start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
#
#     config_mem_limit = CONFIG["clisops:read"]["chunk_memory_limit"]
#     config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
#
#     CONFIG["clisops:write"]["file_size_limit"] = "750MiB"
#     CONFIG["clisops:read"]["chunk_memory_limit"] = "100MiB"
#
#     memory = memory_usage((subset_for_test, (real_data, start_time, end_time)))
#
#     upper_limit = 100 * 1.1
#
#     assert max(memory) <= upper_limit
#
#     CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size
#     CONFIG["clisops:read"]["chunk_memory_limit"] = config_mem_limit
#
#
# @pytest.fixture(
#     params=[
#         "30MiB",
#         "570MiB",
#         "250MiB"
#     ]
# )
# def mem_limit(request):
#     id = request.param
#     return id
#
#
# @pytest.mark.skipif(
#     os.path.isdir("/group_workspaces") is False, reason="data not available"
# )
# def test_memory_limit__subset(mem_limit, tmpdir):
#     output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)
#
#     ds = xr.open_mfdataset("/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cmip5/output1/MOHC/HadGEM2-ES"
#                            "/rcp85/day/atmos/day/r1i1p1/rsds/v20111128/*.nc")
#
#     time = "2001-01-01T00:00:00/2200-12-30T00:00:00"
#     # area = (0.0, 49.0, 10.0, 65.0)
#     area = None
#     memory = memory_usage((_subset_for_test, (ds, time, area, tmpdir)))
#     print(memory)
#     upper_limit = 500
#
#     assert max(memory) <= upper_limit
#
# # "/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cmip5/output1/MOHC/HadGEM2-ES"
# # "/rcp85/day/atmos/day/r1i1p1/ta/v20111128/*.nc"
#
# # 30GB dataset
#
#
# @pytest.mark.skipif(
#     os.path.isdir("/group_workspaces") is False, reason="data not available"
# )
# def test_memory_limit_subset(mem_limit, tmpdir):
#     output_utils.get_chunk_mem_limit = Mock(return_value=mem_limit)
#
#     real_data = "/group_workspaces/jasmin2/cp4cds1/vol1/data/c3s-cmip5/output1/MOHC/HadGEM2-ES" \
#                 "/rcp85/day/atmos/day/r1i1p1/rsds/v20111128/*.nc"
#
#     start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
#
#     memory = memory_usage((subset_for_test, (real_data, start_time, end_time, tmpdir)))
#     print(max(memory))
#     upper_limit = 500
#
#     assert max(memory) <= upper_limit
