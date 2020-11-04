import math
import os
from datetime import datetime as dt
from typing import List, Tuple, Union

import dask
import pandas as pd
import xarray as xr
from roocs_utils.utils.common import parse_size
from roocs_utils.xarray_utils import xarray_utils as xu

from clisops import CONFIG, chunk_memory_limit, logging

LOGGER = logging.getLogger(__file__)

SUPPORTED_FORMATS = {
    "netcdf": {"method": "to_netcdf", "extension": "nc"},
    "nc": {"method": "to_netcdf", "extension": "nc"},
    "zarr": {"method": "to_zarr", "extension": "zarr"},
    "xarray": {"method": None, "extension": None},
}

SUPPORTED_SPLIT_METHODS = ["time:auto"]


def check_format(fmt):
    """ Checks requested format exists. """
    if fmt not in SUPPORTED_FORMATS:
        raise KeyError(
            f'Format not recognised: "{fmt}". Must be one of: {SUPPORTED_FORMATS}.'
        )


def get_format_writer(fmt):
    """ Finds the output method for the requested output format. """
    check_format(fmt)
    return SUPPORTED_FORMATS[fmt]["method"]


def get_format_extension(fmt):
    """ Finds the extension for the requested output format. """
    check_format(fmt)
    return SUPPORTED_FORMATS[fmt]["extension"]


def _format_time(tm: Union[str, dt], fmt="%Y-%m-%d"):
    """ Convert to datetime if time is a numpy datetime """
    if not hasattr(tm, "strftime"):
        tm = pd.to_datetime(str(tm))

    return tm.strftime(fmt)


def filter_times_within(times, start=None, end=None):
    """
    Takes an array of datetimes, returning a reduced array if start or end times
    are defined and are within the main array.
    """
    filtered = []
    for tm in times:
        ft = _format_time(tm, "%Y-%m-%dT%H:%M:%S")
        if start is not None and ft < start:
            continue
        if end is not None and ft > end:
            break

        filtered.append(tm)

    return filtered


def get_da(ds):
    """ Returns xr.DataArray when format of ds may be either."""
    if isinstance(ds, xr.DataArray):
        da = ds
    else:
        var_id = xu.get_main_variable(ds)
        da = ds[var_id]

    return da


def get_time_slices(
    ds: Union[xr.Dataset, xr.DataArray],
    split_method,
    start=None,
    end=None,
    file_size_limit: str = None,
) -> List[Tuple[str, str]]:

    """

    Take an xarray Dataset or DataArray, assume it can be split on the time axis
    into a sequence of slices. Optionally, take a start and end date to specify
    a sub-slice of the main time axis.

    Use the prescribed file size limit to generate a list of
    ("YYYY-MM-DD", "YYYY-MM-DD") slices so that the output files do
    not (significantly) exceed the file size limit.

    Parameters
    ----------
    ds: Union[xr.Dataset, xr.DataArray]
    split_method
    start
    end
    file_size_limit: str
      a string specifying "<number><units>".

    Returns
    -------
    List[Tuple[str, str]]
    """

    if split_method not in SUPPORTED_SPLIT_METHODS:
        raise NotImplementedError(
            f"The split method {split_method} is not implemented."
        )

    # Use default file size limit if not provided
    if not file_size_limit:
        file_size_limit = parse_size(CONFIG["clisops:write"]["file_size_limit"])

    da = get_da(ds)

    times = filter_times_within(da.time.values, start=start, end=end)
    n_times = len(times)

    if n_times == 0:
        raise Exception(f"Zero time steps found between {start} and {end}.")

    n_slices = da.nbytes / file_size_limit
    slice_length = int(n_times // n_slices)

    if slice_length == 0:
        raise Exception("Unable to calculate slice length for splitting output files.")

    slices = []
    indx = 0
    final_indx = n_times - 1

    while indx <= final_indx:

        start_indx = indx
        indx += slice_length
        end_indx = indx - 1

        if end_indx > final_indx:
            end_indx = final_indx
        slices.append(
            (f"{_format_time(times[start_indx])}", f"{_format_time(times[end_indx])}")
        )

    return slices


def get_chunk_length(da):
    """
    Calculate the chunk length to use when chunking xarray datasets.

    Based on memory limit provided in config and the size of th dataset.
    """
    size = da.nbytes
    n_times = len(da.time.values)
    mem_limit = parse_size(chunk_memory_limit)

    if size > 0:
        n_chunks = math.ceil(size / mem_limit)
    else:
        n_chunks = 1

    chunk_length = math.ceil(n_times / n_chunks)

    return chunk_length


def _get_chunked_dataset(ds):
    """
    Chunk xr.Dataset and return chunked dataset
    """
    da = get_da(ds)
    chunk_length = get_chunk_length(da)
    chunked_ds = ds.chunk({"time": chunk_length})
    da.unify_chunks()
    return chunked_ds


def get_output(ds, output_type, output_dir, namer):

    """
    Return output after applying chunking and determining
    the output format and chunking
    """
    fmt_method = get_format_writer(output_type)
    LOGGER.info(f"fmt_method={fmt_method}, output_type={output_type}")

    if not fmt_method:
        LOGGER.info(f"Returning output as {type(ds)}")
        return ds

    file_name = namer.get_file_name(ds, fmt=output_type)
    chunked_ds = _get_chunked_dataset(ds)

    if not output_dir:
        output_dir = "."
    output_path = os.path.join(output_dir, file_name)

    # TODO: writing output works currently only in sync mode, see:
    #  - https://github.com/roocs/rook/issues/55
    #  - https://docs.dask.org/en/latest/scheduling.html
    with dask.config.set(scheduler="synchronous"):

        writer = getattr(chunked_ds, fmt_method)
        delayed_obj = writer(output_path, compute=False)
        delayed_obj.compute()

    LOGGER.info(f"Wrote output file: {output_path}")
    return output_path
