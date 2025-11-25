"""Utility functions for handling output formats and file writing in CLISOPS."""

import os
import shutil
import tempfile
import time
from datetime import datetime as dt
from datetime import timedelta as td
from math import ceil
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from clisops import CONFIG, chunk_memory_limit
from clisops.utils.common import parse_size
from clisops.utils.dataset_utils import get_main_variable

SUPPORTED_FORMATS = {
    "netcdf": {"method": "to_netcdf", "extension": "nc", "engine": "h5netcdf"},
    "nc": {"method": "to_netcdf", "extension": "nc", "engine": "h5netcdf"},
    "zarr": {"method": "to_zarr", "extension": "zarr", "engine": "zarr"},
    "xarray": {"method": None, "extension": None},
}

SUPPORTED_SPLIT_METHODS = ["time:auto"]


def check_format(fmt: str) -> None:
    """
    Check that the requested format exists.

    Parameters
    ----------
    fmt : str
        The format to check against the supported formats.

    Raises
    ------
    KeyError
        If the format is not recognized.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise KeyError(f'Format not recognised: "{fmt}". Must be one of: {SUPPORTED_FORMATS}.')


def get_format_writer(fmt: str) -> str | None:
    """
    Find the output method for the requested output format.

    Parameters
    ----------
    fmt : str
        The format for which to find the output method.

    Returns
    -------
    str or None
        The method to use for writing the output format, or None if no method is defined.
    """
    check_format(fmt)
    return SUPPORTED_FORMATS[fmt]["method"]


def get_format_extension(fmt: str) -> str:
    """
    Find the extension for the requested output format.

    Parameters
    ----------
    fmt : str
        The format for which to find the file extension.

    Returns
    -------
    str
        The file extension associated with the requested format.
    """
    check_format(fmt)
    return SUPPORTED_FORMATS[fmt]["extension"]


def get_format_engine(fmt: str) -> str:
    """
    Find the engine for the requested output format.

    Parameters
    ----------
    fmt : str
        The format for which to find the engine.

    Returns
    -------
    str
        The engine to use for writing the output format.
    """
    check_format(fmt)
    return SUPPORTED_FORMATS[fmt]["engine"]


def _format_time(tm: str | dt, fmt="%Y-%m-%d"):
    """Convert to datetime if time is a numpy datetime."""
    if not hasattr(tm, "strftime"):
        tm = pd.to_datetime(str(tm))

    return tm.strftime(fmt)


def filter_times_within(times: np.array, start: str | None = None, end: str | None = None):
    """
    Return a reduced array if start or end times are defined and are within the main array.

    Parameters
    ----------
    times : array-like
        An array of datetime objects or strings representing times.
    start : str, optional
        A string representing the start date in "YYYY-MM-DD" format. If None, no start filter is applied.
    end : str, optional
        A string representing the end date in "YYYY-MM-DD" format. If None, no end filter is applied.

    Returns
    -------
    list
        A list of datetime objects that fall within the specified start and end times.
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


def get_da(ds: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """
    Return xr.DataArray when the format of ds may be either xr.Dataset or xr.DataArray.

    If ds is an xr.Dataset, it will extract the main variable DataArray.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The dataset or data array to extract the main variable from.

    Returns
    -------
    xr.DataArray
        The main variable DataArray from the dataset.
    """
    if isinstance(ds, xr.DataArray):
        da = ds
    else:
        var_id = get_main_variable(ds)
        da = ds[var_id]

    return da


def get_time_slices(
    ds: xr.Dataset | xr.DataArray,
    split_method: str,
    start: str | None = None,
    end: str | None = None,
    file_size_limit: str | None = None,
) -> list[tuple[str, str]]:
    """
    Get time slices for a dataset or data array.

    Take an xarray Dataset or DataArray, assume it can be split on the time axis into a sequence of slices.
    Optionally, take a start and end date to specify a sub-slice of the main time axis.

    Use the prescribed file size limit to generate a list of ("YYYY-MM-DD", "YYYY-MM-DD") slices
    so that the output files do not (significantly) exceed the file size limit.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        A dataset or data array that contains a time dimension.
    split_method : str
        The method to use for splitting the dataset.
    start : str, optional
        A string specifying the start date in "YYYY-MM-DD" format.
    end : str, optional
        A string specifying the end date in "YYYY-MM-DD" format.
    file_size_limit : str
        A string specifying "<number><units>".

    Returns
    -------
    list of tuples
        A list of tuples, each containing two strings representing the start and end dates of each slice.
    """
    if split_method not in SUPPORTED_SPLIT_METHODS:
        raise NotImplementedError(f"The split method {split_method} is not implemented.")

    # Use the default file size limit if not provided
    if not file_size_limit:
        file_size_limit = parse_size(CONFIG["clisops:write"]["file_size_limit"])

    da = get_da(ds)
    slices = []

    try:
        times = filter_times_within(da.time.values, start=start, end=end)
    # catch where "time" attribute cannot be accessed in ds
    except AttributeError:
        slices.append(None)
        return slices

    n_times = len(times)

    if n_times == 0:
        raise Exception(f"Zero time steps found between {start} and {end}.")

    n_slices = da.nbytes / file_size_limit
    slice_length = int(n_times // n_slices)

    if slice_length == 0:
        raise Exception("Unable to calculate slice length for splitting output files.")

    ind_x = 0
    final_ind_x = n_times - 1

    while ind_x <= final_ind_x:
        start_ind_x = ind_x
        ind_x += slice_length
        end_ind_x = ind_x - 1

        if end_ind_x > final_ind_x:
            end_ind_x = final_ind_x
        slices.append((f"{_format_time(times[start_ind_x])}", f"{_format_time(times[end_ind_x])}"))

    return slices


def get_chunk_length(da: xr.DataArray) -> int:
    """
    Calculate the chunk length to use when chunking xarray datasets.

    Based on the memory limit provided in config and the size of the dataset.

    Parameters
    ----------
    da : xr.DataArray
        The data array to be chunked.

    Returns
    -------
    int
        The length of the chunk to be used for the time dimension.
    """
    size = da.nbytes
    n_times = len(da.time.values)
    mem_limit = parse_size(chunk_memory_limit)

    if size > 0:
        n_chunks = ceil(size / mem_limit)
    else:
        n_chunks = 1

    chunk_length = ceil(n_times / n_chunks)

    return chunk_length


def _get_chunked_dataset(ds):
    """Chunk xr.Dataset and return chunked dataset."""
    da = get_da(ds)
    chunk_length = get_chunk_length(da)
    chunked_ds = ds.chunk({"time": chunk_length})
    return chunked_ds


def get_output(ds: xr.Dataset, output_type: str, output_dir: str | Path, namer: object):
    """
    Return output after applying chunking and determining the output format and chunking.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be processed.
    output_type : str
        The type of output format to be used (e.g., "netcdf", "zarr").
    output_dir : str or Path
        The directory where the output file will be saved. If None, the current directory is used.
    namer : object
        An object responsible for generating the file name based on the dataset attributes and output type.

    Returns
    -------
    str or xarray.Dataset
        The path to the output file if written, or the original dataset if no writing is performed.
    """
    format_writer = get_format_writer(output_type)
    logger.info(f"format_writer={format_writer}, output_type={output_type}")

    # If there is no writer for this output type, just return the `ds` object
    if not format_writer:
        logger.info(f"Returning output as {type(ds)}")
        return ds

    # Use the file namer to get the required file name
    file_name = namer.get_file_name(ds, fmt=output_type)

    # Get the chunked Dataset object
    try:
        chunked_ds = _get_chunked_dataset(ds)
    # Catch where "time" attribute is not found in ds:
    # - just set the chunked Dataset to the original Dataset
    except AttributeError:
        chunked_ds = ds

    # If `output_dir` is not set, use the current directory
    if not output_dir:
        output_dir = Path().cwd().expanduser()
    else:
        output_dir = Path(output_dir)

    # Set output path
    output_path = output_dir.joinpath(file_name).as_posix()

    # If "output_staging_dir" is set, then write outputs to a temporary
    # dir, then move them to the correct: output_path
    staging_dir = CONFIG["clisops:write"].get("output_staging_dir", "")

    if os.path.isdir(staging_dir):
        tmp_dir = tempfile.TemporaryDirectory(dir=staging_dir)
        fname = os.path.basename(output_path)
        target_path = os.path.join(tmp_dir.name, fname)
        logger.info(f"Writing to temporary path: {target_path}")
    else:
        target_path = output_path

    # TODO: writing output works currently only in sync mode, see:
    #  - https://github.com/roocs/rook/issues/55
    #  - https://docs.dask.org/en/latest/scheduling.html
    with dask.config.set(scheduler="synchronous"):
        writer = getattr(chunked_ds, format_writer)
        engine = get_format_engine(output_type)
        delayed_obj = writer(target_path, engine=engine, compute=False)
        delayed_obj.compute()

    # If "output_staging_dir" is set, then pause, move the output file,
    # and clean up the temporary directory
    if os.path.isdir(staging_dir):
        shutil.move(target_path, output_path)
        # Sleeping, to allow file system caching/syncing delays
        time.sleep(3)
        tmp_dir.cleanup()

    logger.info(f"Wrote output file: {output_path}")
    return output_path


def _fix_str_encoding(s, encoding="utf-8"):
    """
    Helper function to fix string encoding of surrogates.

    Parameters
    ----------
    s : str or byte
        The string to be fixed. If the input is not of type str or bytes,
        it is returned as is.
    encoding : str, optional
        The encoding to be used. Default is "utf-8".

    Returns
    -------
    str
        The fixed string.
    """
    if isinstance(s, bytes):
        # Decode directly from bytes, potentially replacing undecodable sequences
        try:
            return s.decode(encoding, errors="surrogateescape")
        except UnicodeDecodeError:
            return s.decode(encoding, errors="replace")
    elif isinstance(s, str):
        try:
            # If this works, no surrogates present:
            s.encode(encoding)
            return s
        except UnicodeEncodeError:
            # Handle surrogate escapes
            b = s.encode(encoding, "surrogateescape")
            return b.decode(encoding, errors="replace")
    return s


def fix_netcdf_attrs_encoding(ds, encoding="utf-8"):
    """
    Fix strings that contain invalid chars in Dataset attrs to be safe for NetCDF writing.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset with attrs to be fixed.
    encoding : str, optional
        The encoding to be used. Default is "utf-8".

    Returns
    -------
    xarray.Dataset
        The dataset with fixed attrs.
    """
    # Work on a shallow copy so original ds is untouched
    ds = ds.copy()

    # Fix global attributes
    for k, v in list(ds.attrs.items()):
        fixed_v = _fix_str_encoding(v, encoding)
        if fixed_v is not v:
            ds.attrs[k] = fixed_v

    # Fix variable attributes
    for var in ds.variables:
        for k, v in list(ds[var].attrs.items()):
            fixed_v = _fix_str_encoding(v, encoding)
            if fixed_v is not v:
                ds[var].attrs[k] = fixed_v

    return ds


class FileLock:
    """
    Create and release a lockfile.

    Adapted from https://github.com/cedadev/cmip6-object-store/cmip6_zarr/file_lock.py

    Parameters
    ----------
    fpath : str
        The file path for the lock file to be created.
    """

    def __init__(self, fpath):
        """Initialize Lock for 'fpath'."""
        self._fpath = fpath
        dr = os.path.dirname(fpath)
        if dr and not os.path.isdir(dr):
            os.makedirs(dr)

        self.state = "UNLOCKED"

    def acquire(self, timeout: int = 10):
        """
        Create actual lockfile, raise error if already exists beyond 'timeout'.

        Parameters
        ----------
        timeout : int
            Maximum time in seconds to wait for the lockfile to be created.
            Default is 10 seconds.

        Raises
        ------
        Exception
            If the lockfile cannot be created within the specified timeout.
        """
        start = dt.now()
        deadline = start + td(seconds=timeout)

        while dt.now() < deadline:
            if not os.path.isfile(self._fpath):
                Path(self._fpath).touch()
                break

            time.sleep(3)
        else:
            raise Exception(f"Could not obtain file lock on {self._fpath}")

        self.state = "LOCKED"

    def release(self):
        """Release lock, i.e. delete lockfile."""
        if os.path.isfile(self._fpath):
            try:
                os.remove(self._fpath)
            except FileNotFoundError:
                logger.info("Lock file already removed.")
                pass

        self.state = "UNLOCKED"


def create_lock(fname: str | Path) -> FileLock | None:
    """
    Check whether lockfile already exists and else creates lockfile.

    Parameters
    ----------
    fname : str
        Path of the lockfile to be created.

    Returns
    -------
    FileLock or None
        Returns a FileLock object if the lockfile is created successfully,
        or None if the lockfile already exists.
    """
    lock_obj = FileLock(fname)
    try:
        lock_obj.acquire(timeout=10)
        locked = False
    except Exception as exc:
        if str(exc) == f"Could not obtain file lock on {fname}":
            locked = True
        else:
            raise Exception(exc)
    if locked:
        return None
    else:
        return lock_obj
