import inspect
import os

import cf_xarray  # noqa
import cftime
import fsspec
import numpy as np
import xarray as xr
from roocs_utils.project_utils import dset_to_filepaths

known_coord_types = ["time", "level", "latitude", "longitude", "realization"]

KERCHUNK_EXTS = [".json", ".zst", ".zstd"]


def _patch_time_encoding(ds, file_list, **kwargs):
    """
    NOTE: Hopefully this will be fixed in Xarray at some point. The problem is that if
          time is present, the multi-file dataset has an empty `encoding` dictionary.

    Reads the first file in `file_list` to read in the time units attribute. It then
    saves that attribute in `ds.time.encoding["units"]`.

    :param ds: xarray.Dataset
    :file_list: list of file paths
    """
    # Check that first file exists, if not return
    f1 = sorted(file_list)[0]

    if not os.path.isfile(f1):
        return

    # If time is present and the multi-file dataset has an empty `encoding` dictionary.
    # Open the first file to get the time units and add into encoding dictionary.
    if hasattr(ds, "time") and not ds.time.encoding.get("units"):
        ds1 = xr.open_dataset(f1, **kwargs)
        ds.time.encoding["units"] = ds1.time.encoding.get("units", "")


def _get_kwargs_for_opener(otype, **kwargs):
    """
    Returns a dictionary of keyword args for sending to either `xr.open_dataset()`
    of `xr.open_mfdataset()`, based on whether otype="single" or "multi".
    The provided `kwargs` dictionary is used to extend/override the default
    values.

    :param otype: (Str) type of opener (either "single" or "multi")
    :param kwargs: Any further keyword arguments to include when opening the dataset.
    """
    args = {"use_cftime": True, "decode_timedelta": False}

    if otype.lower().startswith("multi"):
        args["combine"] = "by_coords"

    args.update(kwargs)

    # If single file opener, then remove any multifile args that would raise an
    # exception when called
    if otype.lower() == "single":
        [
            args.pop(arg)
            for arg in list(args)
            if arg not in inspect.getfullargspec(xr.open_dataset).kwonlyargs
        ]

    return args


def is_kerchunk_file(dset):
    """
    Returns a boolean based on reading the file extension.
    """
    if not isinstance(dset, str):
        return False

    return os.path.splitext(dset)[-1] in KERCHUNK_EXTS


def _open_as_kerchunk(dset, **kwargs):
    """
    Open the dataset `dset` as a Kerchunk file. Return an Xarray Dataset.
    """
    compression = (
        "zstd"
        if dset.split(".")[-1].startswith("zst")
        else kwargs.get("compression", None)
    )
    remote_options = kwargs.get("remote_options", {})
    remote_protocol = kwargs.get("remote_protocol", None)

    mapper = fsspec.get_mapper(
        "reference://",
        fo=dset,
        target_options={"compression": compression},
        remote_options=remote_options,
        remote_protocol=remote_protocol,
    )

    # Create a copy of kwargs and remove mapper-specific values
    kw = kwargs.copy()
    for key in ("compression", "remote_options", "remote_protocol"):
        if key in kw:
            del kw[key]

    return xr.open_zarr(mapper, consolidated=False, **kw)


def open_xr_dataset(dset, **kwargs):
    r"""
    Opens an xarray dataset from a dataset input.

    :param dset: (Str or Path) ds_id, directory path or file path ending in \*.nc.
    :param kwargs: Any further keyword arguments to include when opening the dataset.
                   use_cftime=True and decode_timedelta=False are used by default,
                   along with combine="by_coords" for open_mfdataset only.

    Any list will be interpreted as list of files
    """
    # Set up dictionaries of arguments to send to all `xr.open_*dataset()` calls
    single_file_kwargs = _get_kwargs_for_opener("single", **kwargs)
    multi_file_kwargs = _get_kwargs_for_opener("multi", **kwargs)

    # Assume that a JSON or ZST/ZSTD file is kerchunk
    if type(dset) not in (list, tuple):
        # Assume that a JSON or ZST/ZSTD file is kerchunk
        if is_kerchunk_file(dset):
            return _open_as_kerchunk(dset, **single_file_kwargs)

        else:
            # Force the value of dset to be a list if not a list or tuple
            # use force=True to allow all file paths to pass through DatasetMapper
            dset = dset_to_filepaths(dset, force=True)

    # If an empty sequence, then raise an Exception
    if len(dset) == 0:
        raise Exception("No files found to open with xarray.")

    # if a list we want a multi-file dataset
    if len(dset) > 1:
        ds = xr.open_mfdataset(dset, **multi_file_kwargs)
        # Ensure that time units are retained
        _patch_time_encoding(ds, dset, **single_file_kwargs)
        return ds

    # if there is only one file we only need to call open_dataset
    else:
        return xr.open_dataset(dset[0], **single_file_kwargs)


# from dachar
def get_coord_by_attr(ds, attr, value):
    """
    Returns a coordinate based on a known attribute of a coordinate.

    :param ds: Xarray Dataset or DataArray
    :param attr: (str) Name of attribute to look for.
    :param value: Expected value of attribute you are looking for.
    :return: Coordinate of xarray dataset if found.
    """
    coords = ds.coords

    for coord in coords.values():
        if coord.attrs.get(attr, None) == value:
            return coord

    return None


def is_latitude(coord):
    """
    Determines if a coordinate is latitude.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is latitude.
    """

    if "latitude" in coord.cf and coord.cf["latitude"].name == coord.name:
        return True

    if coord.attrs.get("standard_name", None) == "latitude":
        return True

    return False


def is_longitude(coord):
    """
    Determines if a coordinate is longitude.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is longitude.
    """
    if "longitude" in coord.cf and coord.cf["longitude"].name == coord.name:
        return True

    if coord.attrs.get("standard_name", None) == "longitude":
        return True

    return False


def is_level(coord):
    """
    Determines if a coordinate is level.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is level.
    """
    if "vertical" in coord.cf and coord.cf["vertical"].name == coord.name:
        return True

    if hasattr(coord, "positive"):
        if coord.attrs.get("positive", None) == "up" or "down":
            return True

    if hasattr(coord, "axis"):
        if coord.attrs.get("axis", None) == "Z":
            return True

    return False


def is_time(coord):
    """
    Determines if a coordinate is time.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is time.
    """
    if "time" in coord.cf and coord.cf["time"].name == coord.name:
        return True

    if np.issubdtype(coord.dtype, np.datetime64):
        return True

    if isinstance(np.atleast_1d(coord.values)[0], cftime.datetime):
        return True

    if hasattr(coord, "axis"):
        if coord.axis == "T":
            return True

    if coord.attrs.get("standard_name", None) == "time":
        return True

    return False


def is_realization(coord):
    """
    Determines if a coordinate is realization.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is longitude.
    """
    if "realization" in coord.cf and coord.cf["realization"].name == coord.name:
        return True

    if coord.attrs.get("standard_name", None) == "realization":
        return True

    return False


def get_coord_type(coord):
    """
    Gets the coordinate type.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: The type of coordinate as a string. Either longitude, latitude, time, level or None
    """

    if is_longitude(coord):
        return "longitude"
    elif is_latitude(coord):
        return "latitude"
    elif is_level(coord):
        return "level"
    elif is_time(coord):
        return "time"
    elif is_realization(coord):
        return "realization"

    return None


def convert_coord_to_axis(coord):
    """
    Converts coordinate type to its single character axis identifier (tzyx).

    :param coord: (str) The coordinate to convert.
    :return: (str) The single character axis identifier of the coordinate (tzyx).
    """

    axis_dict = {
        "time": "t",
        "longitude": "x",
        "latitude": "y",
        "level": "z",
        "realization": "r",
    }
    return axis_dict.get(coord, None)


def get_coord_by_type(ds, coord_type, ignore_aux_coords=True):
    """
    Returns the xarray Dataset or DataArray coordinate of the specified type.

    :param ds: xarray Dataset or DataArray
    :param coord_type: (str) Coordinate type to find.
    :param ignore_aux_coords: (bool) If True then coordinates that are not dimensions are ignored.
                            Default is True.
    :return: Xarray Dataset coordinate (ds.coords[coord_id])
    """
    if coord_type not in known_coord_types:
        raise Exception(f"Coordinate type not known: {coord_type}")

    for coord_id in ds.coords:
        # If ignore_aux_coords is True then ignore coords that are not dimensions
        if ignore_aux_coords and coord_id not in ds.dims:
            continue

        coord = ds.coords[coord_id]

        if get_coord_type(coord) == coord_type:
            return coord

    # add this in for if lat/lon have not been found yet (e.g. they are data variables)
    # will also find time if not yet found
    # only relevant when ignore_aux_coords=False
    if coord_type != "level" and ignore_aux_coords is False:
        try:
            coord = ds.cf[coord_type]
        except KeyError:
            coord = None
        return coord

    return None


def get_main_variable(ds, exclude_common_coords=True):
    """
    Finds the main variable of an xarray Dataset

    :param ds: xarray Dataset
    :param exclude_common_coords: (bool) If True then common coordinates are excluded from the search for the
                                main variable. common coordinates are time, level, latitude, longitude and bounds.
                                Default is True.
    :return: (str) The main variable of the dataset e.g. 'tas'
    """

    data_dims = [data.dims for var_id, data in ds.variables.items()]
    flat_dims = [dim for sublist in data_dims for dim in sublist]

    results = {}
    common_coords = [
        "bnd",
        "bound",
        "lat",
        "lon",
        "time",
        "level",
        "realization_index",
        "realization",
    ]

    for var_id, data in ds.variables.items():
        if var_id in flat_dims:
            continue
        if exclude_common_coords is True and any(
            coord in var_id for coord in common_coords
        ):
            continue
        else:
            results.update({var_id: len(ds[var_id].shape)})
    result = max(results, key=results.get)

    if result is None:
        raise Exception("Could not determine main variable")
    else:
        return result
