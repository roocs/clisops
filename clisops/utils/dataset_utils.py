import inspect
import os
import warnings
from typing import Optional

import cf_xarray as cfxr  # noqa
import cftime
import fsspec
import numpy as np
import xarray as xr

from clisops.exceptions import InvalidParameterValue
from clisops.project_utils import dset_to_filepaths
from clisops.utils.time_utils import str_to_AnyCalendarDateTime

known_coord_types = ["time", "level", "latitude", "longitude", "realization"]

KERCHUNK_EXTS = [".json", ".zst", ".zstd"]


def get_coord_by_type(
    ds, coord_type, ignore_aux_coords=True, return_further_matches=False
):
    """
    Returns the name of the coordinate that matches the given type.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset/DataArray to search for coordinate.
    coord_type : str
        Type of coordinate, e.g. 'time', 'level', 'latitude', 'longitude', 'realization'.
    ignore_aux_coords : bool
        Whether to ignore auxiliary coordinates.
    return_further_matches : bool
        Whether to return further matches.

    Returns
    -------
    str
        Name of the coordinate that matches the given type.
    str, list of str
        If return_further_matches is True, apart from the matching coordinate,
        a list with further potential matches is returned.

    Raises
    ------
    ValueError
        If the coordinate type is not known.
    """
    # List for all potential matches
    coords = list()

    # If coord_type is not in known_coord_types then raise an error
    if coord_type not in known_coord_types:
        raise ValueError(f"Coordinate type not known: {coord_type}")

    # Loop through all (potential) coordinates to find all possible matches
    if isinstance(ds, xr.DataArray):
        coord_vars = list(ds.coords)
    elif isinstance(ds, xr.Dataset):
        # Not all coordinate variables are always classified as such
        coord_vars = list(ds.coords) + list(ds.data_vars)
    else:
        raise TypeError("Not an xarray.Dataset or xarray.DataArray.")
    for coord_id in coord_vars:
        # If ignore_aux_coords is True then: ignore coords that are not dimensions
        if ignore_aux_coords and coord_id not in ds.dims:
            continue

        coord = ds[coord_id]

        if get_coord_type(coord) == coord_type:
            coords.append(coord_id)

    # Return None if no match
    if len(coords) == 0:
        warnings.warn(f"No coordinate variable found for type '{coord_type}'.")
        return None
    elif len(coords) == 1:
        if return_further_matches:
            return coords[0], []
        else:
            return coords[0]
    # If more than one match is found, a selection has to be made
    else:
        warnings.warn(
            f"More than one coordinate variable found for type '{coord_type}'. Selecting the best fit."
        )
        # Sort in terms of number of dimensions
        coords = sorted(coords, key=lambda x: len(ds[x].dims), reverse=True)

        # Get dimensions and singleton coords of main variable
        main_var_dims = list(ds[get_main_variable(ds)].dims)

        # Select coordinate with most dims (matching with main variable dims)
        for coord_id in coords:
            if all([dim in main_var_dims for dim in ds.coords[coord_id].dims]):
                if return_further_matches:
                    return coord_id, [x for x in coords if x != coord_id]
                else:
                    return coord_id
        # If the decision making fails, pass the first match
        if return_further_matches:
            return coords[0], coords[1:]
        else:
            return coords[0]


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

    if (
        "latitude" in coord.cf.coordinates
        and coord.name in coord.cf.coordinates["latitude"]
    ):
        return True

    if (
        "latitude" in coord.cf.standard_names
        and coord.name in coord.cf.standard_names["latitude"]
    ):
        return True

    if hasattr(coord, "standard_name") and coord.standard_name == "latitude":
        return True

    if hasattr(coord, "long_name") and coord.long_name == "latitude":
        return True

    return False


def is_longitude(coord):
    """
    Determines if a coordinate is longitude.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is longitude.
    """
    if (
        "longitude" in coord.cf.coordinates
        and coord.name in coord.cf.coordinates["longitude"]
    ):
        return True

    if (
        "longitude" in coord.cf.standard_names
        and coord.name in coord.cf.standard_names["longitude"]
    ):
        return True

    if hasattr(coord, "standard_name") and coord.standard_name == "longitude":
        return True

    if hasattr(coord, "long_name") and coord.long_name == "longitude":
        return True

    return False


def is_level(coord):
    """
    Determines if a coordinate is level.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is level.
    """
    if (
        "vertical" in coord.cf.coordinates
        and coord.name in coord.cf.coordinates["vertical"]
    ):
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
    if "time" in coord.cf.coordinates and coord.name in coord.cf.coordinates["time"]:
        return True

    if (
        "time" in coord.cf.standard_names
        and coord.name in coord.cf.standard_names["time"]
    ):
        return True

    if np.issubdtype(coord.dtype, np.datetime64):
        return True

    if isinstance(np.atleast_1d(coord.values)[0], cftime.datetime):
        return True

    if hasattr(coord, "axis"):
        if coord.axis == "T":
            return True

    return False


def is_realization(coord):
    """
    Determines if a coordinate is realization.

    :param coord: coordinate of xarray dataset e.g. coord = ds.coords[coord_id]
    :return: (bool) True if the coordinate is longitude.
    """
    if (
        "realization" in coord.cf.standard_names
        and coord.name in coord.cf.standard_names["realization"]
    ):
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


def open_xr_dataset(dset, **kwargs):
    """
    Opens an xarray dataset from a dataset input.

    :param dset: (str or Path) A dataset identifier, directory path, or file path ending in ``*.nc``.
    :param kwargs: Any additional keyword arguments for opening the dataset.
                   ``use_cftime=True`` and ``decode_timedelta=False`` are used by default,
                   along with ``combine="by_coords"`` for ``open_mfdataset`` only.

    Notes:
        - Any list will be interpreted as a list of files.
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


def determine_lon_lat_range(ds, lon, lat, lon_bnds=None, lat_bnds=None, apply_fix=True):
    """Determine the min/max lon/lat values of the dataset (and potentially apply fix for unmasked missing values).

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    lon : str
        Name of longitude coordinate.
    lat : str
        Name of latitude coordinate.
    lon_bnds : str or None, optional
        Name of longitude bounds coordinate. The default is None.
    lat_bnds : str or None, optional
        Name of latitude bounds coordinate. The default is None.
    apply_fix : bool, optional
        Whether to apply fix for unmasked missing values. The default is True.

    Returns
    -------
    xmin : float
        Minimum longitude value.
    xmax : float
        Maximum longitude value.
    ymin : float
        Minimum latitude value.
    ymax : float
        Maximum latitude value.
    """
    # Determine min/max lon/lat values
    xmin = ds[lon].min().item()
    xmax = ds[lon].max().item()
    ymin = ds[lat].min().item()
    ymax = ds[lat].max().item()

    # Potentially apply fix for unmasked missing values
    if apply_fix:
        if fix_unmasked_missing_values_lon_lat(
            ds, lon, lat, lon_bnds, lat_bnds, [xmin, xmax], [ymin, ymax]
        ):
            xmin = ds[lon].min().item()
            xmax = ds[lon].max().item()
            ymin = ds[lat].min().item()
            ymax = ds[lat].max().item()

    return xmin, xmax, ymin, ymax


def fix_unmasked_missing_values_lon_lat(
    ds, lon, lat, lon_bnds, lat_bnds, xminmax, yminmax
):
    """Fix for unmasked missing values in longitude and latitude coordinates and their bounds

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    lon : str
        Name of longitude coordinate.
    lat : str
        Name of latitude coordinate.
    lon_bnds : str or None
        Name of longitude bounds coordinate.
    lat_bnds : str or None
        Name of latitude bounds coordinate.
    xminmax : list
        List of minimum and maximum longitude values.
    yminmax : list
        List of minimum and maximum latitude values.

    Returns
    -------
    fix : bool
        Whether the fix on ds[lon] and ds[lat] (and if specified ds[lon_bnds] and ds[lat_bnds]) was applied or not.
    """
    fix = False
    minval = -999
    maxval = 999

    # Potentially fix unmasked missing values in longitude/latitude arrays
    if any([xymin <= minval for xymin in xminmax + yminmax]) or any(
        [xymax >= maxval for xymax in xminmax + yminmax]
    ):

        # Identify potential missing values by detecting outliers
        mask_y = (ds[lat] <= minval) | (ds[lat] >= maxval)
        possible_missing_values_y = ds[lat].where(mask_y)
        mask_x = (ds[lon] <= minval) | (ds[lon] >= maxval)
        possible_missing_values_x = ds[lon].where(mask_x)

        # TBD - potential TODO - Explicitly check the vertices as well for possible missing values
        #                        and not apply the mask from lat / lon.
        #                      - Check if the fields already contain nans (and if they are consistent
        #                        between lat and lon).

        # Find out if the outlier values are unique
        possible_missing_values_x_min = possible_missing_values_x.min().item()
        possible_missing_values_x_max = possible_missing_values_x.max().item()
        possible_missing_values_y_min = possible_missing_values_y.min().item()
        possible_missing_values_y_max = possible_missing_values_y.max().item()

        possible_missing_values = [
            val
            for val in [
                possible_missing_values_x_min,
                possible_missing_values_x_max,
                possible_missing_values_y_min,
                possible_missing_values_y_max,
            ]
            if not np.isnan(val)
        ]

        # Compare the masks for lat / lon and abort the fix if they differ
        if (
            ds[lat].dims != ds[lon].dims
            and len(ds[lon].dims) == 1
            and len(ds[lat].dims) == 1
        ):
            # Abort fix for regular lat-lon grids (1D coordinate variables should not include missing values
            #  - for some of the operations the outliers will cause an exception later on)
            warnings.warn(
                f"Extreme value(s) (potentially unmasked missing_values) found in {lon} and {lat} arrays: "
                f"{set(possible_missing_values)}. A fix is not possible for regular latitude-longitude grids."
            )
            return fix
        elif not (mask_x == mask_y).all().item():
            # Abort fix if the masks differ
            warnings.warn(
                f"Extreme value(s) (potentially unmasked missing_values) found in {lon} and {lat} arrays: "
                f"{set(possible_missing_values)}. A fix is not possible since their locations are not consistent "
                "between the two arrays."
            )
            return fix

        # Check if there's only one unique extreme value
        if len(set(possible_missing_values)) == 1:
            fix = True
            missing_value = possible_missing_values[0]
            # Replace the missing value with np.NaN in place
            # and add _FillValue and missing_value attributes
            #  (ignoring already present attributes)
            for var in lat, lon:
                ds[var] = ds[var].where(ds[var] != missing_value, other=np.nan)
            if lat_bnds is not None and lon_bnds is not None:
                ds[lat_bnds] = ds[lat_bnds].where(
                    ds[lat] != missing_value, other=np.nan
                )
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon] != missing_value, other=np.nan
                )
            for var in [
                var for var in [lat, lon, lat_bnds, lon_bnds] if var is not None
            ]:
                ds[var].encoding["_FillValue"] = float(1e20)
                ds[var].encoding["missing_value"] = float(1e20)
                ds[var].attrs["_FillValue"] = float(1e20)
                ds[var].attrs["missing_value"] = float(1e20)
            warnings.warn(
                f"Unmasked missing_value found (and treated) in {lon} and {lat} arrays: '{missing_value}'."
            )
        else:
            # Raise warning - the values will likely cause an exception later on, depending on the operation
            warnings.warn(
                f"Multiple extreme values (potentially unmasked missing_values) found in {lon} and {lat} arrays: {set(possible_missing_values)}. This may cause issues."
            )

    return fix


def calculate_offset(lon, first_element_value):
    """Calculate the number of elements to roll the dataset by in order to have longitude from within requested bounds.

    Parameters
    ----------
    lon
        Longitude coordinate of xarray dataset.
    first_element_value
        The value of the first element of the longitude array to roll to.
    """
    # get resolution of data
    res = lon.values[1] - lon.values[0]

    # calculate how many degrees to move by to have lon[0] of rolled subset as lower bound of request
    diff = lon.values[0] - first_element_value

    # work out how many elements to roll by to roll data by 1 degree
    index = 1 / res

    # calculate the corresponding offset needed to change data by diff
    offset = int(round(diff * index))

    return offset


def _crosses_0_meridian(lon_c: xr.DataArray):
    """Determine whether grid extents over the 0-meridian.

    Assumes approximate constant width of grid cells.

    Parameters
    ----------
    lon_c: xr.DataArray
        Longitude coordinate variable in the longitude frame [-180, 180].

    Returns
    -------
    bool
        True for a dataset crossing the 0-meridian, False else.
    """
    if not isinstance(lon_c, xr.DataArray):
        raise InvalidParameterValue("Input needs to be of type xarray.DataArray.")

    # Not crossing the 0-meridian if all values are positive or negative
    lon_n = lon_c.where(lon_c <= 0, 0)
    lon_p = lon_c.where(lon_c >= 0, 0)
    if lon_n.all() or lon_p.all():
        return False

    # Determine min/max lon values
    xc_min = float(lon_c.min())
    xc_max = float(lon_c.max())

    # Determine resolution in zonal direction
    if lon_c.ndim == 1:
        xc_inc = (xc_max - xc_min) / (lon_c.sizes[lon_c.dims[0]] - 1)
    else:
        xc_inc = (xc_max - xc_min) / (lon_c.sizes[lon_c.dims[1]] - 1)

    # Generate a histogram with bins for sections along a latitudinal circle,
    #  width of the bins/sections dependent on the resolution in x-direction
    atol = 2.0 * xc_inc
    extent_hist = np.histogram(
        lon_c,
        bins=np.arange(xc_min - xc_inc, xc_max + atol, atol),
    )

    # If the counts for all bins are greater than zero, the grid is considered crossing the 0-meridian
    if np.all(extent_hist[0]):
        return True
    else:
        return False


def _convert_interval_between_lon_frames(low, high):
    """Convert a longitude interval to another longitude frame, returns Tuple of two floats."""
    diff = high - low
    if low < 0 and high > 0:
        raise ValueError(
            "Cannot convert longitude interval if it includes the 0°- or 180°-meridian."
        )
    elif low < 0:
        return tuple(sorted((low + 360.0, low + 360.0 + diff)))
    elif low < 180 and high > 180:
        raise ValueError(
            "Cannot convert longitude interval if it includes the 0°- or 180°-meridian."
        )
    elif high > 180:
        return tuple(sorted((high - 360.0 - diff, high - 360.0)))
    else:
        return float(low), float(high)


def cf_convert_between_lon_frames(ds_in, lon_interval, force=False):
    """Convert ds or lon_interval (whichever deems appropriate) to the other longitude frame, if the longitude frames do not match.

    If ds and lon_interval are defined on different longitude frames ([-180, 180] and [0, 360]),
    this function will convert one of the input parameters to the other longitude frame, preferably
    the lon_interval.
    Adjusts shifted longitude frames [0-x, 360-x] in the dataset to one of the two standard longitude
    frames, dependent on the specified lon_interval.
    In case of curvilinear grids featuring an additional 1D x-coordinate of the projection,
    this projection x-coordinate will not get converted.

    Parameters
    ----------
    ds_in: xarray.Dataset or xarray.DataArray
        xarray data object with defined longitude dimension.
    lon_interval: tuple or list
        length-2-tuple or -list of floats or integers denoting the bounds of the longitude interval.
    force: bool
        If True, force conversion even if longitude frames match.

    Returns
    -------
    Tuple(ds, lon_low, lon_high)
        The xarray.Dataset and the bounds of the longitude interval, potentially adjusted in terms
        of their defined longitude frame.
    """
    # Collect input specs
    if isinstance(ds_in, (xr.Dataset, xr.DataArray)):
        lon = detect_coordinate(ds_in, "longitude")
        lat = detect_coordinate(ds_in, "latitude")
        lon_bnds = detect_bounds(ds_in, lon)
        # lat_bnds = detect_bounds(ds_in, lat)
        # Do not consider bounds in gridtype detection (yet fails due to open_mfdataset bug that adds
        #  time dimension to bounds - todo)
        gridtype = detect_gridtype(
            ds_in, lon=lon, lat=lat
        )  # lat_bnds=lat_bnds, lon_bnds = lon_bnds)
        ds = ds_in.copy()
    else:
        raise InvalidParameterValue(
            "This function requires an xarray.DataArray or xarray.Dataset as input."
        )
    low, high = lon_interval
    lon_min, lon_max = ds.coords[lon].min().item(), ds.coords[lon].max().item()
    atol = 0.5

    # Conversion between longitude frames if required
    if (lon_min <= low or np.isclose(low, lon_min, atol=atol)) and (
        lon_max >= high or np.isclose(high, lon_max, atol=atol)
    ):
        if not force:
            return ds, low, high

    # Check longitude
    # For longitude frames other than [-180, 180] and [0, 360] in the dataset the following assumptions
    #  are being made:
    # - fixpoint is the 0-meridian
    # - the lon_interval is either defined in the longitude frame [-180, 180] or [0, 360]
    # TODO: possibly sth like
    #  while lon_min < -180, lon[lon<-180]=lon[lon<-180]+360
    #  while lon_max > 360, lon[lon>360]=lon[lon>360]-360
    if lon_max - lon_min > 360 + atol or lon_min < -360 - atol or lon_max > 360 + atol:
        raise ValueError(
            "The longitude coordinate values have to lie within the interval "
            "[-360, 360] degrees and not exceed an extent of 360 degrees."
        )

    # Conversion: longitude is a singleton dimension
    elif (ds[lon].ndim == 1 and ds.sizes[ds[lon].dims[0]] == 1) or (
        ds[lon].ndim > 1 and ds.sizes[ds[lon].dims[1]] == 1
    ):
        if low < 0 and lon_min > 0:
            ds[lon] = ds[lon].where(ds[lon] <= 180, ds[lon] - 360.0)
            if lon_bnds:
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] <= 180, ds[lon_bnds] - 360.0
                )
        elif low > 0 and lon_min < 0:
            ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
            if lon_bnds:
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                )
        return ds, low, high

    # Conversion: 1D or 2D longitude coordinate variable
    else:
        # regional [0 ... 180]
        if lon_min >= 0 and lon_max <= 180:
            return ds, low, high

        # shifted frame beyond -180, eg. [-300, 60]
        elif lon_min < -180 - atol:
            if low < 0:
                ds[lon] = ds[lon].where(ds[lon] > -180, ds[lon] + 360.0)
                if lon_bnds:
                    ds[lon_bnds] = ds[lon_bnds].where(
                        ds[lon_bnds] > -180, ds[lon_bnds] + 360.0
                    )
            elif low >= 0:
                ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
                if lon_bnds:
                    ds[lon_bnds] = ds[lon_bnds].where(
                        ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                    )

        # shifted frame beyond 0, eg. [-60, 300]
        elif lon_min < -atol and lon_max > 180 + atol:
            if low < 0:
                ds[lon] = ds[lon].where(ds[lon] <= 180, ds[lon] - 360.0)
                if lon_bnds:
                    ds[lon_bnds] = ds[lon_bnds].where(
                        ds[lon_bnds] <= 180, ds[lon_bnds] - 360.0
                    )
            elif low >= 0:
                ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
                if lon_bnds:
                    ds[lon_bnds] = ds[lon_bnds].where(
                        ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                    )

        # [-180 ... 180]
        elif lon_min < 0:
            # interval includes 180°-meridian: convert dataset to [0, 360]
            if low < 180 and high > 180:
                ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
                if lon_bnds:
                    ds[lon_bnds] = ds[lon_bnds].where(
                        ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                    )
            # interval does not include 180°-meridian: convert interval to [-180,180]
            else:
                if low >= 0:
                    if not force:
                        low, high = _convert_interval_between_lon_frames(low, high)
                    else:
                        ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
                        if lon_bnds:
                            ds[lon_bnds] = ds[lon_bnds].where(
                                ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                            )
                return ds, low, high

        # [0 ... 360]
        else:
            # interval positive, return unchanged
            if low >= 0:
                return ds, low, high
            # interval includes 0°-meridian: convert dataset to [-180, 180]
            elif high > 0:
                ds[lon] = ds[lon].where(ds[lon] <= 180, ds[lon] - 360.0)
                if lon_bnds:
                    ds[lon_bnds] = ds[lon_bnds].where(
                        ds[lon_bnds] <= 180, ds[lon_bnds] - 360.0
                    )
            # interval negative
            else:
                if not force:
                    low, high = _convert_interval_between_lon_frames(low, high)
                    return ds, low, high
                else:
                    ds[lon] = ds[lon].where(ds[lon] <= 180, ds[lon] - 360.0)
                    if lon_bnds:
                        ds[lon_bnds] = ds[lon_bnds].where(
                            ds[lon_bnds] <= 180, ds[lon_bnds] - 360.0
                        )
        # 1D coordinate variable: Sort, since order might no longer be ascending / descending
        if gridtype == "regular_lat_lon":
            ds = ds.sortby(lon)

        return ds, low, high


def check_lon_alignment(ds, lon_bnds):
    """Check whether the longitude subset requested is within the bounds of the dataset.

    If not try to roll the dataset so that the request is. Raise an exception if rolling is not possible.
    """
    low, high = lon_bnds
    lon = get_coord_by_type(ds, "longitude", ignore_aux_coords=False)
    lon = ds.coords[lon]
    lon_min, lon_max = lon.values.min(), lon.values.max()

    # handle the case where there is only one longitude
    if len(lon.values) == 1:
        lon_value = ds.lon.values[0]
        if low > lon_value:
            ds.coords[lon.name] = ds.coords[lon.name] + 360
        elif high < lon_value:
            ds.coords[lon.name] = ds.coords[lon.name] - 360
        return ds

    # check if the request is in bounds - return ds if it is
    if (lon_min <= low or np.isclose(low, lon_min, atol=0.5)) and (
        lon_max >= high or np.isclose(high, lon_max, atol=0.5)
    ):
        return ds

    else:
        # check if lon is a dimension
        if lon.name not in ds.dims:
            raise Exception(
                f"The requested longitude subset {lon_bnds} is not within the longitude bounds "
                f"of this dataset and the data could not be converted to this longitude frame successfully. "
                f"Please re-run your request with longitudes within the bounds of the dataset: ({lon_min:.2f}, {lon_max:.2f})"
            )
        # roll the dataset and reassign the longitude values
        else:
            first_element_value = low
            offset = calculate_offset(lon, first_element_value)

            # roll the dataset
            ds_roll = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)

            # assign longitude to match the roll and copy attrs
            lon_vals = ds_roll.coords[lon.name].values

            # treat the values differently according to positive/negative offset
            if offset < 0:
                lon_vals[offset:] = lon_vals[offset:] % 360
            else:
                lon_vals[:offset] = lon_vals[:offset] % -360

            ds_roll.coords[lon.name] = lon_vals
            ds_roll.coords[lon.name].attrs = ds.coords[lon.name].attrs
            return ds_roll


def adjust_date_to_calendar(da, date, direction="backwards"):
    """Check that the date specified exists in the calendar type of the dataset.

    If not present, changes the date a day at a time (up to a maximum of 5 times) to find a date that does exist.
    The direction to change the date by is indicated by 'direction'.

    Parameters
    ----------
    da : xarray.Dataset or xarray.DataArray
        The data to examine.
    date : str
        The date to check.
    direction : str
        The direction to move in days to find a date that does exist.
        'backwards' means the search will go backwards in time until an existing date is found.
        'forwards' means the search will go forwards in time.
        The default is 'backwards'.

    Returns
    -------
    str
        The next possible existing date in the calendar of the dataset.
    """
    # turn date into AnyCalendarDateTime object
    d = str_to_AnyCalendarDateTime(date)

    # get the calendar type
    cal = da.cf["time"].data[0].calendar

    for i in range(5):
        try:
            cftime.datetime(
                d.year,
                d.month,
                d.day,
                d.hour,
                d.minute,
                d.second,
                calendar=cal,
            )
            return d.value
        except ValueError:
            if direction == "forwards":
                d.add_day()
            elif direction == "backwards":
                d.sub_day()
            else:
                raise Exception(
                    f"Invalid value for direction: {direction}. This should be either 'backwards' to indicate subtracting a day or 'forwards' for adding a day."
                )

    raise ValueError(
        f"Could not find an existing date near {date} in the calendar: {cal}"
    )


def add_hor_CF_coord_attrs(
    ds, lat="lat", lon="lon", lat_bnds="lat_bnds", lon_bnds="lon_bnds", keep_attrs=False
):
    """
    Add the common CF variable attributes to the horizontal coordinate variables.

    Parameters
    ----------
    lat : str, optional
        Latitude coordinate variable name. The default is "lat".
    lon : str, optional
        Longitude coordinate variable name. The default is "lon".
    lat_bnds : str, optional
        Latitude bounds coordinate variable name. The default is "lat_bnds".
    lon_bnds : str, optional
        Longitude bounds coordinate variable name. The default is "lon_bnds".
    keep_attrs : bool, optional
        Whether to keep original coordinate variable attributes if they do not conflict.
        In case of a conflict, the attribute value will be overwritten independent of this setting.
        The default is False.

    Returns
    -------
    xarray.Dataset
        The input dataset with updated coordinate variable attributes.
    """
    # Define common CF coordinate variable attrs
    lat_attrs = {
        "bounds": lat_bnds,
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
        "axis": "Y",
    }
    lon_attrs = {
        "bounds": lon_bnds,
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude",
        "axis": "X",
    }

    # Overwrite or update coordinate variables of input dataset
    try:
        if keep_attrs:
            ds[lat].attrs.update(lat_attrs)
            ds[lon].attrs.update(lon_attrs)
        else:
            ds[lat].attrs = lat_attrs
            ds[lon].attrs = lon_attrs
            ds[lat_bnds].attrs = {}
            ds[lon_bnds].attrs = {}
    except KeyError:
        raise KeyError("Not all specified coordinate variables exist in the dataset.")

    return ds


def reformat_SCRIP_to_CF(ds, keep_attrs=False):
    """Reformat dataset from SCRIP to CF format.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset in SCRIP format.
    keep_attrs: bool
        Whether to keep the global attributes.

    Returns
    -------
    ds_ref : xarray.Dataset
        Reformatted dataset.
    """
    source_format = "SCRIP"
    target_format = "CF"
    SCRIP_vars = [
        "grid_center_lat",
        "grid_center_lon",
        "grid_corner_lat",
        "grid_corner_lon",
        "grid_dims",
        "grid_area",
        "grid_imask",
    ]

    if not isinstance(ds, xr.Dataset):
        raise InvalidParameterValue(
            "Reformat is only possible for Datasets."
            " DataArrays have to be CF conformal coordinate variables defined."
        )

    # Cannot reformat data variables yet
    if not (
        all([var in SCRIP_vars for var in ds.data_vars])
        and all([coord in SCRIP_vars for coord in ds.coords])
    ):
        raise Exception(
            "Converting the grid format from %s to %s is not yet possible for data variables."
            % (source_format, target_format)
        )

    # center lat and lon arrays will become the lat and lon arrays
    lat = ds.grid_center_lat.values.reshape(
        (ds.grid_dims.values[1], ds.grid_dims.values[0])
    ).astype(np.float32)
    lon = ds.grid_center_lon.values.reshape(
        (ds.grid_dims.values[1], ds.grid_dims.values[0])
    ).astype(np.float32)

    # corner coordinates will become lat_bnds and lon_bnds arrays
    # regular lat-lon case
    # todo: bounds of curvilinear case
    if all(
        [
            np.array_equal(lat[:, i], lat[:, i + 1], equal_nan=True)
            for i in range(ds.grid_dims.values[0] - 1)
        ]
    ) and all(
        [
            np.array_equal(lon[i, :], lon[i + 1, :], equal_nan=True)
            for i in range(ds.grid_dims.values[1] - 1)
        ]
    ):
        # regular lat-lon grid:
        # - 1D coordinate variables
        lat = lat[:, 0]
        lon = lon[0, :]
        # - reshape vertices from (n,2) to (n+1) for lat and lon axes
        lat_b = ds.grid_corner_lat.values.reshape(
            (
                ds.grid_dims.values[1],
                ds.grid_dims.values[0],
                ds.sizes["grid_corners"],
            )
        ).astype(np.float32)
        lon_b = ds.grid_corner_lon.values.reshape(
            (
                ds.grid_dims.values[1],
                ds.grid_dims.values[0],
                ds.sizes["grid_corners"],
            )
        ).astype(np.float32)
        lat_bnds = np.zeros((ds.grid_dims.values[1], 2), dtype=np.float32)
        lon_bnds = np.zeros((ds.grid_dims.values[0], 2), dtype=np.float32)
        lat_bnds[:, 0] = np.min(lat_b[:, 0, :], axis=1)
        lat_bnds[:, 1] = np.max(lat_b[:, 0, :], axis=1)
        lon_bnds[:, 0] = np.min(lon_b[0, :, :], axis=1)
        lon_bnds[:, 1] = np.max(lon_b[0, :, :], axis=1)
        ds_ref = xr.Dataset(
            data_vars={},
            coords={
                "lat": (["lat"], lat),
                "lon": (["lon"], lon),
                "lat_bnds": (["lat", "bnds"], lat_bnds),
                "lon_bnds": (["lon", "bnds"], lon_bnds),
            },
        )
        # todo: Case of other units (rad)
        # todo: Reformat data variables if in ds, apply imask on data variables
        # todo: vertical axis, time axis, ... ?!

        # add common coordinate variable attrs
        ds_ref = add_hor_CF_coord_attrs(ds=ds_ref)

        # transfer global attributes
        if keep_attrs:
            ds_ref.attrs.update(ds.attrs)

        return ds_ref
    else:
        raise Exception(
            "Converting the grid format from %s to %s is yet only possible for regular latitude longitude grids."
            % (source_format, target_format)
        )


def reformat_xESMF_to_CF(ds, keep_attrs=False):
    """Reformat dataset from xESMF to CF format.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset in xESMF format.
    keep_attrs: bool
        Whether to keep the global attributes.

    Returns
    -------
    ds_ref : xarray.Dataset
        Reformatted dataset.
    """
    # source_format="xESMF"
    # target_format="CF"
    # todo: Check if it is regular_lat_lon, Check dimension sizes
    # Define lat, lon, lat_bnds, lon_bnds
    lat = ds.lat[:, 0]
    lon = ds.lon[0, :]
    lat_bnds = np.zeros((lat.shape[0], 2), dtype=np.float32)
    lon_bnds = np.zeros((lon.shape[0], 2), dtype=np.float32)

    # From (N+1, M+1) shaped bounds to (N, M, 4) shaped vertices
    lat_vertices = cfxr.vertices_to_bounds(ds.lat_b, ("bnds", "lat", "lon")).values
    lon_vertices = cfxr.vertices_to_bounds(ds.lon_b, ("bnds", "lat", "lon")).values

    # No longer necessary as of cf_xarray v0.7.5
    # lat_vertices = np.moveaxis(lat_vertices, 0, -1)
    # lon_vertices = np.moveaxis(lon_vertices, 0, -1)

    # From (N, M, 4) shaped vertices to (N, 2)  and (M, 2) shaped bounds
    lat_bnds[:, 0] = np.min(lat_vertices[:, 0, :], axis=1)
    lat_bnds[:, 1] = np.max(lat_vertices[:, 0, :], axis=1)
    lon_bnds[:, 0] = np.min(lon_vertices[0, :, :], axis=1)
    lon_bnds[:, 1] = np.max(lon_vertices[0, :, :], axis=1)

    # Create dataset
    ds_ref = xr.Dataset(
        data_vars={},
        coords={
            "lat": (["lat"], lat.data),
            "lon": (["lon"], lon.data),
            "lat_bnds": (["lat", "bnds"], lat_bnds.data),
            "lon_bnds": (["lon", "bnds"], lon_bnds.data),
        },
    )

    # todo: Case of other units (rad)
    # todo: Reformat data variables if in ds, apply imask on data variables
    # todo: vertical axis, time axis, ... ?!

    # add common coordinate variable attrs
    ds_ref = add_hor_CF_coord_attrs(ds=ds_ref)

    # transfer global attributes
    if keep_attrs:
        ds_ref.attrs.update(ds.attrs)

    return ds_ref
    #        else:
    #            raise Exception(
    #                "Converting the grid format from %s to %s is yet only possible for regular latitude longitude grids."
    #                % (self.format, grid_format)
    #            )


def detect_format(ds):
    """Detect format of a dataset. Yet supported are 'CF', 'SCRIP', 'xESMF'.

    Parameters
    ----------
    ds : xr.Dataset
        xarray.Dataset of which to detect the format.

    Returns
    -------
    str
        The format, if supported. Else raises an Exception.
    """
    # todo: extend for formats CF, xESMF, ESMF, UGRID, SCRIP
    # todo: add more conditions (dimension sizes, ...)
    SCRIP_vars = [
        "grid_center_lat",
        "grid_center_lon",
        "grid_corner_lat",
        "grid_corner_lon",
        # "grid_imask", "grid_area"
    ]
    SCRIP_dims = ["grid_corners", "grid_size", "grid_rank"]

    xESMF_vars = [
        "lat",
        "lon",
        "lat_b",
        "lon_b",
        # "mask",
    ]
    xESMF_dims = ["x", "y", "x_b", "y_b"]

    # Test if SCRIP
    if all([var in ds for var in SCRIP_vars]) and all(
        [dim in ds.dims for dim in SCRIP_dims]
    ):
        return "SCRIP"

    # Test if xESMF
    elif all([var in ds.coords for var in xESMF_vars]) and all(
        [dim in ds.dims for dim in xESMF_dims]
    ):
        return "xESMF"

    # Test if latitude and longitude can be found - standard_names would be set later if undef.
    elif (
        "latitude" in ds.cf.standard_names and "longitude" in ds.cf.standard_names
    ) or (
        get_coord_by_type(ds, "latitude", ignore_aux_coords=False) is not None
        and get_coord_by_type(ds, "longitude", ignore_aux_coords=False) is not None
    ):
        return "CF"

    else:
        raise Exception("The grid format is not supported.")


def detect_shape(ds, lat, lon, grid_type) -> tuple[int, int, int]:
    """Detect the shape of the grid.

    Returns a tuple of (nlat, nlon, ncells). For an unstructured grid nlat and nlon are not defined
    and therefore the returned tuple will be (ncells, ncells, ncells).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the grid / coordinate variables.
    lat : str
        Latitude variable name.
    lon : str
        Longitude variable name.
    grid_type: str
        One of "regular_lat_lon", "curvilinear", "unstructured"

    Returns
    -------
    int
        Number of latitude points in the grid.
    int
        Number of longitude points in the grid.
    int
        Number of cells in the grid.
    """
    if grid_type not in ["regular_lat_lon", "curvilinear", "unstructured"]:
        raise Exception(f"The specified grid_type '{grid_type}' is not supported.")

    if ds[lon].ndim != ds[lat].ndim:
        raise Exception(
            f"The coordinate variables {lat} and {lon} do not have the same number of dimensions."
        )
    elif ds[lat].ndim == 2:
        nlat = ds[lat].shape[0]
        nlon = ds[lat].shape[1]
        ncells = nlat * nlon
    elif ds[lat].ndim == 1:
        if ds[lat].shape == ds[lon].shape and grid_type == "unstructured":
            nlat = ds[lat].shape[0]
            nlon = nlat
            ncells = nlat
        else:
            nlat = ds[lat].shape[0]
            nlon = ds[lon].shape[0]
            ncells = nlat * nlon
    else:
        raise Exception(
            f"The coordinate variables {lat} and {lon} are not 1- or 2-dimensional."
        )
    return nlat, nlon, ncells


def _lonbnds_mids_trans_check(lon1, lon2, lon3, lon4):
    """Checks if the midpoints of the bounds traverse the Greenwich Meridian or
    antimeridian.If so, the midpoints are adjusted.
    """
    arr = np.array([lon1, lon2, lon3, lon4])
    diff = abs(arr.max() - arr.min())
    if diff > 180:
        # print("---------")
        # print(arr)
        arr = np.where(arr < 0.0, arr + 360.0, arr)
        # print(arr)
    mn = arr.mean()
    # print(mn)
    if mn > 180.0:
        mn = mn - 360.0
    # print(mn)
    return mn


def _lonbnds_mids_trans_check_diff(lon1, lon2):
    """Checks if the midpoints of the bounds traverse the Greenwich Meridian or
    antimeridian.If so, the midpoints are adjusted.
    """
    arr = np.array([lon1, lon2])
    if abs(arr[0] - arr[1]) > 180:
        arr = np.where(arr < 0.0, arr + 360.0, arr)
    val = arr[0] - (arr[1] - arr[0])
    if val > 180.0:
        val = val - 360.0
    return val


def _lonbnds_mids_trans_check_sum(lon1, lon2):
    """Checks if the midpoints of the bounds traverse the Greenwich Meridian or
    antimeridian.If so, the midpoints are adjusted.
    """
    arr = np.array([lon1, lon2])
    if abs(arr[0] - arr[1]) > 180:
        arr = np.where(arr < 0.0, arr + 360.0, arr)
    val = arr[0] + (arr[0] - arr[1])
    if val > 180.0:
        val = val - 360.0
    return val


def _determine_grid_orientation(lon):
    """Determine grid orientation by checking the longitude range along each axis."""
    # Compute the range of longitude values along both axes
    lon_range_axis_0 = abs(lon.max(axis=0) - lon.min(axis=0)).mean().item()
    lon_range_axis_1 = abs(lon.max(axis=1) - lon.min(axis=1)).mean().item()
    # print(lon_range_axis_0, lon_range_axis_1)

    if lon_range_axis_1 > lon_range_axis_0:
        return "nlat_nlon"  # Axis 1 corresponds to longitude (nlat, nlon)
    else:
        return "nlon_nlat"  # Axis 0 corresponds to longitude (nlon, nlat)


def generate_bounds_curvilinear(ds, lat, lon, clip_latitude=True, roll=True):
    """Compute bounds for curvilinear grids.

    Assumes 2D latitude and longitude coordinate variables. The bounds will be attached as coords
    to the xarray.Dataset. Assumes the longitudes are defined on the longitude frame [-180, 180].
    The default setting for 'roll' ensures that the longitudes
    are converted if this is not the case.

    The bound calculation for curvilinear grids was adapted from
    https://github.com/SantanderMetGroup/ATLAS/blob/mai-devel/scripts/ATLAS-data/\
    bash-interpolation-scripts/AtlasCDOremappeR_CORDEX/grid_bounds_calc.py
    which based on work by Caillaud Cécile and Samuel Somot from Meteo-France.
    Compared with the original code, there is an additional correction performed in the calculation,
    ensuring that at the Greenwich meridian and anti meridian the sign of the bounds does not differ.
    The new implementation is also significantly faster, as it replaces for loops with numpy.vectorize
    and index slicing.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to compute the bounds for.
    lon : str
        Longitude variable name.
    lat : str
        Latitude variable name.
    clip_latitude : bool, optional
        Whether to clip latitude values to [-90, 90]. The default is True.
    roll : bool, optional
        Whether to roll longitude values to [-180, 180]. The default is True.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with bounds attached variables.
    """
    # Assume lon frame -180, 180
    if roll:
        ds, lonmin, lonmax = cf_convert_between_lon_frames(ds, (-180, 180), force=True)
    assert lonmin == -180 and lonmax == 180

    # Detect shape
    nlat, nlon, ncells = detect_shape(ds=ds, lat=lat, lon=lon, grid_type="curvilinear")

    lats = ds[lat].values
    lons = ds[lon].values

    orientation = _determine_grid_orientation(ds[lon])
    if orientation == "nlat_nlon":
        londim = 1
    else:
        nlon, nlat = nlat, nlon
        londim = 0
    # print(orientation)
    if londim == 0:
        lons_crnr = np.full((nlon + 1, nlat + 1), np.nan)
        lats_crnr = np.full((nlon + 1, nlat + 1), np.nan)
    else:
        lons_crnr = np.full((nlat + 1, nlon + 1), np.nan)
        lats_crnr = np.full((nlat + 1, nlon + 1), np.nan)

    if londim == 1 or londim == 0:
        lats_crnr[1:-1, 1:-1] = (
            lats[:-1, :-1] + lats[1:, :-1] + lats[:-1, 1:] + lats[1:, 1:]
        ) / 4.0
        lons_crnr[1:-1, 1:-1] = np.vectorize(
            lambda x1, x2, x3, x4: _lonbnds_mids_trans_check(x1, x2, x3, x4)
        )(lons[:-1, :-1], lons[1:, :-1], lons[:-1, 1:], lons[1:, 1:])

    # print(lons_crnr)

    # Grid points at boundaries - incl correction for cells crossing the prime/anti meridian
    lons_crnr[0, :] = np.vectorize(
        lambda x1, x2: _lonbnds_mids_trans_check_diff(x1, x2)
    )(lons_crnr[1, :], lons_crnr[2, :])
    # lons_crnr[1, :] - (lons_crnr[2, :] - lons_crnr[1, :])
    lons_crnr[-1, :] = np.vectorize(
        lambda x1, x2: _lonbnds_mids_trans_check_sum(x1, x2)
    )(lons_crnr[-2, :], lons_crnr[-3, :])
    # lons_crnr[-2, :] + (lons_crnr[-2, :] - lons_crnr[-3, :])
    lons_crnr[:, 0] = np.vectorize(
        lambda x1, x2: _lonbnds_mids_trans_check_diff(x1, x2)
    )(lons_crnr[:, 1], lons_crnr[:, 2])
    # lons_crnr[:, 1] - (lons_crnr[:, 2] - lons_crnr[:, 1])
    lons_crnr[:, -1] = np.vectorize(
        lambda x1, x2: _lonbnds_mids_trans_check_sum(x1, x2)
    )(lons_crnr[:, -2], lons_crnr[:, -3])
    # lons_crnr[:, -2] + (lons_crnr[:, -2] - lons_crnr[:, -3])

    lats_crnr[0, :] = lats_crnr[1, :] - (lats_crnr[2, :] - lats_crnr[1, :])
    lats_crnr[-1, :] = lats_crnr[-2, :] + (lats_crnr[-2, :] - lats_crnr[-3, :])
    lats_crnr[:, 0] = lats_crnr[:, 1] - (lats_crnr[:, 2] - lats_crnr[:, 1])
    lats_crnr[:, -1] = lats_crnr[:, -2] + (lats_crnr[:, -2] - lats_crnr[:, -3])

    if londim == 1:
        vertices_longitude = np.zeros((nlat, nlon, 4))
        vertices_latitude = np.zeros((nlat, nlon, 4))
    else:
        vertices_longitude = np.zeros((nlon, nlat, 4))
        vertices_latitude = np.zeros((nlon, nlat, 4))

    # Fill in counter clockwise
    vertices_longitude[:, :, 0] = lons_crnr[:-1, :-1]
    vertices_longitude[:, :, 1] = lons_crnr[:-1, 1:]
    vertices_longitude[:, :, 2] = lons_crnr[1:, 1:]
    vertices_longitude[:, :, 3] = lons_crnr[1:, :-1]
    vertices_latitude[:, :, 0] = lats_crnr[:-1, :-1]
    vertices_latitude[:, :, 1] = lats_crnr[:-1, 1:]
    vertices_latitude[:, :, 2] = lats_crnr[1:, 1:]
    vertices_latitude[:, :, 3] = lats_crnr[1:, :-1]

    # Clip latitudes
    if clip_latitude:
        vertices_latitude = np.clip(vertices_latitude, -90.0, 90.0)

    # Once more correct meridian crossing cells
    lon_range = vertices_longitude.max(axis=2) - vertices_longitude.min(axis=2)
    lon_range = np.repeat(lon_range[:, :, np.newaxis], 4, axis=2)

    # a=vertices_longitude[np.where(lon_range>180)]
    # b=vertices_longitude[np.where(vertices_longitude>180)]
    # print(a.shape)
    # print(a)
    # print("-----------------------------------")
    # print(b.shape)
    # print(b)
    # print("-----------------------------------")
    vertices_longitude = np.where(
        np.logical_and(lon_range > 180, vertices_longitude < 0),
        vertices_longitude + 360.0,
        vertices_longitude,
    )
    # lon_range = vertices_longitude.max(axis=2)-vertices_longitude.min(axis=2)
    # lon_range = np.repeat(lon_range[:, :, np.newaxis], 4, axis=2)
    # a=vertices_longitude[np.where(lon_range>180)]
    # print(a.shape)
    # print(a)
    # print("-----------------------------------")
    # print(b.shape)
    # print(b)

    # Add to the dataset
    ds["vertices_latitude"] = (
        (ds[lat].dims[0], ds[lat].dims[1], "vertices"),
        vertices_latitude,
    )
    ds["vertices_longitude"] = (
        (ds[lon].dims[0], ds[lon].dims[1], "vertices"),
        vertices_longitude,
    )
    ds[lat].attrs["bounds"] = "vertices_latitude"
    ds[lon].attrs["bounds"] = "vertices_longitude"

    return ds


def generate_bounds_rectilinear(ds, lat, lon):
    """Compute bounds for rectilinear grids.

    The bounds will be attached as coords to the xarray.Dataset of the Grid object.
    If no bounds can be created, a warning is issued. It is assumed but not ensured that no
    duplicated cells are present in the grid.

    Parameters
    ----------
    ds : xarray.Dataset
        .
    lat : str
        Latitude variable name.
    lon : str
        Longitude variable name.

    Returns
    -------
    xarray.Dataset
        Dataset with attached bounds.
    """
    # Detect shape
    nlat, nlon, ncells = detect_shape(
        ds=ds, lat=lat, lon=lon, grid_type="regular_lat_lon"
    )

    # Assuming lat / lon values are strong monotonically decreasing/increasing
    # Latitude / Longitude bounds shaped (nlat, 2) / (nlon, 2)
    lat_bnds = np.zeros((ds[lat].shape[0], 2), dtype=np.float32)
    lon_bnds = np.zeros((ds[lon].shape[0], 2), dtype=np.float32)

    # lat_bnds
    #  positive<0 for strong monotonically increasing
    #  positive>0 for strong monotonically decreasing
    positive = ds[lat].values[0] - ds[lat].values[1]
    gspacingl = abs(positive)
    gspacingu = abs(ds[lat].values[-1] - ds[lat].values[-2])
    if positive < 0:
        lat_bnds[1:, 0] = (ds[lat].values[:-1] + ds[lat].values[1:]) / 2.0
        lat_bnds[:-1, 1] = lat_bnds[1:, 0]
        lat_bnds[0, 0] = ds[lat].values[0] - gspacingl / 2.0
        lat_bnds[-1, 1] = ds[lat].values[-1] + gspacingu / 2.0
    elif positive > 0:
        lat_bnds[1:, 1] = (ds[lat].values[:-1] + ds[lat].values[1:]) / 2.0
        lat_bnds[:-1, 0] = lat_bnds[1:, 1]
        lat_bnds[0, 1] = ds[lat].values[0] + gspacingl / 2.0
        lat_bnds[-1, 0] = ds[lat].values[-1] - gspacingu / 2.0
    else:
        warnings.warn(
            "The bounds could not be calculated since the latitude and/or longitude "
            "values are not strong monotonically decreasing/increasing."
        )
        return ds

    lat_bnds = np.where(lat_bnds < -90.0, -90.0, lat_bnds)
    lat_bnds = np.where(lat_bnds > 90.0, 90.0, lat_bnds)

    # lon_bnds
    positive = ds[lon].values[0] - ds[lon].values[1]
    gspacingl = abs(positive)
    gspacingu = abs(ds[lon].values[-1] - ds[lon].values[-2])
    if positive < 0:
        lon_bnds[1:, 0] = (ds[lon].values[:-1] + ds[lon].values[1:]) / 2.0
        lon_bnds[:-1, 1] = lon_bnds[1:, 0]
        lon_bnds[0, 0] = ds[lon].values[0] - gspacingl / 2.0
        lon_bnds[-1, 1] = ds[lon].values[-1] + gspacingu / 2.0
    elif positive > 0:
        lon_bnds[1:, 1] = (ds[lon].values[:-1] + ds[lon].values[1:]) / 2.0
        lon_bnds[:-1, 0] = lon_bnds[1:, 1]
        lon_bnds[0, 1] = ds[lon].values[0] + gspacingl / 2.0
        lon_bnds[-1, 0] = ds[lon].values[-1] - gspacingu / 2.0
    else:
        warnings.warn(
            "The bounds could not be calculated since the latitude and/or longitude "
            "values are not strong monotonically decreasing/increasing."
        )
        return ds

    # Add to the dataset
    ds["lat_bnds"] = ((ds[lat].dims[0], "bnds"), lat_bnds)
    ds["lon_bnds"] = ((ds[lon].dims[0], "bnds"), lon_bnds)
    ds[lat].attrs["bounds"] = "lat_bnds"
    ds[lon].attrs["bounds"] = "lon_bnds"

    return ds


def detect_coordinate(ds, coord_type):
    """Use cf_xarray to obtain the variable name of the requested coordinate.

    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
        Dataset the coordinate variable name shall be obtained from.
    coord_type: str
        Coordinate type understood by cf-xarray, eg. 'lat', 'lon', ...

    Raises
    ------
    KeyError
        Raised if the requested coordinate cannot be identified.

    Returns
    -------
    str
        Coordinate variable name.
    """
    error_msg = f"A {coord_type} coordinate cannot be identified in the dataset."

    # Make use of cf-xarray accessor
    coord = get_coord_by_type(ds, coord_type, ignore_aux_coords=False)
    if coord is None:
        coord = get_coord_by_attr(ds, "standard_name", coord_type)
        if coord is None:
            raise KeyError(error_msg)

    # Return the name of the coordinate variable
    return coord


def detect_bounds(ds, coordinate) -> Optional[str]:
    """Use cf_xarray to obtain the variable name of the requested coordinates bounds.

    Parameters
    ----------
    ds : xarray.Dataset, xarray.DataArray
        Dataset the coordinate bounds variable name shall be obtained from.
    coordinate : str
        Name of the coordinate variable to determine the bounds from.

    Returns
    -------
    str or None
        Returns the variable name of the requested coordinate bounds.
        Returns None if the variable has no bounds or if they cannot be identified.
    """
    try:
        return ds.cf.bounds[coordinate][0]
    except (KeyError, AttributeError):
        warnings.warn(
            "For coordinate variable '%s' no bounds can be identified." % coordinate
        )
    return


def detect_gridtype(ds, lon, lat, lon_bnds=None, lat_bnds=None):
    """Detect type of the grid as one of "regular_lat_lon", "curvilinear", "unstructured".

    Assumes the grid description / structure follows the CF conventions.
    """
    # 1D coordinate variables
    if ds[lat].ndim == 1 and ds[lon].ndim == 1:
        lat_1D = ds[lat].dims[0]
        lon_1D = ds[lon].dims[0]
        if not lat_bnds or not lon_bnds:
            if lat_1D == lon_1D:
                return "unstructured"
            else:
                return "regular_lat_lon"
        else:
            # unstructured: bounds [ncells, nvertices]
            if (
                lat_1D == lon_1D
                and all([ds[bnds].ndim == 2 for bnds in [lon_bnds, lat_bnds]])
                and all(
                    [
                        ds.sizes[dim] > 2
                        for dim in [
                            ds[lon_bnds].dims[-1],
                            ds[lat_bnds].dims[-1],
                        ]
                    ]
                )
            ):
                return "unstructured"
            # rectilinear: bounds [nlat/nlon, 2]
            elif all([ds[bnds].ndim == 2 for bnds in [lon_bnds, lat_bnds]]) and all(
                [
                    ds.sizes[dim] == 2
                    for dim in [
                        ds[lon_bnds].dims[-1],
                        ds[lat_bnds].dims[-1],
                    ]
                ]
            ):
                return "regular_lat_lon"
            else:
                raise ValueError("The grid type is not supported.")

    # 2D coordinate variables
    elif ds[lat].ndim == 2 and ds[lon].ndim == 2:
        # Test for curvilinear or restructure lat/lon coordinate variables
        # todo: Check if regular_lat_lon despite 2D
        #  - requires additional function checking
        #      lat[:,i]==lat[:,j] for all i,j
        #      lon[i,:]==lon[j,:] for all i,j
        #  - and if that is the case to extract lat/lon and *_bnds
        #      lat[:]=lat[:,j], lon[:]=lon[j,:]
        #      lat_bnds[:, 2]=[min(lat_bnds[:,j, :]), max(lat_bnds[:,j, :])]
        #      lon_bnds similar
        if not ds[lat].shape == ds[lon].shape:
            raise ValueError(
                "The horizontal coordinate variables have differing shapes."
            )
        else:
            if not lat_bnds or not lon_bnds:
                return "curvilinear"
            else:
                # Shape of curvilinear bounds either [nlat, nlon, 4] or [nlat+1, nlon+1]
                if list(ds[lat].shape) + [4] == list(ds[lat_bnds].shape) and list(
                    ds[lon].shape
                ) + [4] == list(ds[lon_bnds].shape):
                    return "curvilinear"
                elif [si + 1 for si in ds[lat].shape] == list(ds[lat_bnds].shape) and [
                    si + 1 for si in ds[lon].shape
                ] == list(ds[lon_bnds].shape):
                    return "curvilinear"
                else:
                    raise ValueError("The grid type is not supported.")

    # >2D coordinate variables, or coordinate variables of different dimensionality
    else:
        raise ValueError(
            "The horizontal coordinate variables have more than 2 dimensions."
        )
