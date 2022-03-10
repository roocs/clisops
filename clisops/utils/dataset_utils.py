import math
import warnings

import cf_xarray as cfxr
import cftime
import numpy as np
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue
from roocs_utils.utils.time_utils import str_to_AnyCalendarDateTime
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_type

from clisops import logging

LOGGER = logging.getLogger(__file__)


def calculate_offset(lon, first_element_value):
    """
    Calculate the number of elements to roll the dataset by in order to have
    longitude from within requested bounds.

    :param lon: longitude coordinate of xarray dataset.
    :param first_element_value: the value of the first element of the longitude array to roll to.
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


def _crosses_0_meridian(lon_c):
    """
    Determine whether grid extents over the 0-meridian.

    Assumes approximate constant width of grid cells.

    Parameters
    ----------
    lon_c : TYPE
        Longitude coordinate variable in the longitude frame [-180, 180].

    Returns
    -------
    bool
        True for a dataset crossing the 0-meridian, False else.
    """
    if not isinstance(lon_c, xr.DataArray):
        raise InvalidParameterValue(
            "Input needs to be of type xarray.DataArray or np.ndarray."
        )

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


def cf_convert_between_lon_frames(ds_in, lon_interval):
    """
    Convert ds or lon_interval (whichever deems appropriate) to the other longitude frame, if the longitude frames do not match.

    If ds and lon_interval are defined on different longitude frames ([-180, 180] and [0, 360]),
    this function will convert the input parameters to the other longitude frame, preferably the
    lon_interval.
    Adjusts shifted longitude frames [0-x, 360-x] in the dataset to one of the two standard longitude
    frames, dependent on the specified lon_interval.

    Parameters
    ----------
    ds_in : xarray.Dataset or xarray.DataArray
        xarray data object with defined longitude dimension.
    lon_interval : tuple or list
        length-2-tuple or -list of floats or integers denoting the bounds of the longitude interval.

    Returns
    -------
    Tuple(ds, lon_low, lon_high)
        The xarray.Dataset and the bounds of the longitude interval, potentially adjusted in terms
        of their defined longitude frame.
    """
    # Collect input specs
    if isinstance(ds_in, (xr.Dataset, xr.DataArray)):
        lon = get_coord_by_type(ds_in, "longitude", ignore_aux_coords=False).name
        lat = get_coord_by_type(ds_in, "latitude", ignore_aux_coords=False).name
        lon_bnds = detect_bounds(ds_in, "longitude")
        gridtype = ""
        x = lon
        x_is_index = True
        if ds_in[lon].ndim == 1 and ds_in[lon].dims == ds_in[lat].dims:
            gridtype = "irregular"
        elif ds_in[lon].ndim == 1:
            gridtype = "regular_lat_lon"
        elif ds_in[lon].ndim == 2:
            gridtype = "curvilinear"
            x = ds_in[lon].dims[1]
            if x in ds_in.coords and (
                list(ds_in[x].values) == list(range(ds_in.dims[x]))
                or list(ds_in[x].values) == list(range(1, ds_in.dims[x] + 1))
            ):
                x_is_index = True
            elif x not in ds_in.coords:
                x_is_index = True
            else:
                x_is_index = False
        else:
            raise InvalidParameterValue(
                "The longitude coordinate variable has more than 2 dimensions."
            )
        ds = ds_in.copy()
    else:
        raise InvalidParameterValue(
            "This function requires an xarray.DataArray or xarray.Dataset as input."
        )
    low, high = lon_interval
    lon_min, lon_max = ds.coords[lon].values.min(), ds.coords[lon].values.max()
    atol = 0.5

    # Check longitude
    # For longitude frames other than [-180, 180] and [0, 360] in the dataset the following assumptions
    #  are being made:
    # - fixpoint is the 0-meridian
    # - the lon_interval is either defined in the longitude frame [-180, 180] or [0, 360]
    if lon_max - lon_min > 360 + atol or lon_min < -360 - atol or lon_max > 360 + atol:
        raise ValueError(
            "The longitude coordinate values have to lie within the interval "
            "[-360, 360] degrees and not exceed an extent of 360 degrees."
        )

    # Conversion between longitude frames if required
    if (lon_min <= low or np.isclose(low, lon_min, atol=atol)) and (
        lon_max >= high or np.isclose(high, lon_max, atol=atol)
    ):
        return ds, low, high

    # Conversion: longitude is a singleton dimension
    elif (ds[lon].ndim == 1 and ds.sizes[ds[lon].dims[0]] == 1) or (
        ds[lon].ndim > 1 and ds.sizes[ds[lon].dims[1]] == 1
    ):
        if low < 0 and lon_min > 0:
            ds[lon] = ds[lon].where(ds[lon] <= 180, ds[lon] - 360.0)
            if lon_bnds:
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] > 180, ds[lon_bnds] - 360.0
                )
            if not x_is_index:
                ds[x] = ds[x].where(ds[x] <= 180, ds[x] - 360.0)
        elif low > 0 and lon_min < 0:
            ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
            if lon_bnds:
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                )
            if not x_is_index:
                ds[x] = ds[x].where(ds[x] >= 0, ds[x] + 360.0)
        return ds, low, high

    # Conversion: 1D or 2D longitude coordinate variable
    else:
        # regional [0 ... 180]
        if lon_min >= 0 and lon_max <= 180:
            return ds, low, high

        # shifted frame beyond -180, eg. [-300, 60]
        elif lon_min < -180.5:
            if low < 0:
                ds[lon] = ds[lon].where(ds[lon] > -180, ds[lon] + 360.0)
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] > -180, ds[lon_bnds] + 360.0
                )
                if not x_is_index:
                    ds[x] = ds[x].where(ds[x] > -180, ds[x] + 360.0)
            elif low >= 0:
                ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                )
                if not x_is_index:
                    ds[x] = ds[x].where(ds[x] >= 0, ds[x] + 360.0)

        # shifted frame beyond 0, eg. [-60, 300]
        elif lon_min < -atol and lon_max > 180 + atol:
            if low < 0:
                ds[lon] = ds[lon].where(ds[lon] <= 180, ds[lon] - 360.0)
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] <= 180, ds[lon_bnds] - 360.0
                )
                if not x_is_index:
                    ds[x] = ds[x].where(ds[x] <= 180, ds[x] - 360.0)
            elif low >= 0:
                ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
                ds[lon_bnds] = ds[lon_bnds].where(
                    ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                )
                if not x_is_index:
                    ds[x] = ds[x].where(ds[x] >= 0, ds[x] + 360.0)

        # [-180 ... 180]
        elif lon_min < 0:
            # interval includes 180°-meridian: convert dataset to [0, 360]
            if low < 180 and high > 180:
                ds[lon] = ds[lon].where(ds[lon] >= 0, ds[lon] + 360.0)
                if lon_bnds:
                    ds[lon_bnds] = ds[lon_bnds].where(
                        ds[lon_bnds] >= 0, ds[lon_bnds] + 360.0
                    )
                if not x_is_index:
                    ds[x] = ds[x].where(ds[x] >= 0, ds[x] + 360.0)
            # interval does not include 180°-meridian: convert interval to [-180,180]
            else:
                if low >= 0:
                    low, high = _convert_interval_between_lon_frames(low, high)
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
                if not x_is_index:
                    ds[x] = ds[x].where(ds[x] <= 180, ds[x] - 360.0)
            # interval negative
            else:
                low, high = _convert_interval_between_lon_frames(low, high)
                return ds, low, high

        # Sort since order is no longer ascending / descending
        # 1D coordinate variable
        if gridtype == "irregular":
            return ds, low, high
        elif gridtype == "regular_lat_lon":
            ds = ds.sortby(lon)
            return ds, low, high
        # 2D coordinate variable
        else:
            # If x / longitude dimension does have a coordinate variable assigned that is not just
            # an index, assign a row of the dataset as coordinate for x-dimension,
            # sort by x to obtain ~ascending longitude coordinates
            # todo: sorting even necessary in that case?
            # todo: how does xesmf react to sorted / unsorted datasets?
            if not x_is_index:
                ds = ds.sortby(x)
            else:
                ds = ds.assign_coords({x: ds[lon][0, :].squeeze().values})
                ds = ds.sortby(x)

                if x not in ds_in.coords:
                    ds = ds.drop(x)
                else:
                    ds = ds.assign_coords({x: ds_in[x]})

            return ds, low, high


def check_lon_alignment(ds, lon_bnds):
    """
    Check whether the longitude subset requested is within the bounds of the dataset. If not try to roll the dataset so
    that the request is. Raise an exception if rolling is not possible.
    """
    low, high = lon_bnds
    lon = get_coord_by_type(ds, "longitude", ignore_aux_coords=False)
    lon = ds.coords[lon.name]
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
    """
    Check that the date specified exists in the calendar type of the dataset. If not,
    change the date a day at a time (up to a maximum of 5 times) to find a date that does exist.

    The direction to change the date by is indicated by 'direction'.

    :param da: xarray.Dataset or xarray.DataArray
    :param date: The date to check, as a string.
    :param direction: The direction to move in days to find a date that does exist.
                     'backwards' means the search will go backwards in time until an existing date is found.
                     'forwards' means the search will go forwards in time.
                      The default is 'backwards'.

    :return: (str) The next possible existing date in the calendar of the dataset.
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


def detect_coordinate(ds, coord_type):
    """
    Use cf_xarray to obtain the variable name of the requested coordinate.

    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
        Dataset the coordinate variable name shall be obtained from.
    coord_type : str
        Coordinate type understood by cf-xarray, eg. 'lat', 'lon', ...

    Raises
    ------
    AttributeError
        Raised if the requested coordinate cannot be identified.

    Returns
    -------
    str
        Coordinate variable name.
    """
    # Make use of cf-xarray accessor
    coord = ds.cf[coord_type]
    # coord = get_coord_by_type(ds, coord_type, ignore_aux_coords=False)

    # Return the name of the coordinate variable
    try:
        return coord.name
    except AttributeError:
        raise AttributeError(
            "A %s coordinate cannot be identified in the dataset." % coord_type
        )


def detect_bounds(ds, coordinate):
    """
    Use cf_xarray to obtain the variable name of the requested coordinates bounds.

    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
        Dataset the coordinate bounds variable name shall be obtained from.
    coordinate : str
        Name of the coordinate variable to determine the bounds from.

    Returns
    -------
    str
        Returns the variable name of the requested coordinate bounds,
        returns None if the variable has no bounds or they cannot be identified.
    """
    try:
        return ds.cf.bounds[coordinate][0]
    except (KeyError, AttributeError):
        warnings.warn(
            "For coordinate variable '%s' no bounds can be identified." % coordinate
        )
    return
