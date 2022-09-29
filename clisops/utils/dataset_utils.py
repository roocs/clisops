import warnings
from typing import Optional

import cf_xarray  # noqa
import cftime
import numpy as np
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue
from roocs_utils.utils.time_utils import str_to_AnyCalendarDateTime
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_type


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


def cf_convert_between_lon_frames(ds_in, lon_interval):
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
            # interval negative
            else:
                low, high = _convert_interval_between_lon_frames(low, high)
                return ds, low, high

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
    AttributeError
        Raised if the requested coordinate cannot be identified.

    Returns
    -------
    str
        Coordinate variable name.
    """
    error_msg = f"A {coord_type} coordinate cannot be identified in the dataset."

    # Make use of cf-xarray accessor
    try:
        coord = ds.cf[coord_type]
        # coord = get_coord_by_type(ds, coord_type, ignore_aux_coords=False)
    except KeyError:
        raise KeyError(error_msg)

    # Return the name of the coordinate variable
    try:
        return coord.name
    except AttributeError:
        raise AttributeError(error_msg)


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
                        ds.dims[dim] > 2
                        for dim in [
                            ds[lon_bnds].dims[-1],
                            ds[lat_bnds].dims[-1],
                        ]
                    ]
                )
            ):
                return "unstructured"
            # rectilinear: bounds [nlat/nlon, 2]
            elif (
                all([ds[bnds].ndim == 2 for bnds in [lon_bnds, lat_bnds]])
                and ds.dims[ds.cf.get_bounds_dim_name(lon)] == 2
            ):
                return "regular_lat_lon"
            else:
                raise Exception("The grid type is not supported.")

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
            raise InvalidParameterValue(
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
                    raise Exception("The grid type is not supported.")

    # >2D coordinate variables, or coordinate variables of different dimensionality
    else:
        raise InvalidParameterValue(
            "The horizontal coordinate variables have more than 2 dimensions."
        )
