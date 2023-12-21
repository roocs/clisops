import warnings
from typing import Optional, Tuple

import cf_xarray as cfxr  # noqa
import cftime
import numpy as np
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue
from roocs_utils.utils.time_utils import str_to_AnyCalendarDateTime
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_attr, get_coord_by_type


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
        "bounds": "lat_bnds",
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
        "axis": "Y",
    }
    lon_attrs = {
        "bounds": "lon_bnds",
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
                ds.dims["grid_corners"],
            )
        ).astype(np.float32)
        lon_b = ds.grid_corner_lon.values.reshape(
            (
                ds.grid_dims.values[1],
                ds.grid_dims.values[0],
                ds.dims["grid_corners"],
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


def detect_shape(ds, lat, lon, grid_type) -> Tuple[int, int, int]:
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

    if ds[lon].ndim != ds[lon].ndim:
        raise Exception(
            f"The coordinate variables {lat} and {lon} do not have the same number of dimensions."
        )
    elif ds[lat].ndim == 2:
        nlat = ds[lat].shape[0]
        nlon = ds[lon].shape[1]
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


def generate_bounds_curvilinear(ds, lat, lon):
    """Compute bounds for curvilinear grids.

    Assumes 2D latitude and longitude coordinate variables. The bounds will be attached as coords
    to the xarray.Dataset of the Grid object. If no bounds can be created, a warning is issued.
    It is assumed but not ensured that no duplicated cells are present in the grid.

    The bound calculation for curvilinear grids was adapted from
    https://github.com/SantanderMetGroup/ATLAS/blob/mai-devel/scripts/ATLAS-data/\
    bash-interpolation-scripts/AtlasCDOremappeR_CORDEX/grid_bounds_calc.py
    which based on work by Caillaud Cécile and Samuel Somot from Meteo-France.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to generate the bounds for.
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
    nlat, nlon, ncells = detect_shape(ds=ds, lat=lat, lon=lon, grid_type="curvilinear")

    # Rearrange lat/lons
    lons_row = ds[lon].data.flatten()
    lats_row = ds[lat].data.flatten()

    # Allocate lat/lon corners
    lons_cor = np.zeros(lons_row.size * 4)
    lats_cor = np.zeros(lats_row.size * 4)

    lons_crnr = np.empty((ds[lon].shape[0] + 1, ds[lon].shape[1] + 1))
    lons_crnr[:] = np.nan
    lats_crnr = np.empty((ds[lat].shape[0] + 1, ds[lat].shape[1] + 1))
    lats_crnr[:] = np.nan

    # -------- Calculating corners --------- #

    # Loop through all grid points except at the boundaries
    for ilat in range(1, ds[lon].shape[0]):
        for ilon in range(1, ds[lon].shape[1]):
            # SW corner for each lat/lon index is calculated
            lons_crnr[ilat, ilon] = (
                ds[lon][ilat - 1, ilon - 1]
                + ds[lon][ilat, ilon - 1]
                + ds[lon][ilat - 1, ilon]
                + ds[lon][ilat, ilon]
            ) / 4.0
            lats_crnr[ilat, ilon] = (
                ds[lat][ilat - 1, ilon - 1]
                + ds[lat][ilat, ilon - 1]
                + ds[lat][ilat - 1, ilon]
                + ds[lat][ilat, ilon]
            ) / 4.0

    # Grid points at boundaries
    lons_crnr[0, :] = lons_crnr[1, :] - (lons_crnr[2, :] - lons_crnr[1, :])
    lons_crnr[-1, :] = lons_crnr[-2, :] + (lons_crnr[-2, :] - lons_crnr[-3, :])
    lons_crnr[:, 0] = lons_crnr[:, 1] + (lons_crnr[:, 1] - lons_crnr[:, 2])
    lons_crnr[:, -1] = lons_crnr[:, -2] + (lons_crnr[:, -2] - lons_crnr[:, -3])

    lats_crnr[0, :] = lats_crnr[1, :] - (lats_crnr[2, :] - lats_crnr[1, :])
    lats_crnr[-1, :] = lats_crnr[-2, :] + (lats_crnr[-2, :] - lats_crnr[-3, :])
    lats_crnr[:, 0] = lats_crnr[:, 1] - (lats_crnr[:, 1] - lats_crnr[:, 2])
    lats_crnr[:, -1] = lats_crnr[:, -2] + (lats_crnr[:, -2] - lats_crnr[:, -3])

    # ------------ DONE ------------- #

    # Fill in counterclockwise and rearrange
    count = 0
    for ilat in range(ds[lon].shape[0]):
        for ilon in range(ds[lon].shape[1]):
            lons_cor[count] = lons_crnr[ilat, ilon]
            lons_cor[count + 1] = lons_crnr[ilat, ilon + 1]
            lons_cor[count + 2] = lons_crnr[ilat + 1, ilon + 1]
            lons_cor[count + 3] = lons_crnr[ilat + 1, ilon]

            lats_cor[count] = lats_crnr[ilat, ilon]
            lats_cor[count + 1] = lats_crnr[ilat, ilon + 1]
            lats_cor[count + 2] = lats_crnr[ilat + 1, ilon + 1]
            lats_cor[count + 3] = lats_crnr[ilat + 1, ilon]

            count += 4

    lon_bnds = lons_cor.reshape(nlat, nlon, 4)
    lat_bnds = lats_cor.reshape(nlat, nlon, 4)

    # Add to the dataset
    ds["lat_bnds"] = (
        (ds[lat].dims[0], ds[lat].dims[1], "vertices"),
        lat_bnds,
    )
    ds["lon_bnds"] = (
        (ds[lon].dims[0], ds[lon].dims[1], "vertices"),
        lon_bnds,
    )

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
        coord = get_coord_by_attr(ds, "standard_name", coord_type)
        if coord is None:
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
