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
    # todo: Should at all attributes for the bounds be defined?
    lat_bnds_attrs = {
        "long_name": "latitude_bounds",
        "units": "degrees_north",
    }
    lon_bnds_attrs = {
        "long_name": "longitude_bounds",
        "units": "degrees_east",
    }

    # Overwrite or update coordinate variables of input dataset
    try:
        if keep_attrs:
            ds[lat].attrs.update(lat_attrs)
            ds[lon].attrs.update(lon_attrs)
            ds[lat_bnds].attrs.update(lat_bnds_attrs)
            ds[lon_bnds].attrs.update(lon_bnds_attrs)
        else:
            ds[lat].attrs = lat_attrs
            ds[lon].attrs = lon_attrs
            ds[lat_bnds].attrs = lat_bnds_attrs
            ds[lon_bnds].attrs = lon_bnds_attrs
    except KeyError:
        raise KeyError("Not all specified coordinate variables exist in the dataset.")

    return ds


def reformat_SCRIP_to_CF(ds, keep_attrs=False):
    """
    Reformat dataset from SCRIP to CF format.

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
    """
    Reformat dataset from xESMF to CF format.

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

    lat_vertices = np.moveaxis(lat_vertices, 0, -1)
    lon_vertices = np.moveaxis(lon_vertices, 0, -1)

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
    """
    Detect format of a dataset. Yet supported are 'CF', 'SCRIP', 'xESMF'.

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


def detect_shape(ds, lat, lon, grid_type):
    """
    Detect the shape of the grid.

    Returns a tuple of (nlat, nlon, ncells). For an irregular grid nlat and nlon are not defined
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
        One of "regular_lat_lon", "curvilinear", "irregular"

    Returns
    -------
    nlat : int
        Number of latitude points in the grid.
    nlon : int
        Number of longitude points in the grid.
    ncells : int
        Number of cells in the grid.
    """
    if grid_type not in ["regular_lat_lon", "curvilinear", "irregular"]:
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
        if ds[lat].shape == ds[lon].shape and grid_type == "irregular":
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
    return (nlat, nlon, ncells)


def generate_bounds_curvilinear(ds, lat, lon):
    """
    Compute bounds for curvilinear grids.

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
    """
    Compute bounds for rectilinear grids.

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
