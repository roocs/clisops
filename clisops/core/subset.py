"""Subset module."""

import numbers
import re
from collections.abc import Sequence
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union

import cf_xarray  # noqa
import geopandas as gpd
import numpy as np
import xarray
from packaging import version
from pandas import DataFrame
from pandas.api.types import is_integer_dtype  # noqa
from pyproj import Geod
from pyproj.crs import CRS
from pyproj.exceptions import CRSError
from shapely import vectorized
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split, unary_union
from xarray.core import indexing
from xarray.core.utils import get_temp_dimname

from clisops.utils.dataset_utils import adjust_date_to_calendar, get_coord_by_type
from clisops.utils.time_utils import to_isoformat

from .regrid import XESMF_MINIMUM_VERSION

__all__ = [
    "create_mask",
    "create_weight_masks",
    "distance",
    "subset_bbox",
    "shape_bbox_indexer",
    "subset_gridpoint",
    "subset_shape",
    "subset_time",
    "subset_time_by_values",
    "subset_time_by_components",
    "subset_level",
    "subset_level_by_values",
]

from loguru import logger


def get_lat(ds: Union[xarray.Dataset, xarray.DataArray]) -> xarray.DataArray:
    try:
        return ds.cf["latitude"]
    except KeyError:
        return ds.lat


def get_lon(ds: Union[xarray.Dataset, xarray.DataArray]) -> xarray.DataArray:
    try:
        return ds.cf["longitude"]
    except KeyError:
        return ds.lon


def check_start_end_dates(func: Callable) -> Callable:
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Verify that start and end dates are valid in a time subsetting function."""
        da = args[0]
        if "start_date" not in kwargs or kwargs["start_date"] is None:
            # use string for first year only - .sel() will include all time steps
            kwargs["start_date"] = da.time.min().dt.strftime("%Y").values
        if "end_date" not in kwargs or kwargs["end_date"] is None:
            # use string for last year only - .sel() will include all time steps
            kwargs["end_date"] = da.time.max().dt.strftime("%Y").values

        if isinstance(kwargs["start_date"], int) or isinstance(kwargs["end_date"], int):
            logger.warning(
                "start_date and end_date require dates in (type: str) "
                'using formats of "%Y", "%Y-%m" or "%Y-%m-%d".',
                UserWarning,
                stacklevel=2,
            )
            kwargs["start_date"] = str(kwargs["start_date"])
            kwargs["end_date"] = str(kwargs["end_date"])

        try:
            sel_time = da.time.sel(time=kwargs["start_date"])
            if sel_time.size == 0:
                raise ValueError()
        except KeyError:
            logger.warning(
                '"start_date" not found within input date time range. Defaulting to minimum time step in '
                "xarray object.",
                UserWarning,
                stacklevel=2,
            )
            kwargs["start_date"] = da.time.min().dt.strftime("%Y").values
        except ValueError:
            logger.warning(
                '"start_date" has been nudged to nearest valid time step in xarray object.',
                UserWarning,
                stacklevel=2,
            )
            kwargs["start_date"] = adjust_date_to_calendar(
                da, kwargs["start_date"], "forwards"
            )
            nudged = da.time.sel(time=slice(kwargs["start_date"], None)).values[0]
            kwargs["start_date"] = to_isoformat(nudged)

        try:
            sel_time = da.time.sel(time=kwargs["end_date"])
            if sel_time.size == 0:
                raise ValueError()
        except KeyError:
            logger.warning(
                '"end_date" not found within input date time range. Defaulting to maximum time step in '
                "xarray object.",
                UserWarning,
                stacklevel=2,
            )
            kwargs["end_date"] = da.time.max().dt.strftime("%Y").values
        except ValueError:
            logger.warning(
                '"end_date" has been nudged to nearest valid time step in xarray object.',
                UserWarning,
                stacklevel=2,
            )
            kwargs["end_date"] = adjust_date_to_calendar(
                da, kwargs["end_date"], "backwards"
            )
            nudged = da.time.sel(time=slice(None, kwargs["end_date"])).values[-1]
            kwargs["end_date"] = to_isoformat(nudged)

        if (
            da.time.sel(time=kwargs["start_date"]).min()
            > da.time.sel(time=kwargs["end_date"]).max()
        ):
            raise ValueError(
                f'Start date ("{kwargs["start_date"]}") is after end date ("{kwargs["end_date"]}").'
            )

        return func(*args, **kwargs)

    return func_checker


def check_start_end_levels(func: Callable) -> Callable:
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Verify that first and last levels are valid in a level subsetting function."""
        da = args[0]

        try:
            level = da[get_coord_by_type(da, "level", ignore_aux_coords=True)]
        except ValueError:
            raise Exception(
                f"{subset_level.__name__} requires input data that has a "
                'recognisable "level" coordinate.'
            )

        if "first_level" not in kwargs or kwargs["first_level"] is None:
            # use string for first level only - .sel() will include all levels
            kwargs["first_level"] = float(level.values[0])
        if "last_level" not in kwargs or kwargs["last_level"] is None:
            # use string for last level only - .sel() will include all levels
            kwargs["last_level"] = float(level.values[-1])

        # Check inputs are numbers, if not, try to convert to floats
        for key in ("first_level", "last_level"):
            if not isinstance(kwargs[key], numbers.Number):
                try:
                    kwargs[key] = float(kwargs[key])
                    logger.warning(
                        f'"{key}" should be a number, it has been converted to a float.',
                        UserWarning,
                        stacklevel=2,
                    )
                except Exception:
                    raise TypeError(
                        f'"{key}" could not parsed. It must be provided as a number'
                    )

        try:
            if float(kwargs["first_level"]) not in [float(lev) for lev in level.values]:
                raise ValueError()
        except ValueError:
            try:
                kwargs["first_level"] = level.sel(
                    **{level.name: slice(kwargs["first_level"], None)}
                ).values[0]
                logger.warning(
                    '"first_level" has been nudged to nearest valid level in xarray object.',
                    UserWarning,
                    stacklevel=2,
                )
            except IndexError:
                logger.warning(
                    '"first_level" not found within input level range. Defaulting to first level '
                    "in xarray object.",
                    UserWarning,
                    stacklevel=2,
                )
                kwargs["first_level"] = float(level.values[0])

        try:
            if float(kwargs["last_level"]) not in [float(lev) for lev in level.values]:
                raise ValueError()
        except ValueError:
            try:
                kwargs["last_level"] = level.sel(
                    **{level.name: slice(None, kwargs["last_level"])}
                ).values[-1]
                logger.warning(
                    '"last_level" has been nudged to nearest valid level in xarray object.',
                    UserWarning,
                    stacklevel=2,
                )
            except IndexError:
                logger.warning(
                    '"last_level" not found within input level range. Defaulting to last level '
                    "in xarray object.",
                    UserWarning,
                    stacklevel=2,
                )
                kwargs["last_level"] = float(level.values[-1])

        return func(*args, **kwargs)

    return func_checker


def check_lons(func: Callable) -> Callable:
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Reformat user-specified "lon" or "lon_bnds" values based on the lon dimensions of a supplied Dataset or DataArray.

        Examines an xarray object longitude dimensions and depending on extent (either -180 to +180 or 0 to +360),
        will reformat user-specified lon values to be synonymous with xarray object boundaries.
        Returns a numpy array of reformatted `lon` or `lon_bnds` in kwargs with min() and max() values.
        """
        if "lon_bnds" in kwargs:
            lon = "lon_bnds"
        elif "lon" in kwargs:
            lon = "lon"
        else:
            return func(*args, **kwargs)

        ds_lon = get_lon(args[0])

        if isinstance(args[0], (xarray.DataArray, xarray.Dataset)):
            if kwargs[lon] is None:
                kwargs[lon] = np.asarray(ds_lon.min(), ds_lon.max())
            else:
                kwargs[lon] = np.asarray(kwargs[lon])
            if np.all((ds_lon >= 0) | (np.isnan(ds_lon))) and np.all(kwargs[lon] < 0):
                if isinstance(kwargs[lon], float):
                    kwargs[lon] += 360
                else:
                    kwargs[lon][kwargs[lon] < 0] += 360
            elif np.all((ds_lon >= 0) | (np.isnan(ds_lon))) and np.any(kwargs[lon] < 0):
                raise NotImplementedError(
                    f"Input longitude bounds ({kwargs[lon]}) cross the 0 degree meridian but"
                    " dataset longitudes are all positive."
                )
            if np.all((ds_lon <= 0) | (np.isnan(ds_lon))) and np.any(kwargs[lon] > 180):
                if isinstance(kwargs[lon], float):
                    kwargs[lon] -= 360
                else:
                    kwargs[lon][kwargs[lon] <= 180] -= 360

        return func(*args, **kwargs)

    return func_checker


def check_levels_exist(func: Callable) -> Callable:
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Check the requested levels exist in the input Dataset/DataArray and, if not, raise an Exception.

        if the requested levels are not sorted in the order of the actual array then
        re-sort them to match the array in the input data.

        Modifies the "level_values" list in `kwargs` in place, if required.
        """
        da = args[0]

        req_levels = set(kwargs.get("level_values", set()))
        da_levels = da[get_coord_by_type(da, "level")]
        # round levels to precision 4. There might be level values like 1000.00000001 ... which would not match to 1000
        levels = {round(lev, 4) for lev in da_levels.values}

        if not req_levels.issubset(levels):
            mismatch_levels = req_levels.difference(levels)
            raise ValueError(
                f"Requested levels include some not found in "
                f"the dataset: {mismatch_levels}"
            )

        # Now re-order the requested levels in case they do not match the data order
        req_levels = sorted(req_levels)

        if da_levels.values[-1] < da_levels.values[0]:
            req_levels.reverse()

        # Re-set the requested levels to fixed values
        kwargs["level_values"] = req_levels
        return func(*args, **kwargs)

    return func_checker


def check_datetimes_exist(func: Callable) -> Callable:
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Check the requested datetimes exist in the input Dataset/DataArray and, if not, raise an Exception.

        If the requested datetimes are not sorted in the order of the actual array then
        re-sort them to match the array in the input data.

        Modifies the "time_values" list in `kwargs` in place, if required.
        """
        da = args[0]

        da_times = da[get_coord_by_type(da, "time")]
        tm_class = da_times.values[0].__class__
        times = {tm for tm in da_times.values}

        # Convert time values to required format/type
        req_times = {
            tm_class(*[int(i) for i in re.split("[-:T ]", tm)])
            for tm in kwargs.get("time_values", [])
        }

        if not req_times.issubset(times):
            mismatch_times = req_times.difference(times)
            raise ValueError(
                f"Requested datetimes include some not found in "
                f"the dataset: {mismatch_times}"
            )

        # Now re-order the requested times in case they do not match the data order
        req_times = sorted(req_times)
        if da_times.values[-1] < da_times.values[0]:
            req_times.reverse()

        # Re-set the requested times to fixed values
        kwargs["time_values"] = req_times
        return func(*args, **kwargs)

    return func_checker


def convert_lat_lon_to_da(func: Callable) -> Callable:
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Transform input lat, lon to DataArrays.

        Input can be int, float or any iterable.
        Expects a DataArray as first argument and checks is dim "site" already exists,
        uses "_site" in that case.

        If the input are not already DataArrays, the new lon and lat objects are 1D DataArrays
        with dimension "site".
        """
        lat = kwargs.pop("lat", None)
        lon = kwargs.pop("lon", None)
        if not isinstance(lat, (type(None), xarray.DataArray)) or not isinstance(
            lon, (type(None), xarray.DataArray)
        ):
            try:
                if len(lat) != len(lon):
                    raise ValueError("'lat' and 'lon' must have the same length")
            except TypeError:  # They have no len : not iterables
                lat = [lat]
                lon = [lon]
            ptdim = get_temp_dimname(args[0].dims, "site")
            if ptdim != "site" and len(lat) > 1:
                logger.warning(
                    f"Dimension 'site' already on input, output will use {ptdim} instead.",
                    UserWarning,
                    stacklevel=2,
                )
            lon = xarray.DataArray(lon, dims=(ptdim,))
            lat = xarray.DataArray(lat, dims=(ptdim,))
        return func(*args, lat=lat, lon=lon, **kwargs)

    return func_checker


def wrap_lons_and_split_at_greenwich(func: Callable) -> Callable:
    @wraps(func)
    def func_checker(*args, **kwargs):
        """Split and reproject polygon vectors in a GeoDataFrame whose values cross the Greenwich Meridian.

        Begins by examining whether the geometry bounds the supplied cross longitude = 0 and if so, proceeds to split
        the polygons at the meridian into new polygons and erase a small buffer to prevent invalid geometries when
        transforming the lons from WGS84 to WGS84 +lon_wrap=180 (longitudes from 0 to 360).

        Returns a GeoDataFrame with the new features in a wrap_lon WGS84 projection if needed.
        """
        try:
            poly = kwargs["poly"]
            x_dim = kwargs["x_dim"]
            wrap_lons = kwargs["wrap_lons"]
        except KeyError:
            return func(*args, **kwargs)

        if wrap_lons:
            if (np.min(x_dim) < 0 and np.max(x_dim) >= 360) or (
                np.min(x_dim) < -180 and np.max(x_dim) >= 180
            ):
                # TODO: This should raise an exception, right?
                logger.warning(
                    "DataArray doesn't seem to be using lons between 0 and 360 degrees or between -180 and 180 degrees."
                    " Tread with caution.",
                    UserWarning,
                    stacklevel=4,
                )

            split_features = dict()
            split_flag = False
            for index, feature in poly.iterrows():
                if (feature.geometry.bounds[0] < 0) and (
                    feature.geometry.bounds[2] > 0
                ):
                    split_flag = True
                    logger.warning(
                        "Geometry crosses the Greenwich Meridian. Proceeding to split polygon at Greenwich."
                        " This feature is experimental. Output might not be accurate.",
                        UserWarning,
                        stacklevel=4,
                    )

                    # Create a meridian line at Greenwich, split polygons at this line and erase a buffer line
                    if isinstance(feature.geometry, MultiPolygon):
                        union = MultiPolygon(unary_union(feature.geometry))
                    else:
                        union = Polygon(unary_union(feature.geometry))
                    meridian = LineString([Point(0, 90), Point(0, -90)])
                    buffered = meridian.buffer(0.000000001)
                    split_polygons = split(union, meridian)
                    buffered_split_polygons = [
                        feat.difference(buffered) for feat in split_polygons.geoms
                    ]

                    split_features[index] = [unary_union(buffered_split_polygons)]

            if split_flag:
                split_df = DataFrame.from_dict(
                    split_features,
                    orient="index",
                    columns=["geometry"],
                )
                split_gdf = gpd.GeoDataFrame(split_df, geometry=split_df.geometry)
                poly.update(split_gdf)

            # Set CRS on polygon for correct reprojection
            poly = poly.set_crs(CRS(4326))

            # Reproject features in WGS84 CSR to use 0 to 360 as longitudinal values
            wrapped_lons = CRS.from_string(
                "+proj=longlat +ellps=WGS84 +lon_wrap=180 +datum=WGS84 +no_defs"
            )

            poly = poly.to_crs(crs=wrapped_lons)
            if split_flag:
                logger.warning(
                    "Rebuffering split polygons to ensure edge inclusion in selection.",
                    UserWarning,
                    stacklevel=4,
                )
                poly = gpd.GeoDataFrame(poly.buffer(0.000000001), columns=["geometry"])
                poly.crs = wrapped_lons

            kwargs["poly"] = poly

        return func(*args, **kwargs)

    return func_checker


@wrap_lons_and_split_at_greenwich
def create_mask(
    *,
    x_dim: xarray.DataArray,
    y_dim: xarray.DataArray,
    poly: gpd.GeoDataFrame,
    wrap_lons: bool = False,
    check_overlap: bool = False,
) -> xarray.DataArray:
    """Create a mask with values corresponding to the features in a GeoDataFrame using vectorize methods.

    The returned mask's points have the value of the first geometry of `poly` they fall in.

    Parameters
    ----------
    x_dim : xarray.DataArray
        X or longitudinal dimension of xarray object. Can also be given through `ds_in`.
    y_dim : xarray.DataArray
        Y or latitudinal dimension of xarray object. Can also be given through `ds_in`.
    poly : gpd.GeoDataFrame
        A GeoDataFrame used to create the xarray.DataArray mask. If its index doesn't have an
        integer dtype, it will be reset to integers, which will be used in the mask.
    wrap_lons : bool
        Shift vector longitudes by -180,180 degrees to 0,360 degrees; Default = False
    check_overlap : bool
        Perform a check to verify if shapes contain overlapping geometries.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    .. code-block:: python

        import geopandas as gpd
        from clisops.core.subset import create_mask

        ds = xr.open_dataset(path_to_tasmin_file)
        polys = gpd.read_file(path_to_multi_shape_file)

        # Get a mask from all polygons in the shape file
        mask = create_mask(x_dim=ds.lon, y_dim=ds.lat, poly=polys)
        ds = ds.assign_coords(regions=mask)

        # Operations can be applied to each region with `groupby`. Ex:
        ds = ds.groupby("regions").mean()

        # Extra step to retrieve the names of those polygons stored in another column (here "id")
        region_names = xr.DataArray(polys.id, dims=("regions",))
        ds = ds.assign_coords(regions_names=region_names)
    """
    if check_overlap:
        _check_has_overlaps(polygons=poly)
    if wrap_lons:
        logger.warning("Wrapping longitudes at 180 degrees.", UserWarning, stacklevel=2)

    if len(x_dim.shape) == 1 and len(y_dim.shape) == 1 and x_dim.dims != y_dim.dims:
        # create a 2d grid of lon, lat values
        lon1, lat1 = np.meshgrid(
            np.asarray(x_dim.values), np.asarray(y_dim.values), indexing="ij"
        )
        dims_out = x_dim.dims + y_dim.dims
        coords_out = dict()
        coords_out[dims_out[0]] = x_dim.values
        coords_out[dims_out[1]] = y_dim.values
    else:
        lon1 = x_dim.values
        lat1 = y_dim.values
        dims_out = x_dim.dims
        coords_out = x_dim.coords

    if not is_integer_dtype(poly.index.dtype):
        poly = poly.reset_index()

    geoms = poly.geometry.values
    mask = np.full(lat1.shape, np.nan)
    for val, geom in zip(poly.index[::-1], geoms[::-1]):
        contained = vectorized.contains(geom, lon1.flatten(), lat1.flatten()).reshape(
            lat1.shape
        )
        touched = vectorized.touches(geom, lon1.flatten(), lat1.flatten()).reshape(
            lat1.shape
        )
        intersection = np.logical_or(contained, touched)
        mask[intersection] = val

    mask = xarray.DataArray(mask, dims=dims_out, coords=coords_out)

    return mask


def _rectilinear_grid_exterior_polygon(ds: xarray.Dataset) -> Polygon:
    """Return a polygon tracing a rectilinear grid's exterior.

    Parameters
    ----------
    ds : xarray.Dataset
        CF-compliant input dataset.

    Returns
    -------
    shapely.geometry.Polygon
        Grid cell boundary.
    """

    # Add bounds if not present
    # Note: with cf-xarray <= 0.6.2, the fact that `longitude` is in bounds does not mean it really is...
    # See https://github.com/xarray-contrib/cf-xarray/issues/254
    # So the commented code below does not work.
    # if 'longitude' not in ds.cf.bounds:
    #     ds = ds.cf.add_bounds("longitude")
    # if 'latitude' not in ds.cf.bounds:
    #     ds = ds.cf.add_bounds("latitude")
    #
    # x = ds.cf.get_bounds("longitude")  # lon_bnds
    # y = ds.cf.get_bounds("latitude")  # lat_bnds

    # This is the alternative for now.
    try:
        x = ds.cf.get_bounds("longitude")  # lon_bnds
        y = ds.cf.get_bounds("latitude")  # lat_bnds
    except KeyError:
        ds = ds.cf.add_bounds("longitude")
        ds = ds.cf.add_bounds("latitude")
        x = ds.cf.get_bounds("longitude")  # lon_bnds
        y = ds.cf.get_bounds("latitude")  # lat_bnds

    # Take the grid corner coordinates
    xmin = x[0, 0]
    xmax = x[-1, -1]
    ymin = y[0, 0]
    ymax = y[-1, -1]

    pts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    return Polygon(pts)


def _curvilinear_grid_exterior_polygon(
    ds: xarray.Dataset, mode: str = "bbox"
) -> Polygon:
    """Return a polygon tracing a curvilinear grid's exterior.

    Parameters
    ----------
    ds : xarray.Dataset
        CF-compliant input dataset.
    mode : {bbox, cell_union}
        Calculation mode. `bbox` takes the min and max longitude and latitude bounds and rounds them to 0.1 degree.
        `cell_union` merges all grid cell polygons and finds the exterior. Also rounds and simplifies the coordinates
        to smooth projection errors.

    Returns
    -------
    shapely.geometry.Polygon
        Grid cell boundary.
    """
    import math

    from shapely.ops import unary_union

    def round_up(x, decimal=1):
        f = 10**decimal
        return math.ceil(x * f) / f

    def round_down(x, decimal=1):
        f = 10**decimal
        return math.floor(x * f) / f

    if mode == "bbox":
        try:
            # cf-convention
            x = ds.cf.get_bounds("longitude")  # lon_bnds
            y = ds.cf.get_bounds("latitude")  # lat_bnds
        except KeyError:
            try:
                # xesmf convention
                x = ds.lon_b
                y = ds.lat_b
            except KeyError:
                # Fall-back to grid centers
                x = ds.cf.coordinates["longitude"]
                y = ds.cf.coordinates["latitude"]

        xmin = round_down(x.min())
        xmax = round_up(x.max())
        ymin = round_down(y.min())
        ymax = round_up(y.max())

        pts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]

    elif mode == "cell_union":
        # x and y should be vertices.
        # There is no guarantee that the sides of the array storing the curvilinear grids corresponds to the exterior of
        # the lon/lat grid.
        # For example, in a polar stereographic projection, the pole would be at the center of the native grid.
        # So we need to create individual polygons for each grid cell, take the union and get the exterior. Even then,
        # for some grids, projection distortions might introduce errors.
        # Consider this code experimental.

        # If the following fails, it's probably because the axis attribute is not set for the coordinates.
        xax = ds.cf.axes["X"][0]
        yax = ds.cf.axes["Y"][0]

        # Stack i and j
        sds = ds.stack(zkz_=(xax, yax))

        x = sds.cf.get_bounds("longitude")  # lon_bnds
        y = sds.cf.get_bounds("latitude")  # lat_bnds

        # Grid cell polygons
        polys = [Polygon(zip(lx, ly)) for lx, ly in zip(x.data.T, y.data.T)]

        # Exterior of all these polygons
        pts = unary_union(polys).simplify(0.1).buffer(0.1).exterior
        x, y = np.around(pts.xy, 1)
        y = np.clip(y, -90, 90)
        pts = zip(x, y)
    else:
        raise NotImplementedError(f"mode: {mode}")

    return Polygon(pts)


def grid_exterior_polygon(ds: xarray.Dataset) -> Polygon:
    """Return a polygon tracing the grid's exterior.

    This function is only accurate for a geographic lat/lon projection. For projected grids, it's a rough approximation.

    Parameters
    ----------
    ds : xarray.Dataset
        CF-compliant input dataset.

    Returns
    -------
    shapely.geometry.Polygon
        Grid cell boundary.

    Notes
    -----
    For curvilinear grids, the boundary is the centroid's boundary, not the real cell boundary. Please submit a PR if
    you need this.
    """
    from shapely.geometry import Polygon

    if is_rectilinear(ds):
        return _rectilinear_grid_exterior_polygon(ds)

    return _curvilinear_grid_exterior_polygon(ds, mode="bbox")


def is_rectilinear(ds: Union[xarray.Dataset, xarray.DataArray]) -> bool:
    """Return whether the grid is rectilinear or not."""
    sdims = {ds.cf["longitude"].name, ds.cf["latitude"].name}
    return sdims.issubset(ds.dims)


def shape_bbox_indexer(
    ds: xarray.Dataset,
    poly: Union[gpd.GeoDataFrame, gpd.GeoSeries, gpd.array.GeometryArray],
):
    """Return a spatial indexer that selects the indices of the grid cells covering the given geometries.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    poly : gpd.GeoDataFrame, gpd.GeoSeries, pd.array.GeometryArray, or list of shapely geometries.
        Shapes to cover. Can be of type Polygon, MultiPolygon, Point, or MultiPoint.

    Returns
    -------
    dict
        xarray indexer along native dataset coordinates, to be used as an argument to `isel`.

    Examples
    --------
    >>> indexer = shape_bbox_indexer(ds, poly)
    >>> ds.isel(indexer)

    Notes
    -----
    This is used in particular to restrict the domain of a dataset before computing the weights for a spatial average.
    """
    # Cling wrap around shapes: GeoSeries -> Polygon
    # The first `convex_hull` is necessary to remove any holes in the polygon.
    # The `GeoSeries.unary_union` is necessary to merge all shapes into a single polygon.

    if isinstance(poly, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # Note that this function is somewhat redundant with functionality found in rioxarray (see `clip` and `Window`).
        geom = poly.geometry
        hull = geom.convex_hull.unary_union.convex_hull
    elif isinstance(poly, gpd.array.GeometryArray):
        hull = poly.convex_hull.unary_union().convex_hull
    elif isinstance(poly, (list, tuple)):
        hpoly = [p.convex_hull for p in poly]
        hull = unary_union(hpoly).convex_hull
    else:
        raise ValueError(
            "poly must be a GeoDataFrame, GeoSeries, GeometryArray, or list of shapely geometries."
        )

    # If polygon straddles the grid boundary, we need to roll the grid's coordinates and this is not supported.
    if not grid_exterior_polygon(ds).contains(hull):
        return {}

    # If the grid is rectilinear, we can use the minimum rotated rectangle to simplify the outline further.
    rectilinear = is_rectilinear(ds)
    if rectilinear:
        hull = hull.minimum_rotated_rectangle

    # If the geometries are one or two points, it's hull does not form a polygon, and we need to handle the special
    # cases for one and two points.
    if hasattr(hull.boundary, "geoms"):
        # Handle cases with 1 point. Putting the same point twice to avoid issues with zip later.
        if len(hull.boundary.geoms) == 0:
            coords = hull.xy, hull.xy
        # Handle cases with 2 points
        else:
            # Extract the lon, lat coordinates from the points themselves
            coords = [geom.xy for geom in hull.boundary.geoms]

    # Handle typical polygon case
    else:
        # Extract the edge vertices (last item is just a copy of the first to close the polygon)
        coords = hull.boundary.coords[:-1]

    # Create envelope coordinates
    elon, elat = map(np.array, zip(*coords))
    ind = {ds.cf["longitude"].name: elon, ds.cf["latitude"].name: elat}

    # Find indices nearest the rectangle' corners
    # Note that the nearest indices might be inside the shape, so we'll need to add a *halo* around those indices.
    if rectilinear:
        if version.parse(xarray.__version__) < version.Version("2022.6.0"):
            logger.warning(
                "CLISOPS will require xarray >= 2022.06 in the next minor release. "
                "Please update your environment dependencies.",
                DeprecationWarning,
            )
            native_ind, _ = xarray.core.coordinates.remap_label_indexers(
                ds, ind, method="nearest"
            )
        else:
            native_ind = indexing.map_index_queries(
                ds, ind, method="nearest"
            ).dim_indexers
    else:
        # For curvilinear grids, finding the closest points require a bit more work.
        from scipy.spatial import cKDTree

        # These are going to be 2D grids.
        lon, lat = ds.cf["longitude"], ds.cf["latitude"]
        # Create KDTree to speed up search
        tree = cKDTree(np.vstack([lon.data.ravel(), lat.data.ravel()]).T)
        # Find indices on flattened coordinates
        _, flat_ind = tree.query(np.vstack([elon, elat]).T)
        # Find indices on 2D coordinates
        inds = np.unravel_index(flat_ind, lon.shape)
        # Create index dictionary on native dimensions, e.g. rlon, rlat
        native_ind = dict(zip(lon.dims, inds))

    # Create slices, adding a halo around selection to account for `nearest` grid cell center approximation.
    out = {}
    halo = 2
    for k, v in native_ind.items():
        vmin = np.clip(v.min() - halo, 0, ds[k].size)
        vmax = np.clip(v.max() + halo + 1, 0, ds[k].size)
        out[k] = slice(vmin, vmax)
    return out


def create_weight_masks(
    ds_in: Union[xarray.DataArray, xarray.Dataset],
    poly: gpd.GeoDataFrame,
) -> xarray.DataArray:
    """Create weight masks corresponding to the features in a GeoDataFrame using xESMF.

    The returned masks values are the fraction of the corresponding polygon's area
    that is covered by the grid cell. Summing along the spatial dimension will give 1
    for each geometry. Requires xESMF.

    Parameters
    ----------
    ds_in : Union[xarray.DataArray, xarray.Dataset]
        xarray object containing the grid information, as understood by xESMF.
        For 2D lat/lon coordinates, the bounds arrays are required,
    poly : gpd.GeoDataFrame
        GeoDataFrame used to create the xarray.DataArray mask.
        One mask will be created for each row in the dataframe.
        Will be converted to EPSG:4326 if needed.

    Returns
    -------
    xarray.DataArray
      Has a new `geom` dimension corresponding to the index of the input GeoDataframe.
      Non-geometry columns of `poly` are copied as auxiliary coordinates.

    Examples
    --------
    >>> import geopandas as gpd  # doctest: +SKIP
    >>> import xarray as xr  # doctest: +SKIP
    >>> from clisops.core.subset import create_weight_masks  # doctest: +SKIP
    >>> ds = xr.open_dataset(path_to_tasmin_file)  # doctest: +SKIP
    >>> polys = gpd.read_file(path_to_multi_shape_file)  # doctest: +SKIP
    # Get a weight mask for each polygon in the shape file
    >>> mask = create_weight_masks(
    ...     x_dim=ds.lon, y_dim=ds.lat, poly=polys
    ... )  # doctest: +SKIP
    """
    try:
        from xesmf import SpatialAverager
    except ImportError:
        raise ValueError(
            f"Package xesmf >= {XESMF_MINIMUM_VERSION} is required to use create_weight_masks."
        )

    if poly.crs is not None:
        poly = poly.to_crs(4326)

    poly = poly.copy()
    poly.index.name = "geom"
    poly_coords = poly.drop("geometry", axis="columns").to_xarray()

    savg = SpatialAverager(ds_in, poly.geometry)
    # Unpack weights to full size array, this increases memory use a lot.
    # polygons are along the "geom" dim
    # assign all other columns of poly as auxiliary coords.
    weights = (
        savg.weights.data.todense()
        if isinstance(savg.weights, xarray.DataArray)
        else savg.weights.toarray()
    )
    masks = xarray.DataArray(
        weights.reshape(poly.geometry.size, *savg.shape_in),
        dims=("geom", *savg.in_horiz_dims),
        coords=dict(**poly_coords, **poly_coords.coords),
    )

    # Assign coords from ds_in, but only those with no unknown dims.
    # Otherwise, xarray rises an error.
    masks = masks.assign_coords(
        **{
            k: crd
            for k, crd in ds_in.coords.items()
            if not (set(crd.dims) - set(masks.dims))
        }
    )
    return masks


def subset_shape(
    ds: Union[xarray.DataArray, xarray.Dataset],
    shape: Union[str, Path, gpd.GeoDataFrame],
    raster_crs: Optional[Union[str, int]] = None,
    shape_crs: Optional[Union[str, int]] = None,
    buffer: Optional[Union[int, float]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    first_level: Optional[Union[float, int]] = None,
    last_level: Optional[Union[float, int]] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset a DataArray or Dataset spatially (and temporally) using a vector shape and date selection.

    Return a subset of a DataArray or Dataset for grid points falling within the area of a Polygon and/or
    MultiPolygon shape, or grid points along the path of a LineString and/or MultiLineString. If the shape
    consists of several disjoint polygons, the output is cut to the smallest bbox including all
    polygons.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
        Input values.
    shape : Union[str, Path, gpd.GeoDataFrame]
        Path to a shape file, or GeoDataFrame directly. Supports GeoPandas-compatible formats.
    raster_crs : Optional[Union[str, int]]
        EPSG number or PROJ4 string.
    shape_crs : Optional[Union[str, int]]
        EPSG number or PROJ4 string.
    buffer : Optional[Union[int, float]]
        Buffer the shape in order to select a larger region stemming from it.
        Units are based on the shape degrees/metres.
    start_date : Optional[str]
        Start date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to first day of input data-array.
    end_date : Optional[str]
        End date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to last day of input data-array.
    first_level : Optional[Union[int, float]]
        First level of the subset.
        Can be either an integer or float.
        Defaults to first level of input data-array.
    last_level : Optional[Union[int, float]]
        Last level of the subset.
        Can be either an integer or float.
        Defaults to last level of input data-array.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        A subset of `ds`

    Notes
    -----
    If no CRS is found in the shape provided (e.g. RFC-7946 GeoJSON, https://en.wikipedia.org/wiki/GeoJSON),
    assumes a decimal degree datum (CRS84). Be advised that EPSG:4326 and OGC:CRS84 are not identical as axis order of
    lat and long differs between the two (for more information, see: https://github.com/OSGeo/gdal/issues/2035).

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_shape

        pr = xr.open_dataset(path_to_pr_file).pr

        # Subset data array by shape
        prSub = subset_shape(pr, shape=path_to_shape_file)

        # Subset data array by shape and single year
        prSub = subset_shape(
            pr, shape=path_to_shape_file, start_date="1990-01-01", end_date="1990-12-31"
        )

        # Subset multiple variables in a single dataset
        ds = xr.open_mfdataset([path_to_tasmin_file, path_to_tasmax_file])
        dsSub = subset_shape(ds, shape=path_to_shape_file)
    """
    wgs84 = CRS(4326)
    # PROJ4 definition for WGS84 with longitudes ranged between -180/+180.
    wgs84_wrapped = CRS.from_string(
        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs lon_wrap=180"
    )

    if isinstance(ds, xarray.DataArray):
        ds_copy = ds.to_dataset(name=ds.name or "subsetted")
    else:
        ds_copy = ds.copy()

    if isinstance(shape, gpd.GeoDataFrame):
        poly = shape.copy()
    else:
        poly = gpd.GeoDataFrame.from_file(shape)

    if buffer is not None:
        poly.geometry = poly.buffer(buffer)

    # Determine whether CRS types are the same between shape and raster
    if shape_crs is not None:
        try:
            shape_crs = CRS.from_user_input(shape_crs)
        except ValueError:
            raise
    else:
        try:
            shape_crs = CRS(poly.crs)
        except CRSError:
            minbnd, _, maxbnd, _ = poly.total_bounds
            # This is guessing that lons are wrapped around at 180+ but without much information, this might not be true
            if minbnd >= -180 and maxbnd <= 180:
                shape_crs = wgs84
            elif minbnd >= 0 and minbnd <= 360:
                shape_crs = wgs84_wrapped
            else:
                raise CRSError(
                    "Shapefile CRS could not be determined and does not resemble WGS84."
                )
            poly.crs = shape_crs
    if not shape_crs.equals(wgs84):
        logger.warning(
            "Shapefile CRS is not WGS84 and will be reprojected to WGS84. This may lead to inaccuracies.",
            UserWarning,
            stacklevel=2,
        )
        poly = poly.to_crs(wgs84)
        shape_crs = wgs84

    lon = get_lon(ds_copy)
    wrap_lons = False
    if raster_crs is not None:
        try:
            raster_crs = CRS.from_user_input(raster_crs)
        except ValueError:
            raise
    else:
        try:
            # Extract CF-compliant CRS_WKT from crs variable.
            raster_crs = CRS.from_cf(ds_copy.crs.attrs)
        except AttributeError as e:
            # This is guessing that lons are wrapped around at 180+ but without much information, this might not be true
            if np.min(lon) >= -180 and np.max(lon) <= 180:
                raster_crs = wgs84
            elif np.min(lon) >= 0 and np.max(lon) <= 360:
                wrap_lons = True
                raster_crs = wgs84_wrapped
            else:
                raise CRSError(
                    "Raster CRS is not known and does not resemble WGS84."
                ) from e

    _check_crs_compatibility(shape_crs=shape_crs, raster_crs=raster_crs)

    # Get the shape's bounding box.
    minx, miny, maxx, maxy = poly.total_bounds
    lon_bnds = (minx, maxx)
    lat_bnds = (miny, maxy)

    if np.min(lat_bnds) < -90 or np.max(lat_bnds) > 90:
        raise ValueError("Latitudes exceed domain of WGS84 coordinate system.")
    if np.min(lon_bnds) < -180 or np.max(lon_bnds) > 180:
        raise ValueError("Longitudes exceed domain of WGS84 coordinate system.")

    # If polygon doesn't cross prime meridian, subset bbox first to reduce processing time.
    # Only case not implemented is when lon_bnds cross the 0 deg meridian but dataset grid has all positive lons.
    try:
        ds_copy = subset_bbox(ds_copy, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    except ValueError as e:
        raise ValueError(
            "No grid cell centroids found within provided polygon bounding box. "
            'Try using the "buffer" option to create an expanded area.'
        ) from e
    except NotImplementedError:
        pass

    if start_date or end_date:
        ds_copy = subset_time(ds_copy, start_date=start_date, end_date=end_date)

    if first_level or last_level:
        ds_copy = subset_level(ds_copy, first_level=first_level, last_level=last_level)

    # Get the new lon and lat coordinates after the bbox subset.
    lon = get_lon(ds_copy)
    lat = get_lat(ds_copy)

    mask_2d = create_mask(x_dim=lon, y_dim=lat, poly=poly, wrap_lons=wrap_lons).clip(
        1, 1
    )
    # 1 on the shapes, NaN elsewhere.
    # We simply want to remove the 0s from the zeroth shape, for our outer mask trick below.

    if np.all(mask_2d.isnull()):
        raise ValueError(
            f"No grid cell centroids found within provided polygon bounds ({poly.bounds}). "
            'Try using the "buffer" option to create an expanded areas or verify polygon.'
        )

    sp_dims = set(mask_2d.dims)  # Spatial dimensions

    if len(sp_dims) > 1:
        # Find the outer mask. When subsetting unconnected shapes,
        # we don't want to drop the inner NaN regions, it may cause problems downstream.
        inner_mask = xarray.full_like(mask_2d, True, dtype=bool)
        for dim in sp_dims:
            # For each dimension, propagate shape indexes in either directions
            # Then sum on the other dimension. You get a step function going from 0 to X.
            # The non-zero part that left and right have in common is the "inner" zone.
            left = mask_2d.bfill(dim).sum(sp_dims - {dim})
            right = mask_2d.ffill(dim).sum(sp_dims - {dim})
            # True in the inner zone, False in the outer
            inner_mask = inner_mask & (left != 0) & (right != 0)

        # inner_mask including the shapes
        inner_mask = mask_2d.notnull() | inner_mask
    else:
        # in the locstream case inner_mask remains all True, but all non-polygon values can be dropped,
        # so here "outside inner_mask" is everything outside the polygon.
        inner_mask = mask_2d.notnull()

    # loop through variables
    for v in ds_copy.data_vars:
        if set.issubset(sp_dims, set(ds_copy[v].dims)):
            # 1st mask values outside shape, then drop values outside inner_mask
            ds_copy[v] = ds_copy[v].where(mask_2d.notnull())

    # Remove grid points outside the inner mask
    # Then extract the coords.
    # Using a where(inner_mask) on ds_copy triggers warnings with dask, sel seems safer.
    # But this only works if dims have coords.
    if set(sp_dims).issubset(ds_copy.coords.keys()):
        mask_2d = mask_2d.where(inner_mask, drop=True)
        for dim in sp_dims:
            ds_copy = ds_copy.sel({dim: mask_2d[dim]})
    else:
        ds_copy = ds_copy.where(inner_mask, drop=True)

    # Add a CRS definition using CF conventions and as a global attribute in CRS_WKT for reference purposes
    ds_copy.attrs["crs"] = raster_crs.to_string()
    ds_copy["crs"] = 1
    ds_copy["crs"].attrs.update(raster_crs.to_cf())

    for v in ds_copy.variables:
        if {lat.name, lon.name}.issubset(set(ds_copy[v].dims)):
            ds_copy[v].attrs["grid_mapping"] = "crs"

    if isinstance(ds, xarray.DataArray):
        ds_copy = list(ds_copy.data_vars.values())[0]
        ds_copy.name = ds.name
    return ds_copy


@check_lons
def subset_bbox(
    da: Union[xarray.DataArray, xarray.Dataset],
    lon_bnds: Union[np.array, tuple[Optional[float], Optional[float]]] = None,
    lat_bnds: Union[np.array, tuple[Optional[float], Optional[float]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    first_level: Optional[Union[float, int]] = None,
    last_level: Optional[Union[float, int]] = None,
    time_values: Optional[Sequence[str]] = None,
    level_values: Optional[Union[Sequence[float], Sequence[int]]] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset a DataArray or Dataset spatially (and temporally) using a lat lon bounding box and date selection.

    Return a subset of a DataArray or Dataset for grid points falling within a spatial bounding box
    defined by longitude and latitudinal bounds and for dates falling within provided bounds.

    TODO: returns the what?
    In the case of a lat-lon rectilinear grid, this simply returns the

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
        Input data.
    lon_bnds : Union[np.array, Tuple[Optional[float], Optional[float]]]
        List of minimum and maximum longitudinal bounds. Optional. Defaults to all longitudes in original data-array.
    lat_bnds : Union[np.array, Tuple[Optional[float], Optional[float]]]
        List of minimum and maximum latitudinal bounds. Optional. Defaults to all latitudes in original data-array.
    start_date : Optional[str]
        Start date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to first day of input data-array.
    end_date : Optional[str]
        End date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to last day of input data-array.
    first_level : Optional[Union[int, float]]
        First level of the subset.
        Can be either an integer or float.
        Defaults to first level of input data-array.
    last_level : Optional[Union[int, float]]
        Last level of the subset.
        Can be either an integer or float.
        Defaults to last level of input data-array.
    time_values: Optional[Sequence[str]]
        A list of datetime strings to subset.
    level_values: Optional[Union[Sequence[float], Sequence[int]]]
        A list of level values to select.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        Subsetted xarray.DataArray or xarray.Dataset

    Notes
    -----
    subset_bbox expects the lower and upper bounds to be provided in ascending order.
    If the actual coordinate values are descending then this will be detected
    and your selection reversed before the data subset is returned.

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_bbox

        ds = xr.open_dataset(path_to_pr_file)

        # Subset lat lon
        prSub = subset_bbox(ds.pr, lon_bnds=[-75, -70], lat_bnds=[40, 45])
    """
    lat = get_lat(da).name
    lon = get_lon(da).name

    # Rectilinear case (lat and lon are the 1D dimensions)
    if (lat in da.dims) or (lon in da.dims):
        if lat in da.dims and lat_bnds is not None:
            lat_bnds = _check_desc_coords(coord=da[lat], bounds=lat_bnds, dim=lat)
            da = da.sel({lat: slice(*lat_bnds)})

        if lon in da.dims and lon_bnds is not None:
            lon_bnds = _check_desc_coords(coord=da[lon], bounds=lon_bnds, dim=lon)
            da = da.sel({lon: slice(*lon_bnds)})
    # Locstream case (lat and lon are 1D, sharing the dimension)
    elif da[lat].ndim == 1 and da[lon].ndim == 1 and da[lon].dims == da[lat].dims:
        mask = (
            (da[lat] < max(lat_bnds))
            & (da[lat] > min(lat_bnds))
            & (da[lon] < max(lon_bnds))
            & (da[lon] > min(lon_bnds))
        )
        da = da.sel({da[lat].dims[0]: mask})

    # Curvilinear case (lat and lon are coordinates, not dimensions)
    elif ((lat in da.coords) and (lon in da.coords)) or (
        (lat in da.data_vars) and (lon in da.data_vars)
    ):
        # Define a bounding box along the dimensions
        # This is an optimization, a simple `where` would work but take longer for large hi-res grids.
        if lat_bnds is not None:
            lat_b = assign_bounds(lat_bnds, da[lat])
            lat_cond = in_bounds(lat_b, da[lat])
        else:
            lat_b = None
            lat_cond = True

        if lon_bnds is not None:
            lon_b = assign_bounds(lon_bnds, da[lon])
            lon_cond = in_bounds(lon_b, da[lon])
        else:
            lon_b = None
            lon_cond = True

        # Crop original array using slice, which is faster than `where`.
        ind = np.where(lon_cond & lat_cond)
        args = dict()

        for i, d in enumerate(da[lat].dims):
            try:
                coords = da[d][ind[i]]
                bnds = _check_desc_coords(
                    coord=da[d],
                    bounds=[coords.min().values, coords.max().values],
                    dim=d,
                )
            except ValueError:
                raise ValueError(
                    "There were no valid data points found in the requested subset. Please expand "
                    "the area covered by the bounding box."
                )
            args[d] = slice(*bnds)
        # If the dims of lat and lon do not have coords, sel defaults to isel,
        # and then the last element is not returned.
        da = da.sel(**args)

        if da[lat].size == 0 or da[lon].size == 0:
            raise ValueError(
                "There were no valid data points found in the requested subset. Please expand "
                "the area covered by the bounding box."
            )

        # Recompute condition on cropped coordinates
        if lat_bnds is not None:
            lat_cond = in_bounds(lat_b, da[lat])

        if lon_bnds is not None:
            lon_cond = in_bounds(lon_b, da[lon])

        # Mask coordinates outside the bounding box
        if isinstance(da, xarray.Dataset):
            # If da is a xr.DataSet Mask only variables that have the
            # same 2d coordinates as lat (or lon)
            for var in da.data_vars:
                if set(da[lat].dims).issubset(da[var].dims):
                    da[var] = da[var].where(lon_cond & lat_cond)
        else:
            da = da.where(lon_cond & lat_cond)

    else:
        raise (
            Exception(
                f'{subset_bbox.__name__} requires input data with "lon" and "lat" dimensions, coordinates, or variables.'
            )
        )

    if start_date or end_date:
        da = subset_time(da, start_date=start_date, end_date=end_date)

    elif time_values:
        da = subset_time_by_values(da, time_values=time_values)

    if first_level or last_level:
        da = subset_level(da, first_level=first_level, last_level=last_level)

    elif level_values:
        da = subset_level_by_values(da, level_values=level_values)

    if da[lat].size == 0 or da[lon].size == 0:
        raise ValueError(
            "There were no valid data points found in the requested subset. Please expand "
            "the area covered by the bounding box, the time period or the level range you have selected."
        )

    return da


def assign_bounds(
    bounds: tuple[Optional[float], Optional[float]], coord: xarray.DataArray
) -> tuple[Optional[float], Optional[float]]:
    """Replace unset boundaries by the minimum and maximum coordinates.

    Parameters
    ----------
    bounds : Tuple[Optional[float], Optional[float]]
        Boundaries.
    coord : xarray.DataArray
        Grid coordinates.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        Lower and upper grid boundaries.

    """
    if bounds[0] > bounds[1]:
        bounds = np.flip(bounds)
    bn, bx = bounds
    bn = bn if bn is not None else coord.min()
    bx = bx if bx is not None else coord.max()
    return bn, bx


def in_bounds(bounds: tuple[float, float], coord: xarray.DataArray) -> xarray.DataArray:
    """Check which coordinates are within the boundaries.

    Parameters
    ----------
    bounds : Tuple[float, float]
        Boundaries.
    coord : xarray.DataArray
        Grid coordinates.

    Returns
    -------
    xarray.DataArray

    """
    bn, bx = bounds
    return (coord >= bn) & (coord <= bx)


def _check_desc_coords(
    coord: xarray.Dataset,
    bounds: Union[tuple[float, float], list[np.ndarray]],
    dim: str,
) -> tuple[float, float]:
    """If Dataset coordinates are descending, and bounds are ascending, reverse bounds."""
    if np.all(coord.diff(dim=dim) < 0) and len(coord) > 1 and bounds[1] > bounds[0]:
        bounds = np.flip(bounds)
    return bounds


def _check_has_overlaps(polygons: gpd.GeoDataFrame) -> None:
    non_overlapping = []
    for n, p in enumerate(polygons["geometry"][:-1], 1):
        if not any(p.overlaps(g) for g in polygons["geometry"][n:]):
            non_overlapping.append(p)
    if len(polygons) != len(non_overlapping):
        logger.warning(
            "List of shapes contains overlap between features. Results will vary on feature order.",
            UserWarning,
            stacklevel=5,
        )


def _check_has_overlaps_old(polygons: gpd.GeoDataFrame) -> None:
    for i, (inda, pola) in enumerate(polygons.iterrows()):
        for indb, polb in polygons.iloc[i + 1 :].iterrows():
            if pola.geometry.intersects(polb.geometry):
                logger.warning(
                    f"List of shapes contains overlap between {inda} and {indb}. Points will be assigned to {inda}.",
                    UserWarning,
                    stacklevel=5,
                )


def _check_crs_compatibility(shape_crs: CRS, raster_crs: CRS) -> None:
    """If CRS definitions are not WGS84 or incompatible, raise operation warnings."""
    wgs84 = CRS(4326)
    if not shape_crs.equals(raster_crs):
        if (shape_crs.coordinate_system.name != raster_crs.coordinate_system.name) and (
            shape_crs.axis_info[0].unit_name != raster_crs.axis_info[0].unit_name
        ):
            raise CRSError(
                "CRS definitions are not compatible. Please ensure both are using the same coordinate system and units."
            )
        elif (
            "lon_wrap" in raster_crs.to_string()
            and "lon_wrap" not in shape_crs.to_string()
        ):
            logger.warning(
                "CRS definitions are similar but raster lon values must be wrapped.",
                UserWarning,
                stacklevel=3,
            )
        elif not shape_crs.equals(wgs84) and not raster_crs.equals(wgs84):
            logger.warning(
                "CRS definitions are not similar or both not using WGS84 datum. Tread with caution.",
                UserWarning,
                stacklevel=3,
            )


@check_lons
@convert_lat_lon_to_da
def subset_gridpoint(
    da: Union[xarray.DataArray, xarray.Dataset],
    lon: Optional[Union[float, Sequence[float], xarray.DataArray]] = None,
    lat: Optional[Union[float, Sequence[float], xarray.DataArray]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    first_level: Optional[Union[float, int]] = None,
    last_level: Optional[Union[float, int]] = None,
    tolerance: Optional[float] = None,
    add_distance: bool = False,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Extract one or more of the nearest gridpoint(s) from datarray based on lat lon coordinate(s).

    Return a subsetted data array (or Dataset) for the grid point(s) falling nearest the input longitude and latitude
    coordinates. Optionally subset the data array for years falling within provided date bounds.
    Time series can optionally be subsetted by dates.
    If 1D sequences of coordinates are given, the gridpoints will be concatenated along the new dimension "site".

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
        Input data.
    lon : Optional[Union[float, Sequence[float], xarray.DataArray]]
        Longitude coordinate(s). Must be of the same length as lat.
    lat : Optional[Union[float, Sequence[float], xarray.DataArray]]
        Latitude coordinate(s). Must be of the same length as lon.
    start_date : Optional[str]
        Start date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to first day of input data-array.
    end_date : Optional[str]
        End date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to last day of input data-array.
    first_level : Optional[Union[int, float]]
        First level of the subset.
        Can be either an integer or float.
        Defaults to first level of input data-array.
    last_level : Optional[Union[int, float]]
        Last level of the subset.
        Can be either an integer or float.
        Defaults to last level of input data-array.
    tolerance : Optional[float]
        Masks values if the distance to the nearest gridpoint is larger than tolerance in meters.
    add_distance: bool

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
         Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_gridpoint

        ds = xr.open_dataset(path_to_pr_file)

        # Subset lat lon point
        prSub = subset_gridpoint(ds.pr, lon=-75, lat=45)

        # Subset multiple variables in a single dataset
        ds = xr.open_mfdataset([path_to_tasmax_file, path_to_tasmin_file])
        dsSub = subset_gridpoint(ds, lon=-75, lat=45)
    """
    if lat is None or lon is None:
        raise ValueError("Insufficient coordinates provided to locate grid point(s).")

    ptdim = lat.dims[0]

    lon_name = lon.name or "lon"
    lat_name = lat.name or "lat"

    # make sure input data has 'lon' and 'lat'(dims, coordinates, or data_vars)
    if hasattr(da, lon_name) and hasattr(da, lat_name):
        dims = list(da.dims)

        # if 'lon' and 'lat' are present as data dimensions use the .sel method.
        if lat_name in dims and lon_name in dims:
            da = da.sel(lat=lat, lon=lon, method="nearest")

            if tolerance is not None or add_distance:
                # Calculate the geodesic distance between grid points and the point of interest.
                dist = distance(da, lon=lon, lat=lat)
            else:
                dist = None

        else:
            # Calculate the geodesic distance between grid points and the point of interest.
            dist = distance(da, lon=lon, lat=lat)
            pts = []
            dists = []
            for site in dist[ptdim]:
                # Find the indices for the closest point
                inds = np.unravel_index(
                    dist.sel({ptdim: site}).argmin(), dist.sel({ptdim: site}).shape
                )

                # Select data from closest point
                args = {xydim: ind for xydim, ind in zip(dist.dims, inds)}
                pts.append(da.isel(**args))
                dists.append(dist.isel(**args))
            da = xarray.concat(pts, dim=ptdim)
            dist = xarray.concat(dists, dim=ptdim)
    else:
        raise (
            Exception(
                f'{subset_gridpoint.__name__} requires input data with "lon" and "lat" coordinates or data variables.'
            )
        )

    if tolerance is not None and dist is not None:
        da = da.where(dist < tolerance)

    if add_distance:
        da = da.assign_coords(distance=dist)

    if len(lat) == 1:
        da = da.squeeze(ptdim)

    if start_date or end_date:
        da = subset_time(da, start_date=start_date, end_date=end_date)

    if first_level or last_level:
        da = subset_level(da, first_level=first_level, last_level=last_level)

    return da


@check_start_end_dates
def subset_time(
    da: Union[xarray.DataArray, xarray.Dataset],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset input DataArray or Dataset based on start and end years.

    Return a subset of a DataArray or Dataset for dates falling within the provided bounds.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
        Input data.
    start_date : Optional[str]
        Start date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to first day of input data-array.
    end_date : Optional[str]
        End date of the subset.
        Date string format -- can be year ("%Y"), year-month ("%Y-%m") or year-month-day("%Y-%m-%d").
        Defaults to last day of input data-array.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_time

        ds = xr.open_dataset(path_to_pr_file)

        # Subset complete years
        prSub = subset_time(ds.pr, start_date="1990", end_date="1999")

        # Subset single complete year
        prSub = subset_time(ds.pr, start_date="1990", end_date="1990")

        # Subset multiple variables in a single dataset
        ds = xr.open_mfdataset([path_to_tasmax_file, path_to_tasmin_file])
        dsSub = subset_time(ds, start_date="1990", end_date="1999")

        # Subset with year-month precision - Example subset 1990-03-01 to 1999-08-31 inclusively
        txSub = subset_time(ds.tasmax, start_date="1990-03", end_date="1999-08")

        # Subset with specific start_dates and end_dates
        tnSub = subset_time(ds.tasmin, start_date="1990-03-13", end_date="1990-08-17")

    Notes
    -----
    TODO add notes about different calendar types. Avoid "%Y-%m-31". If you want complete month use only "%Y-%m".
    """
    return da.sel(time=slice(start_date, end_date))


@check_datetimes_exist
def subset_time_by_values(
    da: Union[xarray.DataArray, xarray.Dataset],
    time_values: Optional[Sequence[str]] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset input DataArray or Dataset based on a sequence of datetime strings.

    Return a subset of a DataArray or Dataset for datetimes matching those requested.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
        Input data.
    time_values: Optional[Sequence[str]]
        Values for time. Default: ``None``

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_time_by_values

        ds = xr.open_dataset(path_to_pr_file)

        # Subset a selection of datetimes
        times = ["2015-01-01", "2018-12-05", "2021-06-06"]
        prSub = subset_time_by_values(ds.pr, time_values=times)

    Notes
    -----
    If any datetimes are not found, a ValueError will be raised.
    The requested datetimes will automatically be re-ordered to match the order in the
    input dataset.
    """
    return da.sel(time=time_values)


def subset_time_by_components(
    da: Union[xarray.DataArray, xarray.Dataset],
    *,
    time_components: Union[dict, None] = None,
) -> xarray.DataArray:
    """Subsets by one or more time components (year, month, day etc).

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
      Input data.
    time_components: Union[Dict, None] = None

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_time_by_components

        # To select all Winter months (Dec, Jan, Feb) or [12, 1, 2]:
        da = xr.open_dataset(path_to_file).pr
        winter_dict = {"month": [12, 1, 2]}
        res = subset_time_by_components(da, time_components=winter_dict)
    """
    # Create a set of indices that match the requested time components
    req_indices = set(range(len(da.time.values)))

    for t_comp in ("year", "month", "day", "hour", "minute", "second"):
        req_t_comp = time_components.get(t_comp, [])

        # Exclude any time component that has not been requested
        if not req_t_comp:
            continue

        t_comp_indices = da.groupby(f"time.{t_comp}").groups
        req_indices = req_indices.intersection(
            {idx for tc in req_t_comp for idx in t_comp_indices.get(tc, [])}
        )
    if not req_indices:
        raise KeyError("No timesteps are matching the selection criteria.")

    return da.isel(time=sorted(req_indices))


@check_start_end_levels
def subset_level(
    da: Union[xarray.DataArray, xarray.Dataset],
    first_level: Optional[Union[int, float, str]] = None,
    last_level: Optional[Union[int, float, str]] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset input DataArray or Dataset based on first and last levels.

    Return a subset of a DataArray or Dataset for levels falling within the provided bounds.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
        Input data.
    first_level : Optional[Union[int, float, str]]
        First level of the subset (specified as the value, not the index).
        Can be either an integer or float.
        Defaults to first level of input data-array.
    last_level : Optional[Union[int, float, str]]
        Last level of the subset (specified as the value, not the index).
        Can be either an integer or float.
        Defaults to last level of input data-array.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_level

        ds = xr.open_dataset(path_to_pr_file)

        # Subset complete levels
        prSub = subset_level(ds.pr, first_level=0, last_level=30)

        # Subset single level
        prSub = subset_level(ds.pr, first_level=1000, last_level=1000)

        # Subset multiple variables in a single dataset
        ds = xr.open_mfdataset([path_to_tasmax_file, path_to_tasmin_file])
        dsSub = subset_time(ds, first_level=1000.0, last_level=850.0)

    Notes
    -----
    TBA
    """
    level = da[get_coord_by_type(da, "level")]

    first_level, last_level = _check_desc_coords(
        level, (first_level, last_level), level.name
    )

    da = da.sel(**{level.name: slice(first_level, last_level)})

    return da


@check_levels_exist
def subset_level_by_values(
    da: Union[xarray.DataArray, xarray.Dataset],
    level_values: Optional[Union[Sequence[float], Sequence[int]]] = None,
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Subset input DataArray or Dataset based on a sequence of vertical level values.

    Return a subset of a DataArray or Dataset for levels matching those requested.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
        Input data.
    level_values : Optional[Union[Sequence[float], Sequence[int]]]
        A list of level values to select.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        Subsetted xarray.DataArray or xarray.Dataset

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import subset_level_by_values

        ds = xr.open_dataset(path_to_pr_file)

        # Subset a selection of levels
        levels = [1000.0, 850.0, 250.0, 100.0]
        prSub = subset_level_by_values(ds.pr, level_values=levels)

    Notes
    -----
    If any levels are not found, a ValueError will be raised.
    The requested levels will automatically be re-ordered to match the order in the
    input dataset.
    """
    level = da[get_coord_by_type(da, "level")]
    return da.sel(**{level.name: level_values}, method="nearest")


@convert_lat_lon_to_da
def distance(
    da: Union[xarray.DataArray, xarray.Dataset],
    *,
    lon: Union[float, Sequence[float], xarray.DataArray],
    lat: Union[float, Sequence[float], xarray.DataArray],
) -> Union[xarray.DataArray, xarray.Dataset]:
    """Return distance to a point in meters.

    Parameters
    ----------
    da : Union[xarray.DataArray, xarray.Dataset]
        Input data.
    lon : Union[float, Sequence[float], xarray.DataArray]
        Longitude coordinate.
    lat : Union[float, Sequence[float], xarray.DataArray]
        Latitude coordinate.

    Returns
    -------
    xarray.DataArray
        Distance in meters to point.

    Examples
    --------
    .. code-block:: python

        import xarray as xr
        from clisops.core.subset import distance

        # To get the indices from the closest point, use:
        da = xr.open_dataset(path_to_pr_file).pr
        d = distance(da, lon=-75, lat=45)
        k = d.argmin()
        i, j, _ = np.unravel_index(k, d.shape)
    """
    ptdim = lat.dims[0]

    g = Geod(ellps="WGS84")  # WGS84 ellipsoid - decent globally

    def func(lons, lats, lon, lat):
        return g.inv(lons, lats, lon, lat)[2]

    out = xarray.apply_ufunc(
        func,
        *xarray.broadcast(da.lon.load(), da.lat.load(), lon, lat),
        input_core_dims=[[ptdim]] * 4,
        output_core_dims=[[ptdim]],
    )
    out.attrs["units"] = "m"
    return out
