"""Average module."""
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue
from roocs_utils.xarray_utils.xarray_utils import get_coord_type, known_coord_types

__all__ = ["average_over_dims", "average_shape"]


def average_shape(
    ds: Union[xr.DataArray, xr.Dataset],
    shape: Union[str, Path, gpd.GeoDataFrame],
) -> Union[xr.DataArray, xr.Dataset]:
    """Average a DataArray or Dataset spatially using vector shapes.

    Return a DataArray or Dataset averaged over each Polygon given.
    Requires xESMF >= 0.5.0.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
      Input values, coordinates naming must be compatible with xESMF.
    shape : Union[str, Path, gpd.GeoDataFrame]
      Path to shape file, or directly a geodataframe. Supports formats compatible with geopandas.
      Will be converted to EPSG:4326 if needed.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
      `ds` spatially-averaged over the polygon(s) in `shape`.
      Has a new `geom` dimension corresponding to the index of the input geodataframe.
      Non-geometry columns of the geodataframe are copied as auxiliary coordinates.

    Notes
    -----
    The spatial weights are computed with ESMF, which uses corners given in lat/lon format (EPSG:4326),
    the input dataset `ds` most provide those. In opposition to `subset.subset_shape`, the
    weights computed here take partial overlaps and holes into account.

    As xESMF computes the weight masks only once, skipping missing values is not really feasible. Thus,
    all NaNs propagate when performing the average.

    Examples
    --------
    >>> import xarray as xr  # doctest: +SKIP
    >>> from clisops.core.average import average_shape  # doctest: +SKIP
    >>> pr = xr.open_dataset(path_to_pr_file).pr  # doctest: +SKIP
    ...
    # Average data array over shape
    >>> prAvg = average_shape(pr, shape=path_to_shape_file)  # doctest: +SKIP
    ...
    # Average multiple variables in a single dataset
    >>> ds = xr.open_mfdataset([path_to_tasmin_file, path_to_tasmax_file])  # doctest: +SKIP
    >>> dsAvg = average_shape(ds, shape=path_to_shape_file)  # doctest: +SKIP
    """
    try:
        from xesmf import SpatialAverager
    except ImportError:
        raise ValueError("Package xesmf >= 0.5.0 is required to use average_shape")

    if isinstance(ds, xr.DataArray):
        ds_copy = ds._to_temp_dataset()
    else:
        ds_copy = ds.copy()

    if isinstance(shape, gpd.GeoDataFrame):
        poly = shape.copy()
    else:
        poly = gpd.GeoDataFrame.from_file(shape)

    if poly.crs is not None:
        poly = poly.to_crs(4326)

    savger = SpatialAverager(ds_copy, poly.geometry)
    if savger.weights.nnz == 0:
        raise ValueError(
            "There were no valid data points found in the requested averaging region. Verify objects overlap."
        )
    ds_out = savger(ds_copy)

    # Set geom coords to poly's index
    ds_out["geom"] = poly.index

    # other info in poly
    ds_meta = (
        poly.drop("geometry", axis=1)
        .to_xarray()
        .rename(**{poly.index.name or "index": "geom"})
    )
    ds_meta = ds_meta.set_coords(ds_meta.data_vars)

    ds_out = xr.merge([ds_out, ds_meta])

    if isinstance(ds, xr.DataArray):
        return ds._from_temp_dataset(ds_out)
    return ds_out


def average_over_dims(
    ds: Union[xr.DataArray, xr.Dataset],
    dims: Tuple[str] = None,
    ignore_undetected_dims: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Average a DataArray or Dataset over the dimensions specified.

    Parameters
    ----------
    ds : Union[xr.DataArray, xr.Dataset]
      Input values.
    dims : Tuple[str] = None
      The dimensions over which to apply the average. If None, none of the dimensions are averaged over. Dimensions
      must be one of ["time", "level", "latitude", "longitude"].
    ignore_undetected_dims: bool
      If the dimensions specified are not found in the dataset, an Exception will be raised if set to True.
      If False, an exception will not be raised and the other dimensions will be averaged over. Default = False

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
      New Dataset or DataArray object averaged over the indicated dimensions.
      The indicated dimensions will have been removed.

    Examples
    --------
    >>> from clisops.core.average import average_over_dims  # doctest: +SKIP
    >>> pr = xr.open_dataset(path_to_pr_file).pr  # doctest: +SKIP
    ...
    # Average data array over latitude and longitude
    >>> prAvg = average_over_dims(pr, dims=['latitude', 'longitude'], ignore_undetected_dims=True)  # doctest: +SKIP
    """

    if not dims:
        raise InvalidParameterValue(
            "At least one dimension for averaging must be provided"
        )

    if not set(dims).issubset(set(known_coord_types)):
        raise InvalidParameterValue(
            f"Dimensions for averaging must be one of {known_coord_types}"
        )

    found_dims = dict()

    # Work out real coordinate types for each dimension
    for coord in ds.coords:
        coord = ds.coords[coord]
        coord_type = get_coord_type(coord)

        if coord_type:
            # Check if the coordinate is a dimension
            if coord.name in ds.dims:
                found_dims[coord_type] = coord.name

    # Set a variable for requested dimensions that were not detected
    undetected_dims = set(dims) - set(found_dims.keys())

    if ignore_undetected_dims is False and not set(dims).issubset(
        set(found_dims.keys())
    ):
        raise InvalidParameterValue(
            f"Requested dimensions were not found in input dataset: {undetected_dims}."
        )
    else:
        # Remove undetected dim so that it can be ignored
        dims = [dim for dim in dims if dim not in undetected_dims]

    # Get dims by the name used in dataset
    dims_to_average = []

    for dim in dims:
        dims_to_average.append(found_dims[dim])

    # The mean will be carried out on a Dataset or DataArray
    # Calculate the mean, skip missing values and retain original attributes

    # Short-term solution to error: "NotImplementedError: Computing the mean of an " ...
    #    "array containing cftime.datetime objects is not yet implemented on dask arrays."
    # See GITHUB ISSUE: https://github.com/roocs/clisops/issues/185
    # The fix is simply to force `ds.load()` before processing
    ds_averaged_over_dims = ds.load().mean(
        dim=dims_to_average, skipna=True, keep_attrs=True
    )

    return ds_averaged_over_dims
