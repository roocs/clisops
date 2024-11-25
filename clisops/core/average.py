"""Average module."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import cf_xarray  # noqa
import geopandas as gpd
import numpy as np
import xarray as xr

from clisops.core.regrid import XESMF_MINIMUM_VERSION
from clisops.core.subset import shape_bbox_indexer
from clisops.exceptions import InvalidParameterValue
from clisops.utils.dataset_utils import (
    get_coord_by_type,
    get_coord_type,
    known_coord_types,
)
from clisops.utils.time_utils import create_time_bounds

__all__ = ["average_over_dims", "average_shape", "average_time"]

# see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
freqs = {"day": "1D", "month": "1MS", "year": "1YS"}


def average_shape(
    ds: xr.Dataset,
    shape: Union[str, Path, gpd.GeoDataFrame],
    variable: Union[str, Sequence[str]] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """Average a DataArray or Dataset spatially using vector shapes.

    Return a DataArray or Dataset averaged over each Polygon given. Requires xESMF.

    Parameters
    ----------
    ds : xarray.Dataset
        Input values, coordinate attributes must be CF-compliant.
    shape : Union[str, Path, gpd.GeoDataFrame]
        Path to shape file, or directly a GeoDataFrame. Supports formats compatible with geopandas.
        Will be converted to EPSG:4326 if needed.
    variable : Union[str, Sequence[str], None]
        Variables to average. If None, average over all data variables.

    Returns
    -------
    Union[xarray.DataArray, xarray.Dataset]
        `ds` spatially-averaged over the polygon(s) in `shape`.
        Has a new `geom` dimension corresponding to the index of the input GeoDataFrame.
        Non-geometry columns of the GeoDataFrame are copied as auxiliary coordinates.

    Notes
    -----
    The spatial weights are computed with ESMF, which uses corners given in lat/lon format (EPSG:4326),
    the input dataset `ds` must provide those. In opposition to `subset.subset_shape`, the
    weights computed here take partial overlaps and holes into account.

    As xESMF computes the weight masks only once, skipping missing values is not really feasible. Thus,
    all NaNs propagate when performing the average.

    Examples
    --------
    .. code-block:: python

        import xarray as xr  # doctest: +SKIP
        from clisops.core.average import average_shape

        pr = xr.open_dataset(path_to_pr_file).pr

        # Average data array over shape
        prAvg = average_shape(pr, shape=path_to_shape_file)

        # Average multiple variables in a single dataset
        ds = xr.open_mfdataset([path_to_tasmin_file, path_to_tasmax_file])
        dsAvg = average_shape(ds, shape=path_to_shape_file)
    """
    try:
        from xesmf import SpatialAverager
    except ImportError:
        raise ValueError(
            f"Package xesmf {XESMF_MINIMUM_VERSION} is required to use `average_shape`."
        )

    if isinstance(ds, xr.DataArray):
        warnings.warn(
            "Pass a Dataset object instead of a DataArray.", DeprecationWarning
        )
        ds_copy = ds.to_dataset(name=ds.name)
    else:
        ds_copy = ds.copy()

    if isinstance(shape, gpd.GeoDataFrame):
        poly = shape.copy()
    else:
        poly = gpd.GeoDataFrame.from_file(shape)

    if poly.crs is not None:
        poly = poly.to_crs(4326)

    # First subset to bounding box to reduce memory usage.
    indexer = shape_bbox_indexer(ds_copy, poly)
    ds_sub = ds_copy.isel(indexer)

    # Compute the weights
    savger = SpatialAverager(ds_sub, poly.geometry)

    # Check that some weights are not null. Handle both sparse and scipy weights.
    nonnull = (
        savger.weights.data.nnz
        if isinstance(savger.weights, xr.DataArray)
        else savger.weights.nnz
    )
    if nonnull == 0:
        raise ValueError(
            "There were no valid data points found in the requested averaging region. Verify objects overlap."
        )

    # Select variables to average
    if variable is not None:
        ds_sub = ds_sub[variable]

    # Apply the weights to the actual data -> spatial average
    # We transfer the global and variable attributes of the input to the output
    ds_out = savger(ds_sub, keep_attrs=True)

    # Set geom coords to poly's index
    ds_out["geom"] = poly.index

    # Add polygon attributes to Dataset output as coordinates
    ds_meta = (
        poly.drop("geometry", axis=1)
        .to_xarray()
        .rename(**{poly.index.name or "index": "geom"})
    )
    ds_meta = ds_meta.set_coords(ds_meta.data_vars)
    ds_out = xr.merge([ds_out, ds_meta])

    # Maybe returning a DataArray should be deprecated.
    if isinstance(ds, xr.DataArray):
        return ds_out[ds.name]
    return ds_out


def average_over_dims(
    ds: Union[xr.DataArray, xr.Dataset],
    dims: Sequence[str] = None,
    ignore_undetected_dims: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """Average a DataArray or Dataset over the dimensions specified.

    Parameters
    ----------
    ds : Union[xr.DataArray, xr.Dataset]
        Input values.
    dims : Sequence[{"time", "level", "latitude", "longitude"}]
        The dimensions over which to apply the average. If None, none of the dimensions are averaged over.
        Dimensions must be one of ["time", "level", "latitude", "longitude"].
    ignore_undetected_dims : bool
        If the dimensions specified are not found in the dataset, an Exception will be raised if set to True.
        If False, an exception will not be raised and the other dimensions will be averaged over. Default = False

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        New Dataset or DataArray object averaged over the indicated dimensions.
        The indicated dimensions will have been removed.

    Examples
    --------
    .. code-block:: python

        from clisops.core.average import average_over_dims

        pr = xr.open_dataset(path_to_pr_file).pr

        # Average data array over latitude and longitude
        prAvg = average_over_dims(
            pr, dims=["latitude", "longitude"], ignore_undetected_dims=True
        )
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
    if isinstance(ds, xr.Dataset):
        untouched_ds = ds.drop_dims(dims_to_average)
        ds = ds.drop_vars(untouched_ds.data_vars.keys())

    ds_averaged_over_dims = ds.mean(dim=dims_to_average, skipna=True, keep_attrs=True)

    if isinstance(ds, xr.Dataset):
        return xr.merge((ds_averaged_over_dims, untouched_ds))
    return ds_averaged_over_dims


def average_time(
    ds: Union[xr.DataArray, xr.Dataset],
    freq: str,
) -> Union[xr.DataArray, xr.Dataset]:
    """Average a DataArray or Dataset over the time frequency specified.

    Parameters
    ----------
    ds : Union[xr.DataArray, xr.Dataset]
      Input values.
    freq : str
      The frequency to average over. One of "month", "year".

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
      New Dataset or DataArray object averaged over the indicated time frequency.

    Examples
    --------
    .. code-block:: python

        from clisops.core.average import average_time

        pr = xr.open_dataset(path_to_pr_file).pr

        # Average data array over each month
        prAvg = average_time(pr, freq="month")
    """

    if not freq:
        raise InvalidParameterValue(
            "At least one frequency for averaging must be provided"
        )

    if freq not in list(freqs.keys()):
        raise InvalidParameterValue(
            f"Time frequency for averaging must be one of {list(freqs.keys())}."
        )

    # check time coordinate exists and get name
    t = get_coord_by_type(ds, "time", ignore_aux_coords=False)
    if t is None:
        raise Exception("Time dimension could not be found")
    else:
        t = ds[t]

    # resample and average over time
    ds_t_avg = ds.resample(indexer={t.name: freqs[freq]}).mean(
        dim=t.name, skipna=True, keep_attrs=True
    )

    # generate time_bounds depending on frequency
    time_bounds = create_time_bounds(ds_t_avg, freq)

    # get name of bounds dimension for time
    bnds = ds.cf.get_bounds_dim_name("time")

    # add time bounds to dataset
    ds_t_avg = ds_t_avg.assign({"time_bnds": ((t.name, bnds), np.asarray(time_bounds))})

    return ds_t_avg
