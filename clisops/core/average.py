import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import xarray as xarray


__all__ = [
    "average_shape"
]


def average_shape(
    ds: Union[xarray.DataArray, xarray.Dataset],
    shape: Union[str, Path, gpd.GeoDataFrame],
) -> Union[xarray.DataArray, xarray.Dataset]:
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
        raise ValueError('Package xesmf >= 0.5.0 is required to use average_shape')

    if isinstance(ds, xarray.DataArray):
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
    ds_out = savger(ds_copy)

    # Set geom coords to poly's index
    ds_out['geom'] = poly.index

    # other info in poly
    ds_meta = poly.drop('geometry', axis=1).to_xarray().rename(**{poly.index.name or 'index': 'geom'})
    ds_meta = ds_meta.set_coords(ds_meta.data_vars)

    ds_out = xarray.merge([ds_out, ds_meta])

    if isinstance(ds, xarray.DataArray):
        return ds._from_temp_dataset(ds_out)
    return ds_out
