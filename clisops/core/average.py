"""Average module."""
from typing import Tuple, Union

import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue
from roocs_utils.xarray_utils.xarray_utils import get_coord_type, known_coord_types

__all__ = [
    "average_over_dims",
]


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
    >>> import xarray as xr  # doctest: +SKIP
    >>> from clisops.core.average import average_over_dims  # doctest: +SKIP
    >>> pr = xr.open_dataset(path_to_pr_file).pr  # doctest: +SKIP
    ...
    # Average data array over latitude and longitude
    >>> prAvg = average_over_dims(pr, dims=['latitude', 'longitude'], ignore_undetected_dims=True)  # doctest: +SKIP
    """

    if not dims:
        return ds

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
            # check if the coordinate is a dimension
            if coord.name in ds.dims:
                found_dims[coord_type] = coord.name

    undetected_dims = set(dims) - set(found_dims.keys())
    if ignore_undetected_dims is False and not set(dims).issubset(
        set(found_dims.keys())
    ):
        raise InvalidParameterValue(
            f"Requested dimensions were not found in input dataset: {undetected_dims}."
        )
    else:
        # remove undetected dim so that it can be ignored
        dims = [dim for dim in dims if dim not in undetected_dims]

    # get dims by the name used in dataset
    dims_to_average = []
    for dim in dims:
        dims_to_average.append(found_dims[dim])

    # mean can be carried out on a Dataset or DataArray
    ds_averaged_over_dims = ds.mean(dim=dims_to_average, skipna=True, keep_attrs=True)

    return ds_averaged_over_dims
