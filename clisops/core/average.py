"""Average module."""

from roocs_utils.xarray_utils.xarray_utils import (
    get_coord_type,
    get_main_variable,
    known_coord_types,
)

__all__ = [
    "average_over_dims",
]


# put in static typing
def average_over_dims(ds, dims=None, ignore_unfound_dims=False):
    if not dims:
        return ds

    if not set(dims).issubset(set(known_coord_types)):
        raise Exception(
            f"Unknown dimension requested for averaging, must be within: {known_coord_types}."
        )

    found_dims = dict()

    # Work out real coordinate types for each dimension
    for coord in ds.coords:
        coord = ds.coords[coord]
        coord_type = get_coord_type(coord)
        if coord_type:
            found_dims[coord_type] = coord.name

    if ignore_unfound_dims is False and not set(dims).issubset(set(found_dims.keys())):
        raise Exception(
            f"Requested dimensions were not found in input dataset: {set(dims) - set(found_dims.keys())}."
        )  # this isn't quite right

    # get dims by the name used in dataset
    dims_to_average = []
    for dim in dims:
        dims_to_average.append(found_dims[dim])

    # check if already data array
    var_id = get_main_variable(ds)
    da = ds.to_array(dim=var_id)
    ds_averaged_over_dims = da.mean(dim=dims_to_average, keep_attrs=True)

    return ds_averaged_over_dims
