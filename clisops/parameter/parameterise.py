"""Parameterise inputs to roocs parameter classes."""

import xarray as xr

from clisops.parameter import (
    area_parameter,
    collection_parameter,
    level_parameter,
    time_components_parameter,
    time_parameter,
)


def parameterise(collection=None, area=None, level=None, time=None, time_components=None):
    """
    Parameterise inputs to instances of parameter classes, allowing them to be used throughout roocs.

    For supported formats for each input, please see their individual classes.

    Parameters
    ----------
    collection : str, Path, xr.DataArray, xr.Dataset, or any other supported format
        Input collection to be parameterised.
    area : str, Path, dict, or any other supported format
        Input area to be parameterised.
    level : str, Path, dict, or any other supported format
        Input level to be parameterised.
    time : str, Path, dict, or any other supported format
        Input time to be parameterised.
    time_components : str, Path, dict, or any other supported format
        Input time components to be parameterised.

    Returns
    -------
    dict
        A dictionary containing the parameterised inputs as instances of their respective classes.
    """
    # if a collection is a Dataset/DataArray, it doesn't need to be parameterised
    if type(collection) not in (xr.core.dataarray.DataArray, xr.core.dataset.Dataset):
        collection = collection_parameter.CollectionParameter(collection)

    area = area_parameter.AreaParameter(area)
    level = level_parameter.LevelParameter(level)
    time = time_parameter.TimeParameter(time)
    time_components = time_components_parameter.TimeComponentsParameter(time_components)

    return locals()
