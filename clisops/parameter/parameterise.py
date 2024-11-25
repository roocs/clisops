import xarray as xr

from clisops.parameter import (
    area_parameter,
    collection_parameter,
    level_parameter,
    time_components_parameter,
    time_parameter,
)


def parameterise(
    collection=None, area=None, level=None, time=None, time_components=None
):
    """
    Parameterises inputs to instances of parameter classes which allows
    them to be used throughout roocs.
    For supported formats for each input please see their individual classes.

    :param collection: Collection input in any supported format.
    :param area: Area input in any supported format.
    :param level: Level input in any supported format.
    :param time: Time input in any supported format.
    :param time_components: Time Components input in any supported format.
    :return: Parameters as instances of their respective classes.
    """

    # if collection is a dataset/dataarray it doesn't need to be parameterised
    if type(collection) not in (xr.core.dataarray.DataArray, xr.core.dataset.Dataset):
        collection = collection_parameter.CollectionParameter(collection)

    area = area_parameter.AreaParameter(area)
    level = level_parameter.LevelParameter(level)
    time = time_parameter.TimeParameter(time)
    time_components = time_components_parameter.TimeComponentsParameter(time_components)

    return locals()
