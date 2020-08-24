import logging

from dateutil import parser as date_parser
from roocs_utils.parameter import parameterise

from .exceptions import InvalidParameterValue, MissingParameterValue


def map_params(time=None, area=None, level=None):
    args = dict()

    parameters = parameterise.parameterise(time=time, area=area, level=level)

    for parameter in ["time", "area"]:  # , 'level']: # level not implemented yet

        if parameters.get(parameter).tuple is not None:
            args.update(parameters.get(parameter).asdict())

    return args
