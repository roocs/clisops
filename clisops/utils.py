import logging

from dateutil import parser as date_parser
from roocs_utils.parameter import parameterise

from .exceptions import InvalidParameterValue, MissingParameterValue


def map_params(time=None, area=None, level=None):
    args = dict()
    area, time, level = parameterise.parametrise_clisops(
        time=time, area=area, level=level
    )

    if time:
        args.update(time)
    if area:
        args.update(area)
    if level:
        # TODO: level is missing
        pass
    return args
