import logging

from dateutil import parser as date_parser
from roocs_utils.parameter import parameterise

from .exceptions import InvalidParameterValue, MissingParameterValue


def map_params(time=None, area=None, level=None):
    args = dict()

    area, time, level = parameterise.parametrise_clisops(
        time=time, level=level, area=area
    )

    if time:
        args.update(time.asdict())
    if area:
        args.update(area.asdict())
    if level:
        # TODO: level is missing
        pass
    return args
