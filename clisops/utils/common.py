import logging

from dateutil import parser as date_parser
from roocs_utils.parameter import parameterise

from ..exceptions import InvalidParameterValue, MissingParameterValue


def map_params(ds, time=None, area=None, level=None):
    args = dict()
    # import pdb;pdb.set_trace()
    parameters = parameterise.parameterise(
        collection=ds, time=time, area=area, level=level
    )

    for parameter in ["time", "area"]:  # , 'level']: # level not implemented yet

        if parameters.get(parameter).tuple is not None:
            args.update(parameters.get(parameter).asdict())

    # rename start_time and end_time to start_date and end_date to
    # match clisops/core/subset
    if "start_time" in args:
        args["start_date"] = args.pop("start_time")

    if "end_time" in args:
        args["end_date"] = args.pop("end_time")

    return args
