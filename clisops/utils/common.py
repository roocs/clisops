from roocs_utils.parameter import parameterise


def map_params(ds, time=None, area=None, level=None):
    args = dict()

    parameters = parameterise(collection=ds, time=time, area=area, level=level)

    for parameter in ["time", "area", "level"]:

        if parameters.get(parameter).tuple is not None:
            args.update(parameters.get(parameter).asdict())

    # rename start_time and end_time to start_date and end_date to
    # match clisops/core/subset
    if "start_time" in args:
        args["start_date"] = args.pop("start_time")

    if "end_time" in args:
        args["end_date"] = args.pop("end_time")

    return args
