import logging

from dateutil import parser as date_parser

from .exceptions import InvalidParameterValue, MissingParameterValue


def parse_date(text):
    parsed = date_parser.parse(text)
    return parsed.isoformat().split("T")[0]


def parse_date_year(text):
    return parse_date(text).split("-")[0]


def _map_time(time):
    try:
        if len(time) != 2:
            raise ValueError("expecting start and end time")
        start_date = parse_date_year(time[0])
        end_date = parse_date_year(time[1])
    except Exception:
        msg = f"time parameter is not valid: {time}"
        logging.error(msg, exc_info=True)
        raise InvalidParameterValue(msg)
    return dict(start_date=start_date, end_date=end_date)


def _map_space(space):
    try:
        if len(space) != 4:
            raise ValueError("expecting bbox")
        lon_bnds = (float(space[0]), float(space[2]))
        lat_bnds = (float(space[1]), float(space[3]))
    except Exception:
        msg = f"space parameter is not valid: {space}"
        logging.error(msg, exc_info=True)
        raise InvalidParameterValue(msg)
    return dict(lon_bnds=lon_bnds, lat_bnds=lat_bnds)


def map_params(time=None, space=None, level=None):
    args = dict()
    if time:
        args.update(_map_time(time))
    if space:
        args.update(_map_space(space))
    if level:
        # TODO: level is missing
        pass
    if not time and not space:
        raise MissingParameterValue("missing either time or space parameter.")
    return args
