import calendar
from collections.abc import Sequence

from clisops.exceptions import InvalidParameterValue
from clisops.utils.file_utils import FileMapper
from clisops.utils.time_utils import str_to_AnyCalendarDateTime

__all__ = [
    "Interval",
    "Series",
    "TimeComponents",
    "area",
    "collection",
    "dimensions",
    "interval",
    "level_interval",
    "level_series",
    "parse_datetime",
    "parse_range",
    "parse_sequence",
    "series",
    "string_to_dict",
    "time_components",
    "time_interval",
    "time_series",
    "to_float",
]

# Global variables that are generally useful
month_map = {name.lower(): num for num, name in enumerate(calendar.month_abbr) if num}

time_comp_limits = {
    "year": None,
    "month": (1, 12),
    "day": (1, 40),  # allowing for strange calendars
    "hour": (0, 23),
    "minute": (0, 59),
    "second": (0, 59),
}


# A set of simple parser functions
def parse_range(x, caller):
    if isinstance(x, Sequence) and len(x) == 1:
        x = x[0]

    if x in ("/", None, ""):
        start = None
        end = None

    elif isinstance(x, str):
        if "/" not in x:
            raise InvalidParameterValue(
                f"{caller} should be passed in as a range separated by /"
            )

        # empty string either side of '/' is converted to None
        start, end = (i.strip() or None for i in x.split("/"))

    elif isinstance(x, Sequence):
        if len(x) != 2:
            raise InvalidParameterValue(
                f"{caller} should be a range. Expected 2 values, " f"received {len(x)}"
            )

        start, end = x

    else:
        raise InvalidParameterValue(f"{caller} is not in an accepted format")
    return start, end


def parse_sequence(x, caller):
    if x in (None, ""):
        sequence = None

    # check str or bytes
    elif isinstance(x, (str, bytes)):
        sequence = [i.strip() for i in x.strip().split(",")]

    elif isinstance(x, FileMapper):
        sequence = [x]

    elif isinstance(x, Sequence):
        sequence = x

    else:
        raise InvalidParameterValue(f"{caller} is not in an accepted format")

    return sequence


def parse_datetime(dt, defaults=None):
    """Parses string to datetime and returns isoformat string for it.
    If `defaults` is set, use that in case `dt` is None."""
    return str(str_to_AnyCalendarDateTime(dt, defaults=defaults))


class Series:
    """
    A simple class for handling a series selection, created by
    any sequence as input. It has a `value` that holds the sequence
    as a list.
    """

    def __init__(self, *data):
        if len(data) == 1:
            data = data[0]

        self.value = parse_sequence(data, caller=self.__class__.__name__)


class Interval:
    """
    A simple class for handling an interval of any type.
    It holds a `start` and `end` but does not try to resolve
    the range, it is just a container to be used by other tools.
    The contents can be of any type, such as datetimes, strings etc.
    """

    def __init__(self, *data):
        self.value = parse_range(data, self.__class__.__name__)


class TimeComponents:
    """
    A simple class for parsing and representing a set of time components.

    The components are stored in a dictionary of {time_comp: values},
    such as: {"year": [2000, 2001], "month": [1, 2, 3]}

    Note that you can provide month strings as strings or numbers, e.g.: "feb", "Feb", "February", 2.
    """

    def __init__(
        self, year=None, month=None, day=None, hour=None, minute=None, second=None
    ):
        comps = ("year", "month", "day", "hour", "minute", "second")

        self.value = {}
        for comp in comps:
            if comp in locals():
                value = locals()[comp]

                # Only add to dict if defined
                if value is not None:
                    self.value[comp] = self._parse_component(comp, value)

    def _parse_component(self, time_comp, value):
        limits = time_comp_limits[time_comp]

        if isinstance(value, str):
            if "," in value:
                value = value.split(",")
            else:
                value = [value]

        if not isinstance(value, Sequence):
            value = [value]

        def _month_to_int(month):
            if isinstance(month, str):
                month = month_map.get(month.lower()[:3], month)
            return int(month)

        if time_comp == "month":
            value = [_month_to_int(month) for month in value]
        else:
            value = [int(i) for i in value]

        if limits:
            mn, mx = min(value), max(value)
            if mn < limits[0] or mx > limits[1]:
                raise ValueError(
                    f"Some time components are out of range for {time_comp}: "
                    f"({mn}, {mx})"
                )

        return value


def string_to_dict(s, splitters=("|", ":", ",")):
    """Convert a string to a dictionary of dictionaries, based on
    splitting rules: splitters."""
    dct = {}

    for tdict in s.strip().split(splitters[0]):
        key, value = tdict.split(splitters[1])
        dct[key] = value.split(splitters[2])

    return dct


def to_float(i, allow_none=True):
    try:
        if allow_none and i is None:
            return i
        return float(i)
    except Exception:
        raise InvalidParameterValue("Values must be valid numbers")


# Create some aliases for creating simple selection types
series = time_series = level_series = area = collection = dimensions = Series
interval = time_interval = level_interval = Interval
time_components = TimeComponents
