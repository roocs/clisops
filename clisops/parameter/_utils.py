"""Utility functions and classes for handling parameters in CLISOPS."""

import calendar
from collections.abc import Sequence
from typing import Any

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
def parse_range(x: Sequence[str | Sequence[str]], caller) -> tuple[str | None, str | None]:
    """
    Parse a range input into a start and end value.

    Parameters
    ----------
    x : str or Sequence[str or Sequence[str]]
        The input range to parse. Can be a string like "start/end", a sequence of two values, or a single value.
    caller : str
        The name of the caller for error messages, used to indicate where the error occurred.

    Returns
    -------
    tuple[str or None, str or None]
        A tuple containing the start and end values. If the input is "/", both values will be None.
    """
    if isinstance(x, Sequence) and len(x) == 1:
        x = x[0]

    if x in ("/", None, ""):
        start = None
        end = None

    elif isinstance(x, str):
        if "/" not in x:
            raise InvalidParameterValue(f"{caller} should be passed in as a range separated by /")

        # empty string either side of '/' is converted to None
        start, end = (i.strip() or None for i in x.split("/"))

    elif isinstance(x, Sequence):
        if len(x) != 2:
            raise InvalidParameterValue(f"{caller} should be a range. Expected 2 values, received {len(x)}")

        start, end = x

    else:
        raise InvalidParameterValue(f"{caller} is not in an accepted format")
    return start, end


def parse_sequence(x: Sequence | str | bytes | FileMapper, caller: str) -> Sequence:
    """
    Parse a sequence input into a list of values.

    Parameters
    ----------
    x : Sequence | str | bytes | FileMapper
        The input sequence to parse. Can be a string of comma-separated values, a sequence, or a FileMapper object.
    caller : str
        The name of the caller for error messages, used to indicate where the error occurred.

    Returns
    -------
    Sequence
        A list of values parsed from the input. If the input is None or an empty string, returns None.
    """
    if x is None or x == "":
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


def parse_datetime(dt: str, defaults: list[int] | None = None):
    """
    Parse string to datetime and returns isoformat string for it.

    If `defaults` is set, use that in case `dt` is None.

    Parameters
    ----------
    dt : str
        The datetime string to parse, in ISO 8601 format.
    defaults : list[int] | None
        A list of default values to use if `dt` is None. Should contain year, month, day, hour, minute, second.

    Returns
    -------
    str
        The ISO 8601 formatted string representation of the datetime.
    """
    return str(str_to_AnyCalendarDateTime(dt, defaults=defaults))


class Series:
    """
    A simple class for handling a series selection, created by any sequence as input.

    It has a `value` that holds the sequence as a list.

    Parameters
    ----------
    *data : Sequence or str or bytes or FileMapper
        The input data to parse into a sequence.
        Can be a string of comma-separated values, a sequence, or a FileMapper object.
    """

    def __init__(self, *data):
        """
        Initialize the Series with a sequence of data.

        Parameters
        ----------
        *data : Sequence or str or bytes or FileMapper
            The input data to parse into a sequence.
            Can be a string of comma-separated values, a sequence, or a FileMapper object.
        """
        if len(data) == 1:
            data = data[0]

        self.value = parse_sequence(data, caller=self.__class__.__name__)


class Interval:
    """
    A simple class for handling an interval of any type.

    It holds a `start` and `end` but does not try to resolve the range,
    it is just a container to be used by other tools.
    The contents can be of any type, such as datetimes, strings etc.

    Parameters
    ----------
    *data : str or Sequence[str]
        The input data to parse into a start and end value.
        Can be a string like "start/end", a sequence of two values, or a single value.
    """

    def __init__(self, *data):
        self.value = parse_range(data, self.__class__.__name__)


class TimeComponents:
    """
    A simple class for parsing and representing a set of time components.

    The components are stored in a dictionary of {time_comp: values},
    such as: {"year": [2000, 2001], "month": [1, 2, 3]}

    Note that you can provide month strings as strings or numbers, e.g.: "feb", "Feb", "February", 2.

    Parameters
    ----------
    year : int or str, optional
        The year component, e.g., 2020.
    month : int or str, optional
        The month component, e.g., 1 for January or "feb" for February.
    day : int or str, optional
        The day component, e.g., 15 for the 15th of the month.
    hour : int or str, optional
        The hour component, e.g., 12 for noon.
    minute : int or str, optional
        The minute component, e.g., 30 for half past the hour.
    second : int or str, optional
        The second component, e.g., 45 for 45 seconds past the minute.
    """

    def __init__(self, year=None, month=None, day=None, hour=None, minute=None, second=None):
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
                raise ValueError(f"Some time components are out of range for {time_comp}: ({mn}, {mx})")

        return value


def string_to_dict(s, splitters=("|", ":", ",")):
    """
    Convert a string to a dictionary of dictionaries, based on splitting rules (splitters).

    Parameters
    ----------
    s : str
        The input string to convert, formatted as "key1:value1,value2|key2:value3,value4".
    splitters : tuple[str, str, str]
        A tuple of strings used to split the input string into keys and values.
        The first element is used to split the main entries, the second for key-value pairs,
        and the third for value lists.

    Returns
    -------
    dict
        A dictionary where each key maps to a list of values.
    """
    dct = {}

    for tdict in s.strip().split(splitters[0]):
        key, value = tdict.split(splitters[1])
        dct[key] = value.split(splitters[2])

    return dct


def to_float(i: Any | None, allow_none: bool = True) -> float | None:
    """
    Convert a value to a float, allowing for None if specified.

    Parameters
    ----------
    i : Any | None
        The input value to convert to a float.
    allow_none : bool, optional
        If True, allows the input to be None and returns None. Defaults to True.

    Returns
    -------
    float or None
        The converted float value, or None if the input is None and allow_none is True.
    """
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
