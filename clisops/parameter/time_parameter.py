import datetime

from clisops.exceptions import InvalidParameterValue
from clisops.parameter._utils import parse_datetime
from clisops.parameter.base_parameter import _BaseIntervalOrSeriesParameter


class TimeParameter(_BaseIntervalOrSeriesParameter):
    """
    Class for time parameter used in subsetting operation.

    | Time can be input as:
    | A string of slash separated values: "2085-01-01T12:00:00Z/2120-12-30T12:00:00Z"
    | A sequence of strings: e.g. ("2085-01-01T12:00:00Z", "2120-12-30T12:00:00Z")

    A time input must be 2 values.

    If using a string input a trailing slash indicates you want to use the earliest/
    latest time of the dataset. e.g. "2085-01-01T12:00:00Z/" will subset from 01/01/2085 to the final time in
    the dataset.

    Validates the times input and parses the values into isoformat.

    """

    def _parse_as_interval(self):
        start, end = self.input.value

        try:
            if start is not None:
                start = parse_datetime(
                    start, defaults=[datetime.MINYEAR, 1, 1, 0, 0, 0]
                )
            if end is not None:
                end = parse_datetime(
                    end, defaults=[datetime.MAXYEAR, 12, 31, 23, 59, 59]
                )

        except Exception:
            raise InvalidParameterValue("Unable to parse the time values entered")

        # Set as None if no start or end, otherwise set as tuple
        value = (start, end)

        if set(value) == {None}:
            value = None

        return value

    def _parse_as_series(self):
        try:
            value = [parse_datetime(tm) for tm in self.input.value]
        except Exception:
            raise InvalidParameterValue("Unable to parse the time values entered")

        return value

    def asdict(self):
        """Returns a dictionary of the time values"""
        if self.type in ("interval", "none"):
            value = self._value_as_tuple()
            return {"start_time": value[0], "end_time": value[1]}
        elif self.type == "series":
            return {"time_values": self.value}

    def get_bounds(self):
        """Returns a tuple of the (start, end) times, calculated from
        the value of the parameter. Either will default to None."""
        if self.type in ("interval", "none"):
            return self._value_as_tuple()

        elif self.type == "series":
            return self.value[0], self.value[-1]

    def __str__(self):
        if self.type in ("interval", "none"):
            value = self._value_as_tuple()
            return (
                f"Time period to subset over"
                f"\n start time: {value[0]}"
                f"\n end time: {value[1]}"
            )
        else:
            return f"Time values to select: {self.value}"
