from clisops.exceptions import InvalidParameterValue
from clisops.parameter._utils import interval, series


class _BaseParameter:
    """Base class for parameters used in operations (e.g. subset, average etc.)."""

    allowed_input_types = None

    def __init__(self, input):
        self.input = self.raw = input

        # If the input is already an instance of this class, call its parse method
        if isinstance(self.input, self.__class__):
            self.value = self.input.value
            self.type = getattr(self.input, "type", "undefined")
        else:
            self._check_input_type()
            self.value = self._parse()

    def _check_input_type(self):
        if not self.allowed_input_types:
            return
        if not isinstance(self.input, tuple(self.allowed_input_types)):
            raise InvalidParameterValue(
                f"Input type of {type(self.input)} not allowed. "
                f"Must be one of: {self.allowed_input_types}"
            )

    def _parse(self):
        raise NotImplementedError()

    def get_bounds(self):
        """Returns a tuple of the (start, end) times, calculated from
        the value of the parameter. Either will default to None."""
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        return str(self)

    def __unicode__(self):
        return str(self)


class _BaseIntervalOrSeriesParameter(_BaseParameter):
    """
    A base class for a parameter that can be instantiated from either and `Interval` or `Series` class instance.

    It has a `type` and a `value` reflecting the type, e.g.:
        - type: "interval" --> value: (start, end)
        - type: "series"   --> value: [item1, item2, ..., item_n]
    """

    allowed_input_types = [interval, series, type(None), str]

    def _parse(self):
        if isinstance(self.input, interval):
            self.type = "interval"
            return self._parse_as_interval()
        elif isinstance(self.input, series):
            self.type = "series"
            return self._parse_as_series()
        elif isinstance(self.input, type(None)):
            self.type = "none"
            return None
        elif isinstance(self.input, str):
            if "/" in self.input:
                self.type = "interval"
                self.input = interval(self.input)
                return self._parse_as_interval()
            else:
                self.type = "series"
                self.input = series(self.input)
                return self._parse_as_series()

    def _parse_as_interval(self):
        raise NotImplementedError()

    def _parse_as_series(self):
        raise NotImplementedError()

    def _value_as_tuple(self):
        value = self.value
        if value is None:
            value = None, None

        return value
