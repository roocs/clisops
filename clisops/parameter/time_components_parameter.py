"""TimeComponentsParameter class for handling time components in subsetting operations."""

from clisops.exceptions import InvalidParameterValue
from clisops.parameter._utils import string_to_dict, time_components
from clisops.parameter.base_parameter import _BaseParameter


class TimeComponentsParameter(_BaseParameter):
    """
    Class for time components parameter used in subsetting operation.

    The Time Components are any, or none of:
      - year: [list of years]
      - month: [list of months]
      - day: [list of days]
      - hour: [list of hours]
      - minute: [list of minutes]
      - second: [list of seconds]

    `month` is special: you can use either strings or values:
       "feb", "mar" == 2, 3 == "02,03"

    Validates the times input and parses them into a dictionary.
    """

    allowed_input_types = [dict, str, time_components, type(None)]

    def _parse(self):
        try:
            if self.input in (None, ""):
                return None
            elif isinstance(self.input, time_components):
                return self.input.value
            elif isinstance(self.input, str):
                time_comp_dict = string_to_dict(self.input, splitters=("|", ":", ","))
                return time_components(**time_comp_dict).value
            else:  # Must be a dict to get here
                return time_components(**self.input).value
        except Exception:
            raise InvalidParameterValue(f"Cannot create TimeComponentsParameter from: {self.input}")

    def asdict(self):
        """
        Return a dictionary of the time components.

        Returns
        -------
        dict
            A dictionary with a single key "time_components" containing the time components.
        """
        # Just return the value, either a dict or None
        return {"time_components": self.value}

    def get_bounds(self):
        """
        Return a tuple of the (start, end) times, calculated from the value of the parameter.

        Either will default to None.

        Returns
        -------
        tuple
            A tuple containing the start and end times in ISO format.
            If no year is specified, both will be None.
        """
        if "year" in self.value:
            start = f"{self.value['year'][0]}-01-01T00:00:00"
            end = f"{self.value['year'][-1]}-12-31T23:59:59"
        else:
            start = end = None
        return (start, end)

    def __str__(self):
        """Returns a string representation of the time components."""
        if self.value is None:
            return "No time components specified"

        resp = "Time components to select:"
        for key, value in self.value.items():
            resp += f"\n    {key} => {value}"
        return resp
