"""Area parameter for subsetting operations."""

from collections.abc import Sequence

from clisops.exceptions import InvalidParameterValue
from clisops.parameter._utils import area, parse_sequence, to_float
from clisops.parameter.base_parameter import _BaseParameter


class AreaParameter(_BaseParameter):
    """
    Class for area parameter used in subsetting operation.

    | Area can be input as:
    | A string of comma separated values: "0.,49.,10.,65"
    | A sequence of strings: ("0", "-10", "120", "40")
    | A sequence of numbers: [0, 49.5, 10, 65]

    An area must have four values.

    Validates the area input and parses the values into numbers.
    """

    allowed_input_types = [Sequence, str, area, type(None)]

    def _parse(self):
        if isinstance(self.input, type(None)) or self.input == "":
            return None

        if isinstance(self.input, (str, bytes)):
            value = parse_sequence(self.input, caller=self.__class__.__name__)

        elif isinstance(self.input, Sequence):
            value = self.input

        elif isinstance(self.input, area):
            value = self.input.value

        self.type = "series"

        if value is not None and len(value) != 4:
            raise InvalidParameterValue(
                f"{self.__class__.__name__} should be of length 4 but is of length {len(value)}"
            )

        return tuple([to_float(i, allow_none=False) for i in value])

    def asdict(self):
        """
        Return a dictionary of the area values.

        Returns
        -------
        dict
            A dictionary with keys "lon_bnds" and "lat_bnds" containing tuples of the longitude and latitude bounds.
        """
        if self.value is not None:
            return {
                "lon_bnds": (self.value[0], self.value[2]),
                "lat_bnds": (self.value[1], self.value[3]),
            }

    def __str__(self):
        """String representation of the area values."""
        return f"Area to subset over:\n {self.value}"
