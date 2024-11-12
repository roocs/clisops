from collections.abc import Sequence

from clisops.exceptions import InvalidParameterValue
from clisops.parameter._utils import dimensions, parse_sequence
from clisops.parameter.base_parameter import _BaseParameter
from clisops.utils.dataset_utils import known_coord_types


class DimensionParameter(_BaseParameter):
    """
    Class for dimensions parameter used in averaging operation.

    | Area can be input as:
    | A string of comma separated values: "time,latitude,longitude"
    | A sequence of strings: ("time", "longitude")

    Dimensions can be None or any number of options from time, latitude, longitude and level provided these
    exist in the dataset being operated on.

    Validates the dims input and parses the values into a sequence of strings.

    """

    allowed_input_types = [Sequence, str, dimensions, type(None)]

    def _parse(self):
        classname = self.__class__.__name__

        if self.input in (None, ""):
            return None
        elif isinstance(self.input, dimensions):
            value = self.input.value
        else:
            value = parse_sequence(self.input, caller=classname)

        for item in value:
            if not isinstance(item, str):
                raise InvalidParameterValue("Each dimension must be a string.")

            if item not in known_coord_types:
                raise InvalidParameterValue(
                    f"Dimensions for averaging must be one of {known_coord_types}"
                )

        return tuple(value)

    def asdict(self):
        """Returns a dictionary of the dimensions"""
        if self.value is not None:
            return {"dims": self.value}

    def __str__(self):
        return f"Dimensions to average over:" f"\n {self.value}"
