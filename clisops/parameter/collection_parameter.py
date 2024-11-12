from collections.abc import Sequence

from clisops.exceptions import InvalidParameterValue, MissingParameterValue
from clisops.parameter._utils import collection, parse_sequence
from clisops.parameter.base_parameter import _BaseParameter
from clisops.utils.file_utils import FileMapper


class CollectionParameter(_BaseParameter):
    """
    Class for collection parameter used in operations.

    | A collection can be input as:
    | A string of comma separated values: "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga,cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga"
    | A sequence of strings: e.g. ("cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga", "cmip5.output1.MPI-M.MPI-ESM-LR.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga")
    | A sequence of clisops.utils.file_utils.FileMapper objects

    Validates the input and parses the items.

    """

    allowed_input_types = [Sequence, str, collection, FileMapper]

    def _parse(self):
        classname = self.__class__.__name__

        if self.input in (None, ""):
            raise MissingParameterValue(f"{classname} must be provided")
        elif isinstance(self.input, collection):
            value = self.input.value
        else:
            value = parse_sequence(self.input, caller=classname)

        for item in value:
            if not isinstance(item, (str, FileMapper)):
                raise InvalidParameterValue(
                    f"Each id in a collection must be a string or an instance of {FileMapper}"
                )

        return tuple(value)

    def __str__(self):
        string = "Datasets to analyse:"
        for i in self.value:
            string += f"\n{i}"
        return string
