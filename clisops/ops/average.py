from pathlib import Path
from typing import List, Optional, Tuple, Union

import xarray as xr
from roocs_utils.parameter.dimension_parameter import DimensionParameter
from roocs_utils.xarray_utils.xarray_utils import (
    convert_coord_to_axis,
    get_coord_type,
    known_coord_types,
    open_xr_dataset,
)

from clisops import logging, utils
from clisops.core import average
from clisops.ops.base_operation import Operation
from clisops.utils.common import expand_wildcards
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices

__all__ = [
    "average_over_dims",
]

LOGGER = logging.getLogger(__file__)


class Average(Operation):
    def _resolve_params(self, **params):
        dims = DimensionParameter(params.get("dims", None)).value
        ignore_undetected_dims = params.get("ignore_undetected_dims", False)

        self.params = {"dims": dims, "ignore_undetected_dims": ignore_undetected_dims}

    def _get_file_namer(self):

        if self.params.get("dims", None):
            dims = [convert_coord_to_axis(dim) for dim in self.params["dims"]]
            extra = f"_avg-{''.join(sorted(dims))}"
        else:
            extra = ""

        namer = get_file_namer(self._file_namer)(extra=extra)

        return namer

    def _calculate(self):
        avg_ds = average.average_over_dims(
            self.ds,
            self.params.get("dims", None),
            self.params.get("ignore_undetected_dims", None),
        )

        return avg_ds


def average_over_dims(
    ds,
    dims: Optional[Union[Tuple[str], DimensionParameter]] = None,
    ignore_undetected_dims: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    output_type="netcdf",
    split_method="time:auto",
    file_namer="standard",
) -> List[Union[xr.Dataset, str]]:
    """

    Parameters
    ----------
    ds: Union[xr.Dataset, str]
    dims : Optional[Union[Tuple[str], DimensionParameter]]
      The dimensions over which to apply the average. If None, none of the dimensions are averaged over. Dimensions
      must be one of ["time", "level", "latitude", "longitude"].
    ignore_undetected_dims: bool
      If the dimensions specified are not found in the dataset, an Exception will be raised if set to True.
      If False, an exception will not be raised and the other dimensions will be averaged over. Default = False
    output_dir: Optional[Union[str, Path]] = None
    output_type: {"netcdf", "nc", "zarr", "xarray"}
    split_method: {"time:auto"}
    file_namer: {"standard", "simple"}

    Returns
    -------
    List[Union[xr.Dataset, str]]
    A list of the outputs in the format selected; str corresponds to file paths if the
    output format selected is a file.

    Examples
    --------
    | ds: xarray Dataset or "cmip5.output1.MOHC.HadGEM2-ES.rcp85.mon.atmos.Amon.r1i1p1.latest.tas"
    | dims: ['latitude', 'longitude']
    | ignore_undetected_dims: False
    | output_dir: "/cache/wps/procs/req0111"
    | output_type: "netcdf"
    | split_method: "time:auto"
    | file_namer: "standard"

    """
    op = Average(**locals())
    return op.process()
