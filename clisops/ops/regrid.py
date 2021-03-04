from pathlib import Path
from typing import List, Optional, Tuple, Union

import xarray as xr

from clisops import logging
from clisops.core import regrid
from clisops.ops.base_operation import Operation
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices

__all__ = [
    "regrid",
]

LOGGER = logging.getLogger(__file__)


class Regrid(Operation):
    def get_grid_in(self):
        # self.grid_in =
        pass

    def get_grid_out(self):
        # self.grid_out =
        pass

    def get_weights(self):
        # self.weights =
        pass

    def _resolve_params(self, **params):
        """ Generates a dictionary of regrid parameters """
        # all regrid specific paramterers should be passed in via **params
        # this is where we resolve them and set self.params as a dict or as separate attributes
        # this would be where you make use of your other methods/ attributes e.g.
        # get_grid_in(), get_grid_out() and get_weights() to generate the regridder

        # verify here that grid and method are valid inputs.
        # we use roocs_utils.exceptions.InvalidParameterValue if an input isn't right

        # self.method = params.get("method", "nn")

        adaptive_masking_threshold = params.get("adaptive_masking_threshold", 0.5)

        self.params = {
            "regridder": regridder,
            "adaptive_masking_threshold": adaptive_masking_threshold,
        }

    def _get_file_namer(self):
        # need to overwrite the file namer to make it clear in the output file name
        # that the dataset has been regridded - see ops.average.Average
        # extra is what will go at the end of the file name before .nc
        # this may not make sense so change if needed

        extra = f"_regrid-{self.method}-{self.grid_out}"

        namer = get_file_namer(self._file_namer)(extra=extra)

        return namer

    def _calculate(self):
        # calculate is where clisops.core.regird should be called
        # e.g. in ops.average.Average._calculate()
        # avg_ds = average.average_over_dims(
        #     self.ds,
        #     self.params.get("dims", None),
        #     self.params.get("ignore_undetected_dims", None),
        # )

        # return avg_ds

        # remove halos before regridding

        # the result is saved by the process() method on the base class - so I think that would replace your save()?
        regrid_ds = regrid.regrid(
            ds,
            self.params.get("regridder", None),
            self.params.get("adaptive_masking_threshold", None),
        )

        return regrid_ds


def regrid(
    ds: Union[xr.Dataset, str, Path],
    *,
    method="nn",  # do we want defaults for these values?
    adaptive_masking_threshold=0.5,
    grid="1deg",
    output_dir: Optional[Union[str, Path]] = None,
    output_type="netcdf",
    split_method="time:auto",
    file_namer="standard",
) -> List[Union[xr.Dataset, str]]:
    """

    Parameters
    ----------
    ds: Union[xr.Dataset, str]
    method="nn",
    adaptive_masking_threshold=0.5,
    grid,
    output_dir: Optional[Union[str, Path]] = None
    output_type: {"netcdf", "nc", "zarr", "xarray"}
    split_method: {"time:auto"}
    file_namer: {"standard", "simple"}

    Returns
    -------
    List[Union[xr.Dataset, str]]
        A list of the regridded outputs in the format selected; str corresponds to file paths if the
        output format selected is a file.

    Examples
    --------
    | ds: xarray Dataset or "cmip5.output1.MOHC.HadGEM2-ES.rcp85.mon.atmos.Amon.r1i1p1.latest.tas"
    | method:
    | adaptive_masking_threshold:
    | grid:
    | output_dir: "/cache/wps/procs/req0111"
    | output_type: "netcdf"
    | split_method: "time:auto"
    | file_namer: "standard"

    """
    op = Regrid(**locals())
    return op.process()
