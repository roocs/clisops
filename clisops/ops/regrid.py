from pathlib import Path
from typing import List, Optional, Tuple, Union

import xarray as xr

from clisops import logging
from clisops.core import Grid, Weights
from clisops.core import regrid as core_regrid
from clisops.ops.base_operation import Operation
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices

__all__ = [
    "regrid",
]

LOGGER = logging.getLogger(__file__)

supported_regridding_methods = ["conservative", "patch", "nearest_s2d", "bilinear"]


class Regrid(Operation):
    def _get_grid_in(self, grid_desc):
        if isinstance(grid_desc, (xr.Dataset, xr.DataArray)):
            return Grid(ds=grid_desc)
        raise Exception(
            "An xarray.Dataset or xarray.DataArray has to be provided as input."
        )

    def _get_grid_out(self, grid_desc):
        if isinstance(grid_desc, str):
            return Grid(grid_id=grid_desc)
        elif isinstance(grid_desc, tuple):
            return Grid(grid_instructor=grid_desc)
        elif isinstance(grid_desc, (xr.Dataset, xr.DataArray)):
            return Grid(ds=grid_desc)
        else:
            return Grid()

    def _get_weights(self, grid_in, grid_out, method):
        return Weights(grid_in=grid_in, grid_out=grid_out, method=method)

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
        grid = params.get("grid", "1deg")
        method = params.get("method", "nearest_s2d")

        if method not in supported_regridding_methods:
            raise Exception(
                "The selected regridding method is not supported. "
                "Please choose one of %s." % ", ".join(supported_regridding_methods)
            )

        LOGGER.debug(
            f"Input parameters: method: {method}, grid: {grid}, adaptive_masking: {adaptive_masking_threshold}"
        )

        grid_in = self._get_grid_in(self.ds)
        grid_out = self._get_grid_out(grid)
        weights = self._get_weights(grid_in=grid_in, grid_out=grid_out, method=method)

        self.params = {
            "grid_in": grid_in,
            "grid_out": grid_out,
            "method": method,
            "regridder": weights.Regridder,
            "adaptive_masking_threshold": adaptive_masking_threshold,
        }

        # In case there was a Halo removed, the shape of grid_in.ds and self.ds would not match anymore
        #  An alternative would be to store the halo information (duplicated rows, columns) in an attribute,
        #  and then remove those rows and columns from self.ds?
        self.ds = self.params.get("grid_in").ds

        # Theres no __str__() method for the Regridder object, so I used its filename attribute,
        #  which specifies a default filename (which has but not much to do with the filename we would give the weight file).
        # Better option might be to have the Weights class extend the Regridder class or to define
        #  a __str__() method for the Weights class.
        LOGGER.debug(
            "Resolved parameters: grid_in: {}, grid_out: {}, regridder: {}".format(
                self.params.get("grid_in").__str__(),
                self.params.get("grid_out").__str__(),
                self.params.get("regridder").filename,
            )
        )

    def _get_file_namer(self):
        # need to overwrite the file namer to make it clear in the output file name
        # that the dataset has been regridded - see ops.average.Average
        # extra is what will go at the end of the file name before .nc
        # this may not make sense so change if needed

        extra = "_regrid-{}-{}".format(
            self.params.get("method"), self.params.get("grid_out").__str__()
        )

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
        regrid_ds = core_regrid(
            self.ds,
            self.params.get("regridder", None),
            self.params.get("adaptive_masking_threshold", None),
        )

        # The output ds might not yet be optimal
        # Will need to test which data variables get dropped or modified by the remapping.
        # It seems time_bnds, ... get dropped but the vertices of the old dataset are kept.
        # Also, might want to set more global attributes (clisops+xesmf version, weights information)

        return regrid_ds


def regrid(
    ds: Union[xr.Dataset, str, Path],
    *,
    method="nearest_s2d",  # do we want defaults for these values? Yes, but now I added them at _resolve as well. Where to get rid of them?
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
