from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional, Union

import xarray as xr
from loguru import logger
from roocs_utils.exceptions import InvalidParameterValue

from clisops.core import Grid, Weights
from clisops.core import regrid as core_regrid
from clisops.ops.base_operation import Operation
from clisops.utils.file_namers import get_file_namer

__all__ = [
    "regrid",
]

supported_regridding_methods = ["conservative", "patch", "nearest_s2d", "bilinear"]


class Regrid(Operation):
    """Class for regridding operation, extends clisops.ops.base_operation.Operation."""

    def _get_grid_in(
        self,
        grid_desc: Union[xr.Dataset, xr.DataArray],
        compute_bounds: bool,
    ):
        """
        Create clisops.core.regrid.Grid object as input grid of the regridding operation.

        Return the Grid object.
        """
        if isinstance(grid_desc, (xr.Dataset, xr.DataArray)):
            return Grid(ds=grid_desc, compute_bounds=compute_bounds)
        raise InvalidParameterValue(
            "An xarray.Dataset or xarray.DataArray has to be provided as input for the source grid."
        )

    def _get_grid_out(
        self,
        grid_desc: Union[xr.Dataset, xr.DataArray, int, float, tuple, str],
        compute_bounds: bool,
    ):
        """
        Create clisops.core.regrid.Grid object as target grid of the regridding operation.

        Returns the Grid object
        """
        if isinstance(grid_desc, str):
            if grid_desc in ["auto", "adaptive"]:
                return Grid(
                    ds=self.ds, grid_id=grid_desc, compute_bounds=compute_bounds
                )
            else:
                return Grid(grid_id=grid_desc, compute_bounds=compute_bounds)
        elif isinstance(grid_desc, (float, int, tuple)):
            return Grid(grid_instructor=grid_desc, compute_bounds=compute_bounds)
        elif isinstance(grid_desc, (xr.Dataset, xr.DataArray)):
            return Grid(ds=grid_desc, compute_bounds=compute_bounds)
        else:
            # clisops.core.regrid.Grid will raise the exception
            return Grid()

    def _get_weights(self, grid_in: Grid, grid_out: Grid, method: str):
        """
        Generate the remapping weights using clisops.core.regrid.Weights.

        Returns the Weights object.
        """
        return Weights(grid_in=grid_in, grid_out=grid_out, method=method)

    def _resolve_params(self, **params):
        """Generate a dictionary of regrid parameters."""
        # all regrid specific paramterers should be passed in via **params
        # this is where we resolve them and set self.params as a dict or as separate attributes
        # this would be where you make use of your other methods/ attributes e.g.
        # get_grid_in(), get_grid_out() and get_weights() to generate the regridder

        adaptive_masking_threshold = params.get("adaptive_masking_threshold", None)
        grid = params.get("grid", None)
        method = params.get("method", None)
        keep_attrs = params.get("keep_attrs", None)

        if method not in supported_regridding_methods:
            raise Exception(
                "The selected regridding method is not supported. "
                "Please choose one of %s." % ", ".join(supported_regridding_methods)
            )

        logger.debug(
            f"Input parameters: method: {method}, grid: {grid}, adaptive_masking: {adaptive_masking_threshold}"
        )

        # Compute bounds only when required
        compute_bounds = "conservative" in method

        # Create and check source and target grids
        grid_in = self._get_grid_in(self.ds, compute_bounds)
        grid_out = self._get_grid_out(grid, compute_bounds)

        # Compute the remapping weights
        t_start = dt.now()
        weights = self._get_weights(grid_in=grid_in, grid_out=grid_out, method=method)
        t_end = dt.now()
        logger.info(
            f"Computed/Retrieved weights in {(t_end-t_start).total_seconds()} seconds."
        )

        # Define params dict
        self.params = {
            "grid_in": grid_in,
            "grid_out": grid_out,
            "method": method,
            "regridder": weights.regridder,
            "weights": weights,
            "adaptive_masking_threshold": adaptive_masking_threshold,
            "keep_attrs": keep_attrs,
        }

        # Input grid / Dataset
        self.ds = self.params.get("grid_in").ds

        # Theres no __str__() method for the Regridder object, so I used its filename attribute,
        #  which specifies a default filename (which has but not much to do with the filename we would give the weight file).
        # todo: Better option might be to have the Weights class extend the Regridder class or to define
        #  a __str__() method for the Weights class.
        logger.debug(
            "Resolved parameters: grid_in: {}, grid_out: {}, regridder: {}".format(
                self.params.get("grid_in").__str__(),
                self.params.get("grid_out").__str__(),
                self.params.get("regridder").filename,
            )
        )

    def _get_file_namer(self):
        """Return the appropriate file namer object."""
        # "extra" is what will go at the end of the file name before .nc
        extra = "_regrid-{}-{}".format(
            self.params.get("method"), self.params.get("grid_out").__str__()
        )

        namer = get_file_namer(self._file_namer)(extra=extra)

        return namer

    def _calculate(self):
        """
        Process the regridding request, calls clisops.core.regrid.regrid().

        Returns the resulting xarray.Dataset.
        """
        # the result is saved by the process() method on the base class
        regridded_ds = core_regrid(
            self.params.get("grid_in", None),
            self.params.get("grid_out", None),
            self.params.get("weights", None),
            self.params.get("adaptive_masking_threshold", None),
            self.params.get("keep_attrs", None),
        )

        return regridded_ds


def regrid(
    ds: Union[xr.Dataset, str, Path],
    *,
    method: Optional[str] = "nearest_s2d",
    adaptive_masking_threshold: Optional[Union[int, float]] = 0.5,
    grid: Optional[
        Union[xr.Dataset, xr.DataArray, int, float, tuple, str]
    ] = "adaptive",
    output_dir: Optional[Union[str, Path]] = None,
    output_type: Optional[str] = "netcdf",
    split_method: Optional[str] = "time:auto",
    file_namer: Optional[str] = "standard",
    keep_attrs: Optional[Union[bool, str]] = True,
) -> List[Union[xr.Dataset, str]]:
    """
    Regrid specified input file or xarray object.

    Parameters
    ----------
    ds: Union[xr.Dataset, str]
    method="nearest_s2d",
    adaptive_masking_threshold=0.5,
    grid="adaptive",
    output_dir: Optional[Union[str, Path]] = None
    output_type: {"netcdf", "nc", "zarr", "xarray"}
    split_method: {"time:auto"}
    file_namer: {"standard", "simple"}
    keep_attrs: {True, False, "target"}

    Returns
    -------
    List[Union[xr.Dataset, str]]
        A list of the regridded outputs in the format selected; str corresponds to file paths if the
        output format selected is a file.

    Examples
    --------
    | ds: xarray Dataset or "cmip5.output1.MOHC.HadGEM2-ES.rcp85.mon.atmos.Amon.r1i1p1.latest.tas"
    | method: "nearest_s2d"
    | adaptive_masking_threshold:
    | grid: "1deg"
    | output_dir: "/cache/wps/procs/req0111"
    | output_type: "netcdf"
    | split_method: "time:auto"
    | file_namer: "standard"
    | keep_attrs: True
    """
    op = Regrid(**locals())
    return op.process()
