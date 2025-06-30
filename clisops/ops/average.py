"""Average operations for xarray datasets."""

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import xarray as xr

from clisops.core import average
from clisops.exceptions import InvalidParameterValue
from clisops.ops.base_operation import Operation
from clisops.parameter import DimensionParameter
from clisops.utils.dataset_utils import convert_coord_to_axis
from clisops.utils.file_namers import get_file_namer

__all__ = ["average_over_dims", "average_shape", "average_time"]


class Average(Operation):
    """Average operation for xarray datasets."""

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
    ds: xr.Dataset | str,
    dims: Sequence[str] | DimensionParameter | None = None,
    ignore_undetected_dims: bool = False,
    output_dir: str | Path | None = None,
    output_type: str = "netcdf",
    split_method: str = "time:auto",
    file_namer: str = "standard",
) -> list[xr.Dataset | str]:
    """
    Calculate an average over given dimensions.

    Parameters
    ----------
    ds : xr.Dataset or str
        Xarray dataset.
    dims : Sequence of str or DimensionParameter, optional
        The dimensions over which to apply the average. If None, none of the dimensions are averaged over. Dimensions
        must be one of ["time", "level", "latitude", "longitude"].
    ignore_undetected_dims : bool
        If the dimensions specified are not found in the dataset, an Exception will be raised if set to True.
        If False, an exception will not be raised and the other dimensions will be averaged over. Default = False.
    output_dir : str or Path, optional
        The directory where the output files will be saved. If None, the output will not be saved to disk.
    output_type : {"netcdf", "nc", "zarr", "xarray"}
        The format of the output files. If "xarray", the output will be an xarray Dataset.
        If "netcdf", "nc", or "zarr", the output will be saved to disk in the specified format.
    split_method : {"time:auto"}
        The method to split the output files. Currently only "time:auto" is supported, which will
    file_namer : {"standard", "simple"}
        The file namer to use for generating output file names.
        "standard" uses a more descriptive naming convention, while "simple" uses a numbered sequence.

    Returns
    -------
    list of xr.Dataset or str
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


class AverageShape(Operation):
    """Average operation for xarray datasets over a given shape."""

    def _resolve_params(self, **params):
        shape = params.get("shape")
        variable = params.get("variable", None)

        self.params = {"shape": shape, "variable": variable}

        if not shape:
            raise InvalidParameterValue("At least one area for averaging must be provided")

    def _get_file_namer(self):
        extra = "_avg-shape"

        namer = get_file_namer(self._file_namer)(extra=extra)

        return namer

    def _calculate(self):
        avg_ds = average.average_shape(
            self.ds,
            self.params.get("shape", None),
            self.params.get("variable", None),
        )
        return avg_ds


def average_shape(
    ds: xr.Dataset | Path | str,
    shape: str | Path | gpd.GeoDataFrame,
    variable: str | Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    output_type: str = "netcdf",
    split_method: str = "time:auto",
    file_namer: str = "standard",
) -> list[xr.Dataset | str]:
    """
    Calculate a spatial average over a given shape.

    Parameters
    ----------
    ds : xr.Dataset or str or Path
        Xarray dataset.
    shape : str, Path, or gpd.GeoDataFrame
        Path to shape file, or directly a GeoDataFrame. Supports formats compatible with geopandas.
        Will be converted to EPSG:4326 if needed.
    variable : str or sequence of str, optional
        Variables to average. If None, average over all data variables.
    output_dir : str or Path, optional
        The directory where the output files will be saved. If None, the output will not be saved to disk.
    output_type : {"netcdf", "nc", "zarr", "xarray"}
        The format of the output files. If "xarray", the output will be an xarray Dataset.
    split_method : {"time:auto"}
        The method to split the output files.
        Currently only "time:auto" is supported, which will automatically split the output files based on time.
    file_namer : {"standard", "simple"}
        The file namer to use for generating output file names.
        "standard" uses a more descriptive naming convention, while "simple" uses a numbered sequence.

    Returns
    -------
    list of xr.Dataset or str
        A list of the outputs in the format selected.
        str corresponds to file paths if the output format selected is a file.

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
    op = AverageShape(**locals())
    return op.process()


class AverageTime(Operation):
    """Average operation for xarray datasets over a given time frequency."""

    def _resolve_params(self, **params):
        freq = params.get("freq", None)

        if not freq:
            raise InvalidParameterValue("At least one frequency for averaging must be provided")

        if freq not in list(average.freqs.keys()):
            raise InvalidParameterValue(f"Time frequency for averaging must be one of {list(average.freqs.keys())}.")

        self.params = {"freq": freq}

    def _get_file_namer(self):
        extra = f"_avg-{self.params.get('freq')}"
        namer = get_file_namer(self._file_namer)(extra=extra)

        return namer

    def _calculate(self):
        avg_ds = average.average_time(
            self.ds,
            self.params.get("freq", None),
        )

        return avg_ds


def average_time(
    ds: xr.Dataset | str,
    freq: str,
    output_dir: str | Path | None = None,
    output_type: str = "netcdf",
    split_method: str = "time:auto",
    file_namer: str = "standard",
) -> list[xr.Dataset | str]:
    """
    Calculate an average over time for a given frequency.

    Parameters
    ----------
    ds : xr.Dataset or str
        Xarray dataset.
    freq : str
        The frequency to average over, either "month" or "year".
    output_dir : str or Path, optional
        The directory where the output files will be saved. If None, the output will not be saved to disk.
    output_type : {"netcdf", "nc", "zarr", "xarray"}
        The format of the output files. If "xarray", the output will be an xarray Dataset.
    split_method : {"time:auto"}
        The method to split the output files. Currently only "time:auto" is supported, which will
        automatically split the output files based on time.
    file_namer : {"standard", "simple"}
        The file namer to use for generating output file names.
        "standard" uses a more descriptive naming convention, while "simple" uses a numbered sequence.

    Returns
    -------
    List of datasets or file paths
        A list of the outputs in the format selected.
        str corresponds to file paths if the output format selected is a file.

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
    op = AverageTime(**locals())
    return op.process()
