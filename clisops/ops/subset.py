"""Subset operations for xarray datasets."""

from pathlib import Path

import xarray as xr
from loguru import logger

from clisops.core import (
    subset_bbox,
    subset_level,
    subset_level_by_values,
    subset_time,
    subset_time_by_components,
    subset_time_by_values,
)
from clisops.core.subset import assign_bounds, get_lat, get_lon
from clisops.ops.base_operation import Operation
from clisops.parameter import (
    Interval,
    LevelParameter,
    Series,
    TimeComponents,
    TimeComponentsParameter,
    TimeParameter,
    parameterise,
)
from clisops.parameter.area_parameter import AreaParameter
from clisops.utils.dataset_utils import cf_convert_between_lon_frames

__all__ = ["Subset", "subset"]


class Subset(Operation):
    """
    Subset operation for xarray datasets.

    This operation allows subsetting of datasets based on time, area, and level parameters.

    Attributes
    ----------
    ds : xr.Dataset or str or Path
        The dataset to be subsetted, can be a path to a file or an xarray Dataset.
    time : str or tuple or TimeParameter or Series or Interval, optional
        Time parameter for subsetting, can be a string, tuple, or TimeParameter instance.
    area : str or tuple or AreaParameter, optional
        Area parameter for subsetting, can be a string, tuple, or AreaParameter instance.
    level : str or tuple or LevelParameter or Interval, optional
        Level parameter for subsetting, can be a string, tuple, or LevelParameter instance.
    time_components : str or dict or TimeComponents or TimeComponentsParameter, optional
        Time components for subsetting, can be a string, dictionary, or TimeComponentsParameter instance.
    output_dir : str or Path, optional
        Directory where the output will be saved. If None, the output will not be saved to a file.
    output_type : str, default "netcdf"
        The format of the output, can be "netcdf", "nc", "zarr", or "xarray".
    split_method : str, default "time:auto"
        Method for splitting the output, currently only supports "time:auto".
    file_namer : str, default "standard"
        The file naming strategy to use for the output files, can be "standard" or "simple".

    Methods
    -------
    _resolve_params(**params)
        Generates a dictionary of subset parameters based on the provided arguments.
    _calculate()
        Processes the subsetting request and returns the subsetted dataset.
    process()
        Executes the subsetting operation and returns the result.
    """

    def _resolve_params(self, **params):
        """Generates a dictionary of subset parameters."""
        time = params.get("time", None)
        area = params.get("area", None)
        level = params.get("level", None)
        time_comps = params.get("time_components", None)

        logger.debug(f"Mapping parameters: time: {time}, area: {area}, level: {level}, time_components: {time_comps}.")

        # Set up args dictionary to be used by `self._calculate()`
        args = dict()

        parameters = parameterise(
            collection=self.ds,
            time=time,
            area=area,
            level=level,
            time_components=time_comps,
        )

        # For each required parameter, check if the parameter can be accessed as a tuple
        # If not: then use the dictionary representation for it
        for param_name in ["time", "area", "level", "time_components"]:
            param_value = parameters.get(param_name)
            if param_value.value is not None:
                args.update(param_value.asdict())

        # Rename start_time and end_time to start_date and end_date to
        # match clisops.core.subset function parameters.
        if "start_time" in args:
            args["start_date"] = args.pop("start_time")

        if "end_time" in args:
            args["end_date"] = args.pop("end_time")

        self.params = args

    def _calculate(self):
        if "lon_bnds" and "lat_bnds" in self.params:
            lon = get_lon(self.ds)
            lat = get_lat(self.ds)

            # ensure lat/lon bounds are in the same order as data, before trying to roll
            # if descending in dataset, they will be flipped in subset_bbox
            self.params["lon_bnds"], self.params["lat_bnds"] = (
                assign_bounds(self.params.get("lon_bnds"), self.ds[lon.name]),
                assign_bounds(self.params.get("lat_bnds"), self.ds[lat.name]),
            )

            # subset with space and optionally time and level
            logger.debug(f"subset_bbox with parameters: {self.params}")
            # bounds are always ascending, so if lon is descending rolling will not work.
            ds, lb, ub = cf_convert_between_lon_frames(self.ds, self.params.get("lon_bnds"))
            self.params["lon_bnds"] = (lb, ub)
            try:
                kwargs = {}
                valid_args = [
                    "lon_bnds",
                    "lat_bnds",
                    "start_date",
                    "end_date",
                    "first_level",
                    "last_level",
                    "time_values",
                    "level_values",
                ]
                for arg in valid_args:
                    kwargs.setdefault(arg, self.params.get(arg, None))
                result = subset_bbox(ds, **kwargs)
            except NotImplementedError:
                lon_min, lon_max = lon.values.min(), lon.values.max()
                raise Exception(
                    f"The requested longitude subset {self.params.get('lon_bnds')} is not within the longitude bounds "
                    "of this dataset and the data could not be converted to this longitude frame successfully. "
                    "Please re-run your request with longitudes within the bounds of the dataset: "
                    f"({lon_min:.2f}, {lon_max:.2f})"
                )
        else:
            kwargs = {}
            valid_args = ["start_date", "end_date"]
            for arg in valid_args:
                kwargs.setdefault(arg, self.params.get(arg, None))

            # Subset over time interval if requested
            if any(kwargs.values()):
                logger.debug(f"subset_time with parameters: {kwargs}")
                result = subset_time(self.ds, **kwargs)
            # Subset a series of time values if requested
            elif self.params.get("time_values"):
                result = subset_time_by_values(self.ds, time_values=self.params["time_values"])
            else:
                result = self.ds

            # Now test for level subsetting
            kwargs = {}
            valid_args = ["first_level", "last_level"]

            for arg in valid_args:
                kwargs.setdefault(arg, self.params.get(arg, None))

            # Subset with level only
            if any(kwargs.values()):
                # ensure bounds are ascending
                if self.params.get("first_level") > self.params.get("last_level"):
                    first, last = self.params.get("first_level"), self.params.get("last_level")
                    self.params["first_level"], self.params["last_level"] = last, first

                logger.debug(f"subset_level with parameters: {kwargs}")
                result = subset_level(result, **kwargs)

            elif self.params.get("level_values", None):
                kwargs = {"level_values": self.params["level_values"]}
                logger.debug(f"subset_level_by_values with parameters: {kwargs}")
                result = subset_level_by_values(result, **kwargs)

        # Now apply time components if specified
        time_comps = self.params.get("time_components")
        if time_comps:
            logger.debug(f"subset_by_time_components with parameters: {time_comps}")
            result = subset_time_by_components(result, time_components=time_comps)

        return result


def subset(  # noqa: E501
    ds: xr.Dataset | str | Path,
    *,
    time: str | tuple[str, str] | TimeParameter | Series | Interval | None = None,
    area: None
    | (
        str
        | tuple[
            int | float | str,
            int | float | str,
            int | float | str,
            int | float | str,
        ]
        | AreaParameter
    ) = None,
    level: None | (str | tuple[int | float | str, int | float | str] | LevelParameter | Interval) = None,
    time_components: None | (str | dict | TimeComponents | TimeComponentsParameter) = None,
    output_dir: str | Path | None = None,
    output_type="netcdf",
    split_method="time:auto",
    file_namer="standard",
) -> list[xr.Dataset | str]:
    """
    Subset operation.

    Parameters
    ----------
    ds : xarray.Dataset or str or Path
        The dataset to be subsetted, can be a path to a file or an xarray Dataset.
    time : str or Tuple[str, str] or TimeParameter or Series or Interval, optional
        Time parameter for subsetting, can be a string, tuple of strings, TimeParameter instance,
        Series, or Interval. If None, no time subsetting is applied.
    area : str or AreaParameter or Tuple[int or float or str, int or float or str, int or float or str, int or float or str], optional
        Area parameter for subsetting, can be a string, AreaParameter instance, or a tuple of four values
        representing the bounding box (lon_min, lat_min, lon_max, lat_max). If None, no area subsetting is applied.
    level : str or Tuple[int or float or str, int or float or str] or LevelParameter or Interval, optional
        Level parameter for subsetting, can be a string, tuple of two values, LevelParameter instance,
        or Interval. If None, no level subsetting is applied.
    time_components : str or dict or TimeComponentsParameter, optional
        Time components for subsetting, can be a string, dictionary, or TimeComponentsParameter instance.
    output_dir : str or Path, optional
        Directory where the output will be saved. If None, the output will not be saved to a file.
    output_type : {"netcdf", "nc", "zarr", "xarray"}
        The format of the output, can be "netcdf", "nc", "zarr", or "xarray". Default is "netcdf".
    split_method : {"time:auto"}
        Method for splitting the output, currently only supports "time:auto". Default is "time:auto".
    file_namer : {"standard", "simple"}
        The file naming strategy to use for the output files, can be "standard" or "simple". Default is "standard".

    Returns
    -------
    list of xr.Dataset or list of str
        A list of the subsetted outputs in the format selected; str corresponds to file paths if the
        output format selected is a file.

    Notes
    -----
    If you request a selection range (such as level, latitude or longitude) that specifies the lower
    and upper bounds in the opposite direction to the actual coordinate values then clisops.ops.subset
    will detect this issue and reverse your selection before returning the data subset.

    Examples
    --------
    | ds: xarray Dataset or "cmip5.output1.MOHC.HadGEM2-ES.rcp85.mon.atmos.Amon.r1i1p1.latest.tas"
    | time: ("1999-01-01T00:00:00", "2100-12-30T00:00:00") or "2085-01-01T12:00:00Z/2120-12-30T12:00:00Z"
    | area: (-5.,49.,10.,65) or "0.,49.,10.,65" or [0, 49.5, 10, 65] with the order being lon_0, lat_0, lon_1, lat_1
    | level: (1000.,) or "1000/2000" or ("1000.50", "2000.60")
    | time_components: "year:2000,2004,2008|month:01,02" or {"year": (2000, 2004, 2008), "months": (1, 2)}
    | output_dir: "/cache/wps/procs/req0111"
    | output_type: "netcdf"
    | split_method: "time:auto"
    | file_namer: "standard"
    """
    op = Subset(**locals())
    return op.process()
