from pathlib import Path
from typing import List, Optional, Tuple, Union

import xarray as xr

from clisops import logging, utils
from clisops.core import subset_bbox, subset_level, subset_time
from clisops.utils.common import expand_wildcards
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices

__all__ = [
    "subset",
]

LOGGER = logging.getLogger(__file__)


def _subset(ds, args):
    if "lon_bnds" and "lat_bnds" in args:
        # subset with space and optionally time and level
        LOGGER.debug(f"subset_bbox with parameters: {args}")
        result = subset_bbox(ds, **args)
    else:
        kwargs = {}
        valid_args = ["start_date", "end_date"]
        for arg in valid_args:
            kwargs.setdefault(arg, args.get(arg, None))

        # subset with time only
        if any(kwargs.values()):
            LOGGER.debug(f"subset_time with parameters: {kwargs}")
            result = subset_time(ds, **kwargs)

        else:
            result = ds

        kwargs = {}
        valid_args = ["first_level", "last_level"]
        for arg in valid_args:
            kwargs.setdefault(arg, args.get(arg, None))

        # subset with level only
        if any(kwargs.values()):
            LOGGER.debug(f"subset_level with parameters: {kwargs}")
            result = subset_level(ds, **kwargs)

    return result


def subset(
    ds: Union[xr.Dataset, str, Path],
    *,
    time: Optional[Union[str, Tuple[str, str]]] = None,
    area: Optional[
        Union[
            str,
            Tuple[
                Union[int, float, str],
                Union[int, float, str],
                Union[int, float, str],
                Union[int, float, str],
            ],
        ]
    ] = None,
    level: Optional[
        Union[str, Tuple[Union[int, float, str], Union[int, float, str]]]
    ] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_type="netcdf",
    split_method="time:auto",
    file_namer="standard",
) -> List[Union[xr.Dataset, str]]:
    """

    Parameters
    ----------
    ds: Union[xr.Dataset, str]
    time: Optional[Union[str, Tuple[str, str]]] = None,
    area: Optional[
        Union[
            str,
            Tuple[
                Union[int, float, str],
                Union[int, float, str],
                Union[int, float, str],
                Union[int, float, str],
            ],
        ]
    ] = None,
    level: Optional[Union[str, Tuple[Union[int, float, str], Union[int, float, str]]]] = None
    output_dir: Optional[Union[str, Path]] = None
    output_type: {"netcdf", "nc", "zarr", "xarray"}
    split_method: {"time:auto"}
    file_namer: {"standard", "simple"}

    Returns
    -------
    List[Union[xr.Dataset, str]]

    Examples
    --------
    | ds: xarray Dataset or "cmip5.output1.MOHC.HadGEM2-ES.rcp85.mon.atmos.Amon.r1i1p1.latest.tas"
    | time: ("1999-01-01T00:00:00", "2100-12-30T00:00:00") or "2085-01-01T12:00:00Z/2120-12-30T12:00:00Z"
    | area: (-5.,49.,10.,65) or "0.,49.,10.,65" or [0, 49.5, 10, 65]
    | level: (1000.,) or "1000/2000" or ("1000.50", "2000.60")
    | output_dir: "/cache/wps/procs/req0111"
    | output_type: "netcdf"
    | split_method: "time:auto"
    | file_namer: "standard"

    """
    if isinstance(ds, (str, Path)):
        ds = expand_wildcards(ds)
        if len(ds) > 1:
            ds = xr.open_mfdataset(ds, use_cftime=True, combine="by_coords")
        else:
            ds = xr.open_dataset(ds[0], use_cftime=True)

    LOGGER.debug(f"Mapping parameters: time: {time}, area: {area}, level: {level}")
    args = utils.map_params(ds, time, area, level)

    subset_ds = _subset(ds, args)

    outputs = list()
    namer = get_file_namer(file_namer)()

    time_slices = get_time_slices(subset_ds, split_method)

    for tslice in time_slices:
        result_ds = subset_ds.sel(time=slice(tslice[0], tslice[1]))
        LOGGER.info(f"Processing subset for times: {tslice}")

        output = get_output(result_ds, output_type, output_dir, namer)
        outputs.append(output)

    return outputs
