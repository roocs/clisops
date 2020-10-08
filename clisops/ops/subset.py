import os
import tempfile

import xarray as xr
from roocs_utils.parameter import parameterise
from roocs_utils.xarray_utils import xarray_utils as xu

from clisops import logging, utils
from clisops.core import subset_bbox, subset_level, subset_time
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
    ds,
    time=None,
    area=None,
    level=None,
    output_dir=None,
    output_type="netcdf",
    split_method="time:auto",
    file_namer="standard",
):
    """
    Example:
        ds: Xarray Dataset
        time: ("1999-01-01T00:00:00", "2100-12-30T00:00:00")
        area: (-5.,49.,10.,65)
        level: (1000.,)
        output_dir: "/cache/wps/procs/req0111"
        output_type: "netcdf"
        split_method: "time:auto"
        file_namer: "standard"

    :param ds:
    :param time:
    :param area:
    :param level:
    :param output_dir:
    :param output_type:
    :param split_method:
    :param file_namer:
    :return:
    """

    # Convert all inputs to Xarray Datasets
    if isinstance(ds, str):
        ds = xr.open_mfdataset(ds, use_cftime=True, combine="by_coords")

    LOGGER.debug(f"Mapping parameters: time: {time}, area: {area}, level: {level}")
    args = utils.map_params(ds, time, area, level)

    subset_ds = _subset(ds, args)

    outputs = []
    namer = get_file_namer(file_namer)()

    time_slices = get_time_slices(subset_ds, split_method)

    for tslice in time_slices:

        result_ds = subset_ds.sel(time=slice(tslice[0], tslice[1]))
        LOGGER.info(f"Processing subset for times: {tslice}")

        output = get_output(result_ds, output_type, output_dir, namer)
        outputs.append(output)

    return outputs
