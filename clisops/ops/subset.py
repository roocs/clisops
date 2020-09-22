import os
import tempfile

import xarray as xr
from roocs_utils.parameter import parameterise
from roocs_utils.xarray_utils import xarray_utils as xu

from clisops import logging, utils
from clisops.core import subset_bbox, subset_time
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_da, get_output, get_time_slices

__all__ = [
    "subset",
]

LOGGER = logging.getLogger(__file__)


def _subset(ds, args):
    if "lon_bnds" and "lat_bnds" in args:
        # subset with space and optionally time
        LOGGER.debug(f"subset_bbox with parameters: {args}")
        result = subset_bbox(ds, **args)
    else:
        # subset with time only
        LOGGER.debug(f"subset_time with parameters: {args}")
        result = subset_time(ds, **args)
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

    time_slices = get_time_slices(
        ds, split_method, start=args["start_date"], end=args["end_date"]
    )
    outputs = []

    namer = get_file_namer(file_namer)()
    for tslice in time_slices:
        # update args with tslice start and end
        args["start_date"], args["end_date"] = tslice[0], tslice[1]

        LOGGER.info(f"Processing subset for times: {tslice}")
        result_ds = _subset(ds, args)

        output = get_output(result_ds, output_type, output_dir, namer)

        outputs.append(output)

    return outputs
