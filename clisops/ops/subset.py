import os

import xarray as xr

from roocs_utils.parameter import parameterise

from clisops import utils, logging
from clisops.core import subset_bbox, subset_time
from clisops.utils.output_utils import get_time_slices, get_format_writer
from clisops.utils.file_namers import get_file_namer

__all__ = [
    "subset",
]

LOGGER = logging.getLogger(__file__)



def _subset(ds, time=None, area=None, level=None):
    LOGGER.debug(f"Mapping parameters: time: {time}, area: {area}, level: {level}")

    args = utils.map_params(ds, time, area, level)

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

    parameters = parameterise.parameterise(time=time, area=area, level=level)

    # Convert all inputs to Xarray Datasets
    if isinstance(ds, str):
        ds = xr.open_mfdataset(ds, use_cftime=True, combine="by_coords")

    time_slices = get_time_slices(ds, *parameters['time'].tuple)
    outputs = []

    namer = get_file_namer(file_namer)()

    for tslice in time_slices:

        LOGGER.info(f'Processing subset for times: {tslice}')
        result_ds = _subset(ds, tslice, parameters['area'].tuple, parameters['level'].tuple)

        fmt_method = get_format_writer(output_type)
        if not fmt_method: outputs.append(result_ds)

        file_name = namer.get_file_name(result_ds, fmt=output_type)

        writer = getattr(result_ds, fmt_method)
        output_path = os.path.join(output_dir, file_name)

        writer(output_path)
        LOGGER.info(f"Wrote output file: {output_path}")

        outputs.append(output_path)

    return outputs
