import logging
import os

import xarray as xr

from clisops import utils
from clisops.core import subset_bbox, subset_time

__all__ = [
    "subset",
]


def _subset(ds, time=None, area=None, level=None):
    logging.debug(f"Before mapping args: {time}, {area}, {level}")
    args = utils.map_params(time, area, level)
    if area:
        # subset with space and optionally time
        logging.debug(f"subset_bbox with args: {args}")
        result = subset_bbox(ds, **args)
    else:
        # subset with time only
        logging.debug(f"subset_time with args: {args}")
        result = subset_time(ds, **args)
    return result


def subset(
    ds,
    time=None,
    area=None,
    level=None,
    output_type="netcdf",
    output_dir=None,
    chunk_rules=None,
    filenamer="simple_namer",
):
    """
    Example:
        ds: Xarray Dataset
        time: ("1999-01-01T00:00:00", "2100-12-30T00:00:00")
        area: (-5.,49.,10.,65)
        level: (1000.,)
        output_type: "netcdf"
        output_dir: "/cache/wps/procs/req0111"
        chunk_rules: "time:decade"
        filenamer: "facet_namer"

    :param ds:
    :param time:
    :param area:
    :param level:
    :param output_type:
    :param output_dir:
    :param chunk_rules:
    :param filenamer:
    :return:
    """

    # Convert all inputs to Xarray Datasets
    if isinstance(ds, str):
        ds = xr.open_mfdataset(ds)

    result = _subset(ds, time, area, level)

    if output_type == "netcdf":
        output_path = os.path.join(output_dir, "output.nc")
        result.to_netcdf(output_path)

        logging.info(f"Wrote output file: {output_path}")
        return output_path

    return result
