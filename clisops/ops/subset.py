import logging
import os

import xarray as xr

from clisops import utils
from clisops.core import subset_bbox, subset_time

__all__ = [
    "subset",
]


def _subset(dset, time=None, space=None, level=None):
    logging.debug(f"Before mapping args: {time}, {space}, {level}")
    args = utils.map_params(time, space, level)
    if space:
        # subset with space and optionally time
        logging.debug(f"subset_bbox with args: {args}")
        result = subset_bbox(dset, **args)
    else:
        # subset with time only
        logging.debug(f"subset_time with args: {args}")
        result = subset_time(dset, **args)
    return result


def subset(
    dset,
    time=None,
    space=None,
    level=None,
    output_type="netcdf",
    output_dir=None,
    chunk_rules=None,
    filenamer="simple_namer",
):
    """
    Example:
        dset: Xarray Dataset
        time: ("1999-01-01T00:00:00", "2100-12-30T00:00:00")
        space: (-5.,49.,10.,65)
        level: (1000.,)
        output_type: "netcdf"
        output_dir: "/cache/wps/procs/req0111"
        chunk_rules: "time:decade"
        filenamer: "facet_namer"

    :param dset:
    :param time:
    :param space:
    :param level:
    :param output_type:
    :param output_dir:
    :param chunk_rules:
    :param filenamer:
    :return:
    """
    # Convert all inputs to Xarray Datasets
    if isinstance(dset, str):
        dset = xr.open_mfdataset(dset)

    result = _subset(dset, time, space, level)

    if output_type == "netcdf":
        output_path = os.path.join(output_dir, "output.nc")
        result.to_netcdf(output_path)

        logging.info(f"Wrote output file: {output_path}")
        return output_path

    return result
