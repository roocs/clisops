from pathlib import Path
from typing import List, Optional, Tuple, Union

import xarray as xr
from roocs_utils.xarray_utils.xarray_utils import (
    get_coord_type,
    known_coord_types,
    open_xr_dataset,
)

from clisops import logging, utils
from clisops.core import average
from clisops.utils.common import expand_wildcards
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices

__all__ = [
    "average_over_dims",
]

LOGGER = logging.getLogger(__file__)


def average_over_dims(
    ds,
    dims=None,
    ignore_unfound_dims=False,
    output_dir: Optional[Union[str, Path]] = None,
    output_type="netcdf",
    split_method="time:auto",
    file_namer="standard",
):

    if isinstance(ds, (str, Path)):
        ds = expand_wildcards(ds)
        ds = open_xr_dataset(ds)

    avg_ds = average.average_over_dims(ds, dims, ignore_unfound_dims)

    outputs = list()
    namer = get_file_namer(file_namer)()

    time_slices = get_time_slices(avg_ds, split_method)

    for tslice in time_slices:
        if tslice is None:
            result_ds = avg_ds
        else:
            result_ds = avg_ds.sel(time=slice(tslice[0], tslice[1]))

        LOGGER.info(f"Processing subset for times: {tslice}")

        output = get_output(result_ds, output_type, output_dir, namer)
        outputs.append(output)

    return outputs

    # outputs = list()
    #
    # avg_result = average.average_over_dims(ds, dims, ignore_unfound_dims)
    #
    # namer = get_file_namer(file_namer)()
    #
    # output = get_output(avg_result, output_type, output_dir, namer)
    # outputs.append(output)
    #
    # return output
