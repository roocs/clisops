from pathlib import Path

from roocs_utils.xarray_utils.xarray_utils import open_xr_dataset

from clisops import logging, utils
from clisops.utils.common import expand_wildcards
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices

LOGGER = logging.getLogger(__file__)


class Operation(object):
    def __init__(
        self,
        ds,
        file_namer="standard",
        split_method="time:auto",
        output_dir=None,
        output_type="netcdf",
        **params,
    ):
        self._file_namer = file_namer
        self._split_method = split_method
        self._output_dir = output_dir
        self._output_type = output_type
        self._resolve_dsets(ds)
        self._resolve_params(**params)

    def _resolve_dsets(self, ds):
        if isinstance(ds, (str, Path)):
            ds = expand_wildcards(ds)
            ds = open_xr_dataset(ds)

        self.ds = ds

    def _resolve_params(self, **params):
        self.params = params

    def _set_file_namer(self):
        namer = get_file_namer(self._file_namer)()
        return namer

    def _calculate(self):
        raise NotImplementedError

    def process(self):

        outputs = list()
        namer = self._set_file_namer()

        processed_ds = self._calculate()

        time_slices = get_time_slices(processed_ds, self._split_method)

        for tslice in time_slices:
            if tslice is None:
                result_ds = processed_ds
            else:
                result_ds = processed_ds.sel(time=slice(tslice[0], tslice[1]))

            LOGGER.info(f"Processing subset for times: {tslice}")

            output = get_output(result_ds, self._output_type, self._output_dir, namer)
            outputs.append(output)

        return outputs
