from pathlib import Path

from roocs_utils.xarray_utils.xarray_utils import open_xr_dataset

from clisops import logging, utils
from clisops.utils.common import expand_wildcards
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices

LOGGER = logging.getLogger(__file__)


class Operation(object):
    """
    Base class for all Operations.
    """

    def __init__(
        self,
        ds,
        file_namer="standard",
        split_method="time:auto",
        output_dir=None,
        output_type="netcdf",
        **params,
    ):
        """
        Constructor for each operation.
        Sets common input parameters as attributes.
        Parameters that are specific to each operation are handled in:
          self._resolve_params()
        """
        self._file_namer = file_namer
        self._split_method = split_method
        self._output_dir = output_dir
        self._output_type = output_type
        self._resolve_dsets(ds)
        self._resolve_params(**params)

    def _resolve_dsets(self, ds):
        """
        Take in the `ds` object and load it as an xarray Dataset if it
        is a path/wildcard. Set the result to `self.ds`.
        """
        if isinstance(ds, (str, Path)):
            ds = expand_wildcards(ds)
            ds = open_xr_dataset(ds)

        self.ds = ds

    def _resolve_params(self, **params):
        """
        Resolve the operation-specific input parameters to `self.params`.
        """
        self.params = params

    def _get_file_namer(self):
        """
        Return the appropriate file namer object.
        """
        namer = get_file_namer(self._file_namer)()
        return namer

    def _calculate(self):
        """
        The `_calculate()` method is implemented within each operation
        sub-class.
        """
        raise NotImplementedError

    def process(self):
        """
        Main processing method used by all sub-classes.

        Returns a list of outputs, which might be:
        - netCDF file paths
        - Zarr file paths
        - xarray Datasets
        """
        # Create an empty list for outputs
        outputs = list()

        # Get the file namer object for naming output files
        # NOTE: It won't be used if the output type required is "xarray"
        namer = self._get_file_namer()

        # Process the xarray Dataset - this will (usually) be lazily evaluated so
        # no actual data will be read
        processed_ds = self._calculate()

        # Work out how many outputs should be created based on the size
        # of the array. Manage this as a list of time slices.
        time_slices = get_time_slices(processed_ds, self._split_method)

        # Loop through each time slice
        for tslice in time_slices:

            # If there is only one time slice and it is None:
            # - then just set the result Dataset to the processed Dataset
            if tslice is None:
                result_ds = processed_ds
            # If there is a time slice then extract the time slice from the
            # processed Dataset
            else:
                result_ds = processed_ds.sel(time=slice(tslice[0], tslice[1]))

            LOGGER.info(f"Processing {self.__class__.__name__} for times: {tslice}")

            # Get the output (file or xarray Dataset)
            # When this is a file: xarray will read all the data and write the file
            output = get_output(result_ds, self._output_type, self._output_dir, namer)
            outputs.append(output)

        return outputs
