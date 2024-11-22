from collections import ChainMap
from pathlib import Path
from typing import Optional, Union

import xarray as xr
from loguru import logger

from clisops.utils.common import expand_wildcards
from clisops.utils.dataset_utils import open_xr_dataset
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import get_output, get_time_slices


class Operation:
    """Base class for all Operations."""

    def __init__(
        self,
        ds,
        file_namer: str = "standard",
        split_method: str = "time:auto",
        output_dir: Optional[Union[str, Path]] = None,
        output_type: str = "netcdf",
        **params,
    ):
        """Constructor for each operation.

        Sets common input parameters as attributes.
        Parameters that are specific to each operation are handled in `self._resolve_params()`
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

    def _resolve_params(self, **params) -> None:
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
        """The `_calculate()` method is implemented within each operation subclass."""
        raise NotImplementedError()

    def _remove_str_compression(self, ds):
        """
        netCDF4 datatypes of variable length are decoded to str by xarray<2023.11.0.
        As of xarray 2023.11.0 they are decoded to one of np.dtypes.StrDType (eg. "<U20")
        of variable length and stripped of all encoding settings. In netcdf-c versions >= 4.9.0
        and xarray < 2023.11.0 the latter part needs to be conducted manually to avoid an Exception
        when writing the xarray.Dataset to disk.
        See issue:  https://github.com/Unidata/netcdf4-python/issues/1205
        See PR: https://github.com/roocs/clisops/pull/319
        """
        if isinstance(ds, xr.Dataset):
            varlist = list(ds.coords) + list(ds.data_vars)
        elif isinstance(ds, xr.DataArray):
            varlist = list(ds.coords)

        for var in varlist:
            if "dtype" in ds[var].encoding:
                if ds[var].encoding["dtype"] == str:
                    for en in [
                        "compression",
                        "complevel",
                        "shuffle",
                        "fletcher32",
                        "endian",
                        "zlib",
                    ]:
                        if en in ds[var].encoding:
                            del ds[var].encoding[en]
        return ds

    def _cap_deflate_level(self, ds):
        """
        For CMOR3 / CMIP6 it was investigated which netCDF4 deflate_level should be set to optimize
        the balance between reduction of file size and degradation in performance. The values found
        were deflate_level=1, shuffle=True. To keep the write times at a minimum, compression level 1
        is not exceeded.
        See issue: https://github.com/PCMDI/cmor/issues/403
        """
        if isinstance(ds, xr.Dataset):
            varlist = list(ds.coords) + list(ds.data_vars)
        elif isinstance(ds, xr.DataArray):
            varlist = list(ds.coords)

        for var in varlist:
            complevel = ds[var].encoding.get("complevel", 0)
            compression = ds[var].encoding.get("compression_opts", 0)
            if complevel > 1:
                ds[var].encoding["complevel"] = 1
            elif compression > 1:
                ds[var].encoding["compression_opts"] = 1

        return ds

    @staticmethod
    def _remove_redundant_fill_values(ds):
        """Get coordinate and data variables and remove fill values added by xarray.

        CF-conventions say that coordinate variables cannot have missing values.

        See Also
        --------
        https://github.com/roocs/clisops/issues/224
        """
        if isinstance(ds, xr.Dataset):
            var_list = list(ds.coords) + list(ds.data_vars)
        elif isinstance(ds, xr.DataArray):
            var_list = list(ds.coords)
        else:
            raise ValueError(
                f"Expected xarray.Dataset or xarray.DataArray, got {type(ds)}"
            )

        for var in var_list:
            fval = ChainMap(ds[var].attrs, ds[var].encoding).get("_FillValue", None)
            mval = ChainMap(ds[var].attrs, ds[var].encoding).get("missing_value", None)
            if not fval and not mval:
                ds[var].encoding["_FillValue"] = None
            elif not mval:
                ds[var].encoding["missing_value"] = fval
                ds[var].encoding["_FillValue"] = fval
                ds[var].attrs.pop("_FillValue", None)
            elif not fval:
                ds[var].encoding["_FillValue"] = mval
                ds[var].encoding["missing_value"] = mval
                ds[var].attrs.pop("missing_value", None)
            else:
                # Issue 308 - Assert missing_value and _FillValue are the same
                if fval != mval:
                    ds[var].encoding["_FillValue"] = mval
                    ds[var].encoding["missing_value"] = mval
                    ds[var].attrs.pop("missing_value", None)
                    ds[var].attrs.pop("_FillValue", None)
                    logger.warning(
                        f"The defined _FillValue and missing_value for '{var}' are not the same '{fval}' != '{mval}'. Setting '{mval}' for both."
                    )
        return ds

    @staticmethod
    def _remove_redundant_coordinates_attr(ds):
        """This method removes the coordinates attribute added by xarray.

        Example
        -------
        .. code-block:: cpp

            double time_bnds(time, bnds);
            time_bnds:coordinates = "height";

        Programs like `cdo` will complain about this:

        .. code-block:: shell

            Warning (cdf_set_var): Inconsistent variable definition for time_bnds!

        See Also
        --------
        https://github.com/roocs/clisops/issues/224
        """
        if isinstance(ds, xr.Dataset):
            var_list = list(ds.coords) + list(ds.data_vars)
        elif isinstance(ds, xr.DataArray):
            var_list = list(ds.coords)
        else:
            raise ValueError(
                f"Expected xarray.Dataset or xarray.DataArray, got {type(ds)}"
            )

        for var in var_list:
            c_attr = ChainMap(ds[var].attrs, ds[var].encoding).get("coordinates", None)
            if not c_attr:
                ds[var].encoding["coordinates"] = None
            else:
                ds[var].encoding["coordinates"] = c_attr
                ds[var].attrs.pop("coordinates", None)
        return ds

    def process(self) -> list[Union[xr.Dataset, Path]]:
        """Main processing method used by all subclasses.

        Returns
        -------
        List[Union[xarray.Dataset, os.PathLike]]
            A list of outputs, which might be NetCDF file paths, Zarr file paths, or xarray.Dataset
        """
        # Create an empty list for outputs
        outputs = list()

        # Get the file namer object for naming output files
        # NOTE: It won't be used if the output type required is "xarray"
        namer = self._get_file_namer()

        # Process the xarray Dataset - this will (usually) be lazily evaluated so
        # no actual data will be read
        processed_ds = self._calculate()

        # remove fill values from lat/lon/time if required
        processed_ds = self._remove_redundant_fill_values(processed_ds)
        # remove redundant coordinates from bounds
        processed_ds = self._remove_redundant_coordinates_attr(processed_ds)
        # remove compression for string variables (as it is not supported by netcdf-c >= 4.9.0)
        processed_ds = self._remove_str_compression(processed_ds)
        # cap deflate level at 1
        processed_ds = self._cap_deflate_level(processed_ds)

        # Work out how many outputs should be created based on the size
        # of the array. Manage this as a list of time slices.
        time_slices = get_time_slices(processed_ds, self._split_method)

        # Loop through each time slice
        for tslice in time_slices:
            # If there is only one time slice, and it is None:
            # - then just set the result Dataset to the processed Dataset
            if tslice is None:
                result_ds = processed_ds
            # If there is a time slice then extract the time slice from the
            # processed Dataset
            else:
                result_ds = processed_ds.sel(time=slice(tslice[0], tslice[1]))

            logger.info(f"Processing {self.__class__.__name__} for times: {tslice}")

            # Get the output (file or xarray Dataset)
            # When this is a file: xarray will read all the data and write the file
            output = get_output(result_ds, self._output_type, self._output_dir, namer)
            outputs.append(output)

        return outputs
