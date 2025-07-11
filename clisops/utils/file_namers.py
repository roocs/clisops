"""File namers for CLISOPS."""

import xarray as xr

from clisops import CONFIG
from clisops.project_utils import get_project_name
from clisops.utils.dataset_utils import get_main_variable
from clisops.utils.output_utils import get_format_extension


def get_file_namer(name: str) -> object:
    """
    Return the correct filenamer from the provided name.

    Parameters
    ----------
    name : str
        The name of the file namer to return. Options are "standard" or "simple".

    Returns
    -------
    _BaseFileNamer
        The file namer class corresponding to the provided name.
    """
    file_namer = {"standard": StandardFileNamer, "simple": SimpleFileNamer}

    return file_namer.get(name, StandardFileNamer)


class _BaseFileNamer:
    """File namer base class."""

    def __init__(self, replace=None, extra=""):
        self._count = 0
        self._replace = replace
        self._extra = extra

    def get_file_name(self, ds, fmt="nc"):
        """Generate numbered file names."""
        self._count += 1
        extension = get_format_extension(fmt)
        return f"output_{self._count:03d}.{extension}"


class SimpleFileNamer(_BaseFileNamer):
    """
    Simple file namer class.

    Generates numbered file names.
    """

    pass


class StandardFileNamer(SimpleFileNamer):
    """
    Standard file namer class.

    Generates file names based on input dataset.
    """

    @staticmethod
    def _get_project(ds):
        """Gets the project name from the input dataset."""
        return get_project_name(ds)

    def get_file_name(self, ds, fmt="nc") -> str:
        """
        Construct file name.

        Parameters
        ----------
        ds : xr.DataArray | xr.Dataset
            The dataset for which to generate the file name.
        fmt : str
            The format of the output file, by default "nc".

        Returns
        -------
        str
            The generated file name based on the dataset attributes and project configuration.
        """
        template = self._get_template(ds)

        if not template:
            # Default to parent class namer if no method found
            return super().get_file_name(ds, fmt)

        self._count += 1

        attr_defaults = CONFIG[f"project:{self._get_project(ds)}"]["attr_defaults"]
        attrs = attr_defaults.copy()
        attrs.update(ds.attrs)

        self._resolve_derived_attrs(ds, attrs, template, fmt=fmt)
        file_name = template.format(**attrs)

        return file_name

    def _get_template(self, ds):
        """Gets template to use for output file name, based on the project of the dataset."""
        try:
            return CONFIG[f"project:{self._get_project(ds)}"]["file_name_template"]
        except KeyError:
            return None

    def _resolve_derived_attrs(
        self,
        ds: xr.DataArray | xr.Dataset,
        attrs: dict,
        template: dict,
        fmt: str | None = None,
    ) -> None:
        """Finds var_id, time_range and format_extension of dataset and output to generate output file name."""
        if "__derive__var_id" in template:
            attrs["__derive__var_id"] = get_main_variable(ds)

        if "__derive__time_range" in template:
            attrs["__derive__time_range"] = self._get_time_range(ds)

        if "__derive__extension" in template:
            attrs["__derive__extension"] = get_format_extension(fmt)

        if "extra" in template:
            attrs["extra"] = self._extra

        if self._replace:
            for key, value in self._replace.items():
                attrs[key] = value

    @staticmethod
    def _get_time_range(da: xr.DataArray | xr.Dataset) -> str:
        """Finds the time range of the data in the output."""
        try:
            times = da.time.values
            return f"_{times.min().strftime('%Y%m%d')}-{times.max().strftime('%Y%m%d')}"
        # catch where "time" attribute cannot be accessed in ds
        except AttributeError:
            return ""
