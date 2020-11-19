from roocs_utils.project_utils import get_project_name
from roocs_utils.xarray_utils import xarray_utils as xu

from clisops import CONFIG
from clisops.utils.output_utils import get_format_extension


def get_file_namer(name):
    """ Returns the correct filenamer from the provided name"""
    namers = {"standard": StandardFileNamer, "simple": SimpleFileNamer}

    return namers.get(name, StandardFileNamer)


class _BaseFileNamer(object):
    """ File namer base class"""

    def __init__(self):
        self._count = 0

    def get_file_name(self, ds, fmt=None):
        """ Generate numbered file names """
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

    def _get_project(self, ds):
        """ Gets the project name from the input dataset """

        try:
            return get_project_name(ds)
        except Exception:
            return None

    def get_file_name(self, ds, fmt="nc"):
        """ Constructs file name. """
        template = self._get_template(ds)

        if not template:
            # Default to parent class namer if no method found
            return super().get_file_name(ds)

        self._count += 1

        attr_defaults = CONFIG[f"project:{self._get_project(ds)}"]["attr_defaults"]
        attrs = attr_defaults.copy()
        attrs.update(ds.attrs)

        self._resolve_derived_attrs(ds, attrs, template, fmt=fmt)
        file_name = template.format(**attrs)

        return file_name

    def _get_template(self, ds):
        """ Gets template to use for output file name, based on the project of the dataset. """
        try:
            return CONFIG[f"project:{self._get_project(ds)}"]["file_name_template"]
        except KeyError:
            return None

    def _resolve_derived_attrs(self, ds, attrs, template, fmt=None):
        """
        Finds var_id, time_range and format_extension of dataset and output to
        generate output file name.
        """
        if "__derive__var_id" in template:
            attrs["__derive__var_id"] = xu.get_main_variable(ds)

        if "__derive__time_range" in template:
            attrs["__derive__time_range"] = self._get_time_range(ds)

        if "__derive__extension" in template:
            attrs["__derive__extension"] = get_format_extension(fmt)

    def _get_time_range(self, da):
        """ Finds the time range of the data in the output. """
        times = da.time.values
        return times.min().strftime("%Y%m%d") + "-" + times.max().strftime("%Y%m%d")
