import os
import sys

from roocs_utils.project_utils import get_project_name
from roocs_utils.xarray_utils import xarray_utils as xu

from clisops import CONFIG, logging
from clisops.utils.output_utils import get_format_extension, get_format_writer

LOGGER = logging.getLogger(__file__)


def get_file_namer(name):
    namers = {"standard": StandardFileNamer, "simple": SimpleFileNamer}

    return namers.get(name, StandardFileNamer)


class _BaseFileNamer(object):
    def __init__(self):
        self._count = 0

    def get_file_name(self, ds, fmt=None):
        self._count += 1
        extension = get_format_extension(fmt)
        return f"output_{self._count:03d}.{extension}"


class SimpleFileNamer(_BaseFileNamer):
    pass


class StandardFileNamer(SimpleFileNamer):
    def _get_project(self, ds):
        try:
            return get_project_name(ds)
        except Exception:
            return None

    def get_file_name(self, ds, fmt="nc"):
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
        try:
            return CONFIG[f"project:{self._get_project(ds)}"]["file_name_template"]
        except Exception:
            return None

    def _resolve_derived_attrs(self, ds, attrs, template, fmt=None):
        if "__derive__var_id" in template:
            attrs["__derive__var_id"] = xu.get_main_variable(ds)

        if "__derive__time_range" in template:
            attrs["__derive__time_range"] = self._get_time_range(ds)

        if "__derive__extension" in template:
            attrs["__derive__extension"] = get_format_extension(fmt)

    def _get_time_range(self, da):
        times = da.time.values
        return times.min().strftime("%Y%m%d") + "-" + times.max().strftime("%Y%m%d")


def get_output(result_ds, output_type, output_dir, file_namer):

    namer = get_file_namer(file_namer)()
    fmt_method = get_format_writer(output_type)

    if not fmt_method:
        LOGGER.info(f"Returning output as {type(result_ds)}")
        return result_ds

    file_name = namer.get_file_name(result_ds, fmt=output_type)

    writer = getattr(result_ds, fmt_method)
    output_path = os.path.join(output_dir, file_name)

    writer(output_path)
    LOGGER.info(f"Wrote output file: {output_path}")
    return output_path
