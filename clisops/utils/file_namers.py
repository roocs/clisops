import sys

from roocs_utils.project_utils import get_project_name
from roocs_utils.xarray_utils import xarray_utils as xu

from clisops import CONFIG

attr_defaults = {
    "model_id": "no-model",
    "source_id": "no-model",
    "frequency": "no-freq",
    "experiment": "no-expt",
    "realization": "X",
    "initialization_method": "X",
    "physics_version": "X",
}


def get_file_namer(name):
    namers = {"standard": StandardFileNamer, "simple": SimpleFileNamer}

    return namers.get(name, StandardFileNamer)


class _BaseFileNamer(object):
    def __init__(self):
        self._count = 0

    def _get_extension(self, format=None):
        return {"netcdf": "nc"}.get(format, "dat")

    def get_file_name(self, ds, format=None):
        self._count += 1
        return f"output_{self._count:03d}.{self._get_extension(format=format)}"


class SimpleFileNamer(_BaseFileNamer):
    pass


class StandardFileNamer(SimpleFileNamer):
    def get_file_name(self, ds, format=None):
        template = self._get_template(ds)

        if not template:
            # Default to parent class namer if no method found
            return super().get_file_name(ds)

        self._count += 1

        attrs = attr_defaults.copy()
        attrs.update(ds.attrs)

        self._resolve_derived_attrs(ds, attrs, template)
        file_name = template.format(**attrs)

        return file_name

    def _get_template(self, ds):
        try:
            project = get_project_name(ds)
        except Exception:
            project = None

        return CONFIG.get([f"project:{project}"], None).get(
            ["file_name_template"], None
        )

    def _resolve_derived_attrs(self, ds, attrs, template):
        if "__derive__var_id" in template:
            attrs["__derive__var_id"] = xu.get_main_variable(ds)

        if "__derive__time_range" in template:
            attrs["__derive__time_range"] = self._get_time_range(ds)

    def _get_time_range(self, da):
        times = da.time.values
        return times.min().strftime("%Y%m%d") + "-" + times.max().strftime("%Y%m%d")
