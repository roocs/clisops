import sys

import roocs_utils.xarray_utils.xarray_utils as xu


templates = {'cmip5':
    '{__derive__var_id}_{frequency}_{model_id}_{experiment_id}_' \
    'r{realization}i{initialization_method}p{physics_version}_' \
    '{__derive__time_range}.nc'
}

attr_defaults = {'model_id': 'no-model', 'source_id': 'no-model',
    'frequency': 'no-freq', 'experiment': 'no-expt', 
    'realization': 'X', 'initialization_method': 'X', 'physics_version': 'X'}



def get_file_namer(name):
    namers = {
        'standard': StandardFileNamer,
        'simple': SimpleFileNamer
    }

    return namers.get(name, StandardFileNamer)


class _BaseFileNamer(object):

    def __init__(self):
        self._count = 0

    def _get_extension(self, format=None):
        return {'netcdf': 'nc'}.get(format, 'dat')

    def get_file_name(self, ds, format=None):
        self._count += 1
        return f'output_{self._count:03d}.{self._get_extension(format=format)}'


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
        project = None

        if ds.attrs.get('project_id', '').lower() == 'cmip5': 
            project = 'cmip5'

        elif ds.attrs.get('mip_era', '').lower() == 'cmip6':
            project = 'cmip6'

        elif ds.attrs.get('project_id', '').lower() == 'cordex':
            project = 'cordex'

        return templates.get(project, None)

    def _resolve_derived_attrs(self, ds, attrs, template):
        if '__derive__var_id' in template:
            attrs['__derive__var_id'] = xu.get_main_variable(ds)

        if '__derive__time_range' in template:
            attrs['__derive__time_range'] = self._get_time_range(ds)

    def _get_time_range(self, da):
        times = da.time.values
        return times.min().strftime("%Y%m%d") + '-' + times.max().strftime("%Y%m%d") 
