from clisops import CONFIG

from roocs_utils.utils.common import parse_size
import roocs_utils.xarray_utils.xarray_utils as xu


SUPPORTED_FORMATS = {
    'netcdf': {
         'method': 'to_netcdf',
         'extension': 'nc'
    },
    'nc': {
         'method': 'to_netcdf',
         'extension': 'nc'
    },
    'zarr': {
         'method': 'to_zarr',
         'extension': 'zarr'
    },
    'xarray': {
         'method': None,
         'extension': None
    }
}


def check_format(fmt):
    if fmt not in SUPPORTED_FORMATS:
        raise KeyError(f'Format not recognised: "{fmt}". Must be one of: {SUPPORTED_FORMATS}.')


def get_format_writer(fmt):
    check_format(fmt)
    return SUPPORTED_FORMATS[fmt]['method']


def get_format_extension(fmt):
    check_format(fmt)
    return SUPPORTED_FORMATS[fmt]['extension']


def _format_time(tm):
    return tm.strftime('%Y-%m-%d')


def get_time_slices(ds, file_size_limit=None):
    """
    Take an xarray Dataset, assume it can be split on the time axis into
    a sequence of slices. Use the prescribed file size limit to generate
    a list of ("YYYY-MM-DD", "YYYY-MM-DD") slices so that the output files
    do not (significantly) exceed the file size limit.

    :param ds: xarray Dataset
    :file_size_limit: a string specifying "<number><units>"
    :return: list of tuples of date strings.
    """

    # Use default file size limit if not provided
    if not file_size_limit:
        file_size_limit = parse_size(CONFIG['clisops:write']['file_size_limit'])
   
    var_id = xu.get_main_variable(ds)
    times = ds[var_id].time.values

    n_times = len(times)
    bytes_per_time_step = ds[var_id].nbytes / n_times

    n_slices = n_times * bytes_per_time_step / file_size_limit
    slice_length = int(n_times // n_slices)

    if slice_length == 0:
        raise Exception(f'Unable to calculate slice length for variable "{var_id}".')

    slices = []
    indx = 0
    final_indx = n_times - 1

    while indx <= final_indx:

        start = indx
        indx += slice_length
        end = indx - 1

        if end > final_indx: end = final_indx
        slices.append((f'{_format_time(times[start])}', f'{_format_time(times[end])}'))

    return slices

