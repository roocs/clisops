import xarray as xr
import pandas as pd

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


def _format_time(tm, fmt='%Y-%m-%d'):
    # Convert to datetime if time is a numpy datetime
    if not hasattr(tm, 'strftime'):
        tm = pd.to_datetime(str(tm)) 

    return tm.strftime(fmt)


def filter_times_within(times, start=None, end=None):
    """
    Takes an array of datetimes, returning a reduced array if start or end times
    are defined and are within the main array.
    """
    filtered = []

    for tm in times:
        ft = _format_time(tm, '%Y-%m-%dT%H:%M:%S')
        if start is not None and ft < start: continue
        if end is not None and ft > end:
            break

        filtered.append(tm)

    return filtered
 

def get_time_slices(ds, start=None, end=None, file_size_limit=None):
    """
    Take an xarray Dataset or DataArray, assume it can be split on the time axis 
    into a sequence of slices. Optionally, take a start and end date to specify 
    a sub-slice of the main time axis. 

    Use the prescribed file size limit to generate a list of 
    ("YYYY-MM-DD", "YYYY-MM-DD") slices so that the output files do
    not (significantly) exceed the file size limit.

    :param ds: xarray Dataset
    :file_size_limit: a string specifying "<number><units>"
    :return: list of tuples of date strings.
    """

    # Use default file size limit if not provided
    if not file_size_limit:
        file_size_limit = parse_size(CONFIG['clisops:write']['file_size_limit'])
  
    if isinstance(ds, xr.DataArray):
        da = ds
    else: 
        var_id = xu.get_main_variable(ds)
        da = ds[var_id]

    times = filter_times_within(da.time.values, start=start, end=end)
    n_times = len(times)

    if n_times == 0:
        raise Exception('Zero time steps found between {start} and {end}.')

    bytes_per_time_step = da.nbytes / n_times

    n_slices = n_times * bytes_per_time_step / file_size_limit
    slice_length = int(n_times // n_slices)

    if slice_length == 0:
        raise Exception('Unable to calculate slice length for splitting output files.')

    slices = []
    indx = 0
    final_indx = n_times - 1

    while indx <= final_indx:

        start_indx = indx
        indx += slice_length
        end_indx = indx - 1

        if end_indx > final_indx: end_indx = final_indx
        slices.append((f'{_format_time(times[start_indx])}', f'{_format_time(times[end_indx])}'))

    return slices

