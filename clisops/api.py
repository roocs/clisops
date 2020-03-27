import os
import xarray as xr

from .utils import map_args


def general_subset(dset, time=None, space=None, level=None, output_type="netcdf",
                   output_dir=None, chunk_rules=None, filenamer="simple_namer"):
    """
    Example:
        dset: Xarray Dataset
        time: ("1999-01-01T00:00:00", "2100-12-30T00:00:00")
        space: (-5.,49.,10.,65)
        level: (1000.,)
        output_type: "netcdf"
        output_dir: "/cache/wps/procs/req0111"
        chunk_rules: "time:decade"
        filenamer: "facet_namer"

    :param dset:
    :param time:
    :param space:
    :param level:
    :param output_type:
    :param output_dir:
    :param chunk_rules:
    :param filenamer:
    :return:
    """
    # Convert all inputs to Xarray Datasets
    if isinstance(dset, str):
        dset = xr.open_dataset(dset)

    print(f'[INFO] Before mapping args: {time}, {space}, {level}') 
    args = map_args(dset, time=time, space=space, level=level)

    print(f'[INFO] Calling Xarray selector with: {args}')
    result = dset.sel(**args)

    if output_type == 'netcdf':
        output_path = os.path.join(output_dir, 'output.nc')
        result.to_netcdf(output_path)

        print(f'[INFO] Wrote output file: {output_path}')
        return output_path

    return result
