import xarray as xr

from clisops.utils.file_namers import get_file_namer


def test_SimpleFileNamer():
    s = get_file_namer('simple')()

    checks = [(('my.stuff', None), 'output_001.dat'),
              (('other', 'netcdf'), 'output_002.nc')]

    for args, expected in checks:
        resp = s.get_file_name(*args)
        assert(resp == expected)


def test_StandardFileNamer_no_project_match():
    s = get_file_namer('standard')()

    class Thing(object): pass

    mock_ds = Thing()
    mock_ds.attrs = {} 

    checks = [(mock_ds, 'output_001.dat')]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert(resp == expected)
     

def test_StandardFileNamer_cmip5():
    s = get_file_namer('standard')()

    _ds = xr.open_mfdataset(
              'tests/mini-esgf-data/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc',
              use_cftime=True, combine='by_coords')

    checks = [(_ds, 'tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-22991216.nc')]

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert(resp == expected)


def test_StandardFileNamer_cmip5_use_default_attr_names():
    s = get_file_namer('standard')()

    _ds = xr.open_mfdataset(
              'tests/mini-esgf-data/test_data/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc',
              use_cftime=True, combine='by_coords')

    checks = [(_ds, 'tas_mon_no-model_rcp85_r1i1p1_20051216-22991216.nc')]
    del _ds.attrs['model_id']

    for ds, expected in checks:
        resp = s.get_file_name(ds)
        assert(resp == expected)

