import clisops

def drop_lat(ds, *args, **kwargs):
    ds_drop_lat = ds.drop_dims(*args, **kwargs)
    return ds_drop_lat


def update_attrs(ds, *args, **kwargs):
    var_id = clisops.utils.get_main_variable(ds)
    for key, value in kwargs.items():
        ds[var_id].attrs[key] = value
    return ds


def change_data(ds, *args, **kwargs):
    var_id = clisops.utils.get_main_variable(ds)
    ds[var_id].data = ds[var_id].data + args[0]
    return ds
