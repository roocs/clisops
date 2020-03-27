import clisops

def double_array(da):
    var_id = clisops.utils.get_main_variable(da)
    da[var_id].data = da[var_id].data * 2
    return da


def change_lat_name(da):
    da_update_lat_name = da.rename({'lat': 'silly_lat'})
    return da_update_lat_name
