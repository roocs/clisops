def get_coord_by_attr(dset, attr, value):
    coords = dset.coords

    for coord in coords.values():
        if coord.attrs.get(attr, None) == value:
            return coord

    return None


def get_latitude(dset):
    return get_coord_by_attr(dset, "standard_name", "latitude")


def get_longitude(dset):
    return get_coord_by_attr(dset, "standard_name", "longitude")


def get_main_variable(dset):
    data_dims = [data.dims for var_id, data in dset.variables.items()]
    flat_dims = [dim for sublist in data_dims for dim in sublist]
    results = {}
    for var_id, data in dset.variables.items():
        if var_id in flat_dims:
            continue
        if "bnd" in var_id:
            continue
        else:
            results.update({var_id: data.dims})
    result = max(results, key=results.get)

    if result is None:
        raise Exception("Could not determine main variable")
    else:
        return result
