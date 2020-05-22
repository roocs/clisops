from clisops.utils.get_data import _get_xy


def map_args(dset, **kwargs):
    args = {}

    for key, value in kwargs.items():

        if value is None:
            pass
        elif key == "space":
            args.update(_get_xy(dset, value))
        elif key == "level":
            pass
        else:
            args[key] = slice(value[0], value[1])

    return args
