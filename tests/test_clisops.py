""" Tests for clisops library """


def test_general_subset_dset():
    """ Tests clisops api.general_subset function with only a dataset"""
    pass


def test_general_subset_time():
    """ Tests clisops api.general_subset function with a time subset."""
    pass


def test_general_subset_invalid_time():
    """ Tests clisops api.general_subset function with an invalid time subset."""
    pass


def test_general_subset_space():
    """ Tests clisops api.general_subset function with a space subset."""
    pass


def test_general_subset_invalid_space():
    """ Tests clisops api.general_subset function with an invalid space subset."""
    pass


def test_general_subset_level():
    """ Tests clisops api.general_subset function with a level subset."""
    pass


def test_general_subset_invalid_level():
    """ Tests clisops api.general_subset function with an invalid level subset."""
    pass


def test_general_subset_all():
    """ Tests clisops api.general_subset function with time, space, level subsets."""
    pass


def test_general_subset_file_type():
    """ Tests clisops api.general_subset function with a file type that isn't netcdf."""
    pass


def test_get_coord_by_attr_valid():
    """ Tests clisops utils.get_coord_by_attr with a real attribute e.g.
        standard_name or long_name"""
    pass


def test_get_coord_by_attr_invalid():
    """ Tests clisops utils.get_coord_by_attr with an attribute that
        doesn't exist."""
    pass


def test_get_latitude():
    """ Tests clisops utils.get_latitude with a dataset that has
        a latitude coord with standard name latitude."""
    pass


def test_get_latitude_fail():
    """ Tests clisops utils.get_latitude with a dataset on a coord that
    doesn't have the standard name latitude"""
    pass


def test_get_longitude():
    """ Tests clisops utils.get_longitude with a dataset that has
        a latitude coord with standard name longitude."""
    pass


def test_get_longitude_fail():
    """ Tests clisops utils.get_longitude with a dataset on a coord that
    doesn't have the standard name longitude"""
    pass


def test_get_xy_no_space():
    """ Tests clisops utils._get_xy with a dataset but no space
        argument."""
    pass


def test_get_xy_space():
    """ Tests clisops utils._get_xy with a dataset and space
        argument."""
    pass


def test_get_xy_invalid_space():
    """ Tests clisops utils._get_xy with a dataset and space
        argument that is out of the range of the latitudes
        and longitudes."""
    pass


def test_map_args_no_kwargs():
    """ Tests clisops.map_args with no kwargs. """
    pass


def test_map_args_space():
    """ Tests clisops.map_args with only space kwarg."""
    pass


def test_map_args_level():
    """ Tests clisops.map_args with only level kwarg."""
    pass


def test_map_args_level_and_space():
    """ Tests clisops.map_args with level and space kwargs."""
    pass


def test_map_args_include_time():
    """ Tests clisops.map_args with level and space and time kwargs."""
    pass


def test_map_args_all_none():
    """ Tests clisops.map_args with level and space and time kwargs all set to None."""
    pass


def test_map_args_invalid():
    """ Tests clisops.map_args with a kwarg that isn't level, space or time."""
    pass
