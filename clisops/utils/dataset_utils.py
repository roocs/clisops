import math

import cftime
import numpy as np
from roocs_utils.utils.time_utils import str_to_AnyCalendarDateTime
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_type

from clisops import logging

LOGGER = logging.getLogger(__file__)


def calculate_offset(lon, first_element_value):
    """
    Calculate the number of elements to roll the dataset by in order to have
    longitude from within requested bounds.

    :param lon: longitude coordinate of xarray dataset.
    :param first_element_value: the value of the first element of the longitude array to roll to.
    """
    # get resolution of data
    res = lon.values[1] - lon.values[0]

    # calculate how many degrees to move by to have lon[0] of rolled subset as lower bound of request
    diff = lon.values[0] - first_element_value

    # work out how many elements to roll by to roll data by 1 degree
    index = 1 / res

    # calculate the corresponding offset needed to change data by diff
    offset = int(round(diff * index))

    return offset


def check_lon_alignment(ds, lon_bnds):
    """
    Check whether the longitude subset requested is within the bounds of the dataset. If not try to roll the dataset so
    that the request is. Raise an exception if rolling is not possible.
    """
    low, high = lon_bnds
    lon = get_coord_by_type(ds, "longitude", ignore_aux_coords=False)
    lon = ds.coords[lon.name]
    lon_min, lon_max = lon.values.min(), lon.values.max()

    # handle the case where there is only one longitude
    if len(lon.values) == 1:
        lon_value = ds.lon.values[0]
        if low > lon_value:
            ds.coords[lon.name] = ds.coords[lon.name] + 360
        elif high < lon_value:
            ds.coords[lon.name] = ds.coords[lon.name] - 360
        return ds

    # check if the request is in bounds - return ds if it is
    if (lon_min <= low or np.isclose(low, lon_min, atol=0.5)) and (
        lon_max >= high or np.isclose(high, lon_max, atol=0.5)
    ):
        return ds

    else:
        # check if lon is a dimension
        if lon.name not in ds.dims:
            raise Exception(
                f"The requested longitude subset {lon_bnds} is not within the longitude bounds "
                f"of this dataset and the data could not be converted to this longitude frame successfully. "
                f"Please re-run your request with longitudes within the bounds of the dataset: ({lon_min:.2f}, {lon_max:.2f})"
            )
        # roll the dataset and reassign the longitude values
        else:
            first_element_value = low
            offset = calculate_offset(lon, first_element_value)

            # roll the dataset
            ds_roll = ds.roll(shifts={f"{lon.name}": offset}, roll_coords=True)

            # assign longitude to match the roll and copy attrs
            lon_vals = ds_roll.coords[lon.name].values

            # treat the values differently according to positive/negative offset
            if offset < 0:
                lon_vals[offset:] = lon_vals[offset:] % 360
            else:
                lon_vals[:offset] = lon_vals[:offset] % -360

            ds_roll.coords[lon.name] = lon_vals
            ds_roll.coords[lon.name].attrs = ds.coords[lon.name].attrs
            return ds_roll


def adjust_date_to_calendar(da, date, direction="backwards"):
    """
    Check that the date specified exists in the calendar type of the dataset. If not,
    change the date a day at a time (up to a maximum of 5 times) to find a date that does exist.

    The direction to change the date by is indicated by 'direction'.

    :param da: xarray.Dataset or xarray.DataArray
    :param date: The date to check, as a string.
    :param direction: The direction to move in days to find a date that does exist.
                     'backwards' means the search will go backwards in time until an existing date is found.
                     'forwards' means the search will go forwards in time.
                      The default is 'backwards'.

    :return: (str) The next possible existing date in the calendar of the dataset.
    """
    # turn date into AnyCalendarDateTime object
    d = str_to_AnyCalendarDateTime(date)

    # get the calendar type
    cal = da.cf["time"].data[0].calendar

    for i in range(5):
        try:
            cftime.datetime(
                d.year,
                d.month,
                d.day,
                d.hour,
                d.minute,
                d.second,
                calendar=cal,
            )
            return d.value
        except ValueError:
            if direction == "forwards":
                d.add_day()
            elif direction == "backwards":
                d.sub_day()
            else:
                raise Exception(
                    f"Invalid value for direction: {direction}. This should be either 'backwards' to indicate subtracting a day or 'forwards' for adding a day."
                )

    raise ValueError(
        f"Could not find an existing date near {date} in the calendar: {cal}"
    )
