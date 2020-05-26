from clisops.utils.get_coords import get_latitude
from clisops.utils.get_coords import get_longitude


def _get_xy(dset, space):
    """
    NOTE:
    The order of the values in a bounding box does not seem to be consistent.

    http://wiki.openstreetmap.org/wiki/Bounding_Box

    ...says the order is "left,bottom,right,top", or:

    "min Longitude, min Latitude, max Longitude, max Latitude".

    """
    if not space:
        return {}

    lat = get_latitude(dset)
    lon = get_longitude(dset)

    xy = {}

    if hasattr(lat, "name"):
        xy[lat.name] = slice(space[1], space[3])
    if hasattr(lon, "name"):
        xy[lon.name] = slice(space[0], space[2])

    return xy
