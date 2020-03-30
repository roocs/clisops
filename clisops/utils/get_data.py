__author__ = """Elle Smith"""
__contact__ = "eleanor.smith@stfc.ac.uk"
__copyright__ = "Copyright 2018 United Kingdom Research and Innovation"
__license__ = "BSD"
__version__ = "0.1.0"

from clisops.utils.get_coords import get_latitude, get_longitude


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

    if hasattr(lat, 'name'):
        xy[lat.name] = slice(space[1], space[3])
    if hasattr(lon, 'name'):
        xy[lon.name] = slice(space[0], space[2])

    return xy
