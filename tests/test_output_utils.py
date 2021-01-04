from pathlib import Path

import xarray as xr

from clisops.utils.common import expand_wildcards
from clisops.utils.output_utils import get_time_slices

from ._common import CMIP5_RH, CMIP5_TAS


def _open(coll):
    if isinstance(coll, (str, Path)):
        coll = expand_wildcards(coll)
    if len(coll) > 1:
        ds = xr.open_mfdataset(coll, use_cftime=True, combine="by_coords")
    else:
        ds = xr.open_dataset(coll[0], use_cftime=True)
    return ds


def test_get_time_slices_single_slice():

    tas = _open(CMIP5_TAS)

    test_data = [
        (
            tas,
            1000000,
            1,  # Setting file limit to 1000000 bytes
            ("2005-12-16", "2299-12-16"),
        ),
        (tas, None, 1, ("2005-12-16", "2299-12-16")),  # Using size limit from CONFIG
    ]

    split_method = "time:auto"
    for (
        ds,
        limit,
        n_times,
        slices,
    ) in test_data:

        resp = get_time_slices(ds, split_method, file_size_limit=limit)
        assert resp[0] == slices


def test_get_time_slices_multiple_slices():

    tas = _open(CMIP5_TAS)

    test_data = [
        (
            tas,
            16000,
            4,
            ("2005-12-16", "2089-03-16"),
            ("2089-04-16", "2172-06-16"),
            ("2255-11-16", "2299-12-16"),
        ),
        (
            tas,
            32000,
            2,
            ("2005-12-16", "2172-06-16"),
            None,
            ("2172-07-16", "2299-12-16"),
        ),
    ]

    split_method = "time:auto"
    for ds, limit, n_times, first, second, last in test_data:

        resp = get_time_slices(ds, split_method, file_size_limit=limit)
        assert resp[0] == first
        assert resp[-1] == last

        if second:
            assert resp[1] == second
