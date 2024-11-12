import os

import pytest

from clisops.utils.file_utils import FileMapper, is_file_list


@pytest.fixture(scope="module")
def cds_domain():
    return "https://data.mips.climate.copernicus.eu"


@pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
def test_file_mapper():
    file_paths = [
        "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19790101-19881231.nc",
        "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19890101-19981231.nc",
    ]
    fm = FileMapper(file_paths)

    assert (
        fm.dirpath
        == "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest"
    )
    assert fm.file_paths == file_paths
    assert fm.file_list == [
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19790101-19881231.nc",
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19890101-19981231.nc",
    ]


def test_file_mapper_different_dirpath():
    file_paths = [
        "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19790101-19881231.nc",
        "/badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19890101-19981231.nc",
    ]

    with pytest.raises(Exception) as exc:
        FileMapper(file_paths)
    assert (
        str(exc.value)
        == "File inputs are not from the same directory so cannot be resolved."
    )


@pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
def test_file_mapper_fake_files():
    file_paths = [
        "/badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19790101-19881231.nc",
        "/badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
        "tas_day_MIROC6_amip_r1i1p1f1_gn_19890101-19981231.nc",
    ]

    with pytest.raises(FileNotFoundError) as exc:
        FileMapper(file_paths)
    assert str(exc.value) == "Some files could not be found."


def test_is_file_list(cds_domain):
    coll = ["/badc/cmip6/fake1.nc", "/badc/cmip6/fake2.nc"]
    assert is_file_list(coll) is True

    coll = [
        f"{cds_domain}/cmip6/fake1.nc",
        f"{cds_domain}/badc/cmip6/fake2.nc",
    ]
    assert is_file_list(coll) is True

    coll = ["fake1.nc", "/badc/cmip6/fake2.nc"]
    assert is_file_list(coll) is False

    with pytest.raises(Exception) as exc:
        coll = "/badc/cmip6/fake1.nc"
        is_file_list(coll)
    assert (
        str(exc.value) == "Expected collection as a list, have received <class 'str'>"
    )
