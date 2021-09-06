from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from roocs_utils.exceptions import InvalidParameterValue, MissingParameterValue
from roocs_utils.parameter import area_parameter, time_parameter

from clisops import CONFIG
from clisops.ops.subset import Subset, subset
from clisops.utils.file_namers import get_file_namer
from clisops.utils.output_utils import _format_time

from .._common import (
    C3S_CMIP5_TOS,
    C3S_CMIP5_TSICE,
    CMIP5_RH,
    CMIP5_TAS,
    CMIP5_ZOSTOGA,
    CMIP6_MRSOFC,
    CMIP6_RLDS,
    CMIP6_RLDS_ONE_TIME_STEP,
    CMIP6_SICONC,
    CMIP6_SICONC_DAY,
    CMIP6_TA,
    CMIP6_TOS,
    CMIP6_TOS_ONE_TIME_STEP,
)


def _check_output_nc(result, fname="output_001.nc"):
    assert fname in [Path(_).name for _ in result]


def _load_ds(fpath):
    if isinstance(fpath, (str, Path)):
        if fpath.endswith("*.nc"):
            return xr.open_mfdataset(fpath)
        else:
            return xr.open_dataset(fpath)
    return xr.open_mfdataset(fpath)


def test_subset_no_params(cmip5_tas_file, tmpdir):
    """Test subset without area param."""
    result = subset(
        ds=cmip5_tas_file,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_time(cmip5_tas_file, tmpdir):
    """Tests clisops subset function with a time subset."""
    result = subset(
        ds=cmip5_tas_file,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_args_as_parameter_classes(cmip5_tas_file, tmpdir):
    """Tests clisops subset function with a time subset
    with the arguments as parameter classes from roocs-utils."""

    time = time_parameter.TimeParameter(("2000-01-01T00:00:00", "2020-12-30T00:00:00"))
    area = area_parameter.AreaParameter((0, -90.0, 360.0, 90.0))

    result = subset(
        ds=cmip5_tas_file,
        time=time,
        area=area,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_invalid_time(cmip5_tas_file, tmpdir):
    """Tests subset with invalid time param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=cmip5_tas_file,
            time=("yesterday", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )


def test_subset_ds_is_none(tmpdir):
    """Tests subset with ds=None."""
    with pytest.raises(MissingParameterValue):
        subset(
            ds=None,
            time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
        )


def test_subset_no_ds(tmpdir):
    """Tests subset with no dataset provided."""
    with pytest.raises(TypeError):
        subset(
            time=("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
        )


def test_subset_area_simple_file_name(cmip5_tas_file, tmpdir):
    """Tests clisops subset function with a area subset (simple file name)."""
    result = subset(
        ds=cmip5_tas_file,
        area=(0.0, 10.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_area_project_file_name(cmip5_tas_file, tmpdir):
    """Tests clisops subset function with a area subset (derived file name)."""
    result = subset(
        ds=cmip5_tas_file,
        area=(0.0, 10.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="standard",
    )
    _check_output_nc(result, "tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-20301116.nc")


def test_subset_invalid_area(cmip5_tas_file, tmpdir):
    """Tests subset with invalid area param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=cmip5_tas_file,
            area=("zero", 49.0, 10.0, 65.0),
            output_dir=tmpdir,
        )


def test_subset_with_time_and_area(cmip5_tas_file, tmpdir):
    """Tests clisops subset function with time and area subsets.

    On completion:
    - assert all dimensions have been reduced.

    """
    start_time, end_time = ("2019-01-16", "2020-12-16")
    bbox = (0.0, -80, 170.0, 65.0)

    outputs = subset(
        ds=cmip5_tas_file,
        time=(start_time, end_time),
        area=bbox,
        output_dir=tmpdir,
        output_type="xarray",
    )

    ds = outputs[0]

    assert _format_time(ds.time.values.min()) == start_time
    assert _format_time(ds.time.values.max()) == end_time

    assert ds.lon.values.tolist() == [0]
    assert ds.lat.values.tolist() == [35]


def test_subset_4D_data_all_argument_permutations(load_esgf_test_data, tmpdir):
    """Tests clisops subset function with:
    - no args (collection only)
    - time only
    - level only
    - bbox only
    - time + level
    - time + bbox
    - level + bbox
    - time + level + bbox

    On completion:
    - Check the shape of the response

    """
    # Found in file:
    # times = ("2015-01-16 12", "MANY MORE", "2024-12-16 12") [120]
    # plevs = (100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000,
    #          20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100) [19]
    # lats = (-88.9277353522959, -25.9141861518467, 37.1202943109788) [3]
    # lons = (0, 63.28125, 126.5625, 189.84375, 253.125, 316.40625) [6]

    # Requested subset
    time_input = ("2022-01-01", "2022-06-01")
    level_input = (1000, 1000)
    bbox_input = (0.0, -80, 170.0, 65.0)

    # Define a set of inputs and the resulting shape expected
    test_inputs = [
        ["coll only", (None, None, None)],
        ["time only", (time_input, None, None)],
        ["level only", (None, level_input, None)],
        ["bbox only", (None, None, bbox_input)],
        ["time & level", (time_input, level_input, None)],
        ["time & bbox", (time_input, None, bbox_input)],
        ["level & bbox", (None, level_input, bbox_input)],
        ["time, level & bbox", (time_input, level_input, bbox_input)],
    ]

    # Full data shape
    initial_shape = [120, 19, 3, 6]

    # Test each set of inputs, check the output shape (slice) is correct
    for _, inputs in test_inputs:

        expected_shape = initial_shape[:]
        tm, level, bbox = inputs

        if tm:
            expected_shape[0] = 5
        if level:
            expected_shape[1] = 1
        if bbox:
            expected_shape[2:4] = 2, 3

        outputs = subset(
            ds=CMIP6_TA,
            time=tm,
            area=bbox,
            level=level,
            output_dir=tmpdir,
            output_type="xarray",
        )

        ds = outputs[0]
        assert ds.ta.shape == tuple(expected_shape)


def test_subset_with_multiple_files_tas(load_esgf_test_data, tmpdir):
    """Tests with multiple tas files"""
    result = subset(
        ds=CMIP5_TAS,
        time=("2001-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 0.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_multiple_files_zostoga(load_esgf_test_data, tmpdir):
    """Tests with multiple zostoga files"""
    result = subset(
        ds=CMIP5_ZOSTOGA,
        time=("2000-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_multiple_files_rh(load_esgf_test_data, tmpdir):
    """Tests with multiple rh files"""
    result = subset(
        ds=CMIP5_RH,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_subset_with_tas_series(tmpdir, tas_series):
    """Test with tas_series fixture"""
    result = subset(
        ds=tas_series(["20", "22", "25"]),
        time=("2000-07-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    _check_output_nc(result)


def test_time_slices_in_subset_tas(load_esgf_test_data):
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    time_slices = [
        ("2005-12-16", "2040-03-16"),
        ("2040-04-16", "2074-07-16"),
        ("2074-08-16", "2108-10-16"),
        ("2108-11-16", "2143-02-16"),
        ("2143-03-16", "2177-06-16"),
        ("2177-07-16", "2199-12-16"),
    ]

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size

    outputs = subset(
        ds=CMIP5_TAS,
        time=(start_time, end_time),
        area=(0.0, 5.0, 50.0, 90.0),
        output_type="xarray",
        file_namer="simple",
    )
    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    assert _format_time(outputs[0].time.values.min()) >= start_time
    assert _format_time(outputs[-1].time.values.max()) <= end_time

    count = 0
    for _ in outputs:
        assert _format_time(outputs[count].time.values.min()) >= time_slices[count][0]
        assert _format_time(outputs[count].time.values.max()) >= time_slices[count][1]
        count += 1


def test_time_slices_in_subset_rh(load_esgf_test_data):
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    time_slices = [
        ("2001-01-16", "2002-09-16"),
        ("2002-10-16", "2004-06-16"),
        ("2004-07-16", "2005-11-16"),
    ]

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size
    outputs = subset(
        ds=CMIP5_RH,
        time=(start_time, end_time),
        area=(0.0, 5.0, 50.0, 90.0),
        output_type="xarray",
        file_namer="simple",
    )
    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    assert _format_time(outputs[0].time.values.min()) >= start_time
    assert _format_time(outputs[-1].time.values.max()) <= end_time

    count = 0
    for _ in outputs:
        assert _format_time(outputs[count].time.values.min()) >= time_slices[count][0]
        assert _format_time(outputs[count].time.values.max()) >= time_slices[count][1]
        count += 1


# area can be a few degrees out
def test_area_within_area_subset(load_esgf_test_data):
    area = (0.0, 10.0, 175.0, 90.0)

    outputs = subset(
        ds=CMIP5_TAS,
        time=("2001-01-01T00:00:00", "2200-12-30T00:00:00"),
        area=area,
        output_type="xarray",
        file_namer="simple",
    )

    ds = outputs[0]
    assert area[0] <= ds.lon.data <= area[2]
    assert area[1] <= ds.lat.data <= area[3]


def test_area_within_area_subset_cmip6(load_esgf_test_data):
    area = (20.0, 10.0, 250.0, 90.0)

    outputs = subset(
        ds=CMIP6_RLDS,
        time=("2001-01-01T00:00:00", "2002-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    ds = outputs[0]

    assert area[0] <= ds.lon.data <= area[2]
    assert area[1] <= ds.lat.data <= area[3]
    assert ds.lon.data[0] == 250
    assert np.isclose(ds.lat.data[0], 36.76056)


def test_subset_with_lat_lon_single_values(load_esgf_test_data):
    """Creates subset where lat and lon only have one value. Then
    subsets that. This tests that the `lat_bnds` and `lon_bnds`
    are not being reversed by the `_check_desc_coords` function in
    `clisops.core.subset`.
    """
    area = (20.0, 10.0, 250.0, 90.0)

    outputs = subset(
        ds=CMIP6_RLDS,
        time=("2001-01-01T00:00:00", "2002-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    ds = outputs[0]

    outputs2 = subset(
        ds=ds,
        time=("2001-01-01T00:00:00", "2002-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    ds2 = outputs2[0]
    assert len(ds2.lat) == 1
    assert len(ds2.lon) == 1


def test_area_within_area_subset_chunked(load_esgf_test_data):

    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
    area = (0.0, 10.0, 175.0, 90.0)

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size
    outputs = subset(
        ds=CMIP5_TAS,
        time=(start_time, end_time),
        area=area,
        output_type="xarray",
        file_namer="simple",
    )
    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    for ds in outputs:
        assert area[0] <= ds.lon.data <= area[2]
        assert area[1] <= ds.lat.data <= area[3]


def test_subset_level(cmip6_o3):
    """Tests clisops subset function with a level subset."""
    # Levels are: 100000, ..., 100
    ds = _load_ds(cmip6_o3)

    result1 = subset(ds=cmip6_o3, level="100000/100", output_type="xarray")

    np.testing.assert_array_equal(result1[0].o3.values, ds.o3.values)

    result2 = subset(ds=cmip6_o3, level="100/100", output_type="xarray")

    np.testing.assert_array_equal(result2[0].o3.shape, (1200, 1, 2, 3))

    result3 = subset(ds=cmip6_o3, level="101/-23.234", output_type="xarray")

    np.testing.assert_array_equal(result3[0].o3.values, result2[0].o3.values)


def test_aux_variables():
    """
    test auxiliary variables are remembered in output dataset
    Have to create a netcdf file with auxiliary variable
    """

    ds = _load_ds("tests/ops/file.nc")

    assert "do_i_get_written" in ds.variables

    result = subset(
        ds=ds,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_type="xarray",
    )

    assert "do_i_get_written" in result[0].variables


@pytest.mark.skipif(Path("/gws").is_dir() is False, reason="data not available")
def test_coord_variables_exist():
    """
    check coord variables e.g. lat/lon when original data
    is on an irregular grid exist in output dataset
    """
    ds = _load_ds(C3S_CMIP5_TSICE)

    assert "lat" in ds.coords
    assert "lon" in ds.coords

    result = subset(
        ds=C3S_CMIP5_TSICE,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_type="xarray",
    )

    assert "lat" in result[0].coords
    assert "lon" in result[0].coords


@pytest.mark.skipif(Path("/gws").is_dir() is False, reason="data not available")
def test_coord_variables_subsetted_i_j():
    """
    check coord variables e.g. lat/lon when original data
    is on an irregular grid are subsetted correctly in output dataset
    """

    ds = _load_ds(C3S_CMIP5_TSICE)

    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "i" in ds.dims
    assert "j" in ds.dims

    area = (50, -65.0, 250.0, 65.0)

    result = subset(
        ds=C3S_CMIP5_TSICE,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    out = result[0].tsice
    assert out.values.shape == (180, 318, 178)

    # all lats and lons (hence i and j) have been dropped in these ranges as they are all masked, only time dim remains
    assert out.where(
        np.logical_and(out.lon < area[0], out.lon > area[2]), drop=True
    ).values.shape == (180, 0, 0)
    assert out.where(
        np.logical_and(out.lat < area[1], out.lat > area[3]), drop=True
    ).values.shape == (180, 0, 0)

    mask1 = ~(np.isnan(out.sel(time=out.time[0])))

    assert np.all(out.lon.values[mask1.values] >= area[0])
    assert np.all(out.lon.values[mask1.values] <= area[2])
    assert np.all(out.lat.values[mask1.values] >= area[1])
    assert np.all(out.lat.values[mask1.values] <= area[3])


@pytest.mark.skipif(Path("/gws").is_dir() is False, reason="data not available")
def test_coord_variables_subsetted_rlat_rlon():
    """
    check coord variables e.g. lat/lon when original data
    is on an irregular grid are subsetted correctly in output dataset
    """

    ds = _load_ds(C3S_CMIP5_TOS)

    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "rlat" in ds.dims
    assert "rlon" in ds.dims

    area = (5.0, 10.0, 20.0, 65.0)

    result = subset(
        ds=C3S_CMIP5_TOS,
        time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    out = result[0].tos
    assert out.values.shape == (96, 65, 15)

    # all lats and lons (hence rlat and rlon) have been dropped in these ranges as they are all masked, only time dim remains
    assert out.where(
        np.logical_and(out.lon < area[0], out.lon > area[2]), drop=True
    ).values.shape == (96, 0, 0)
    assert out.where(
        np.logical_and(out.lat < area[1], out.lat > area[3]), drop=True
    ).values.shape == (96, 0, 0)

    mask1 = ~(np.isnan(out.sel(time=out.time[0])))

    assert np.all(out.lon.values[mask1.values] >= area[0])
    assert np.all(out.lon.values[mask1.values] <= area[2])
    assert np.all(out.lat.values[mask1.values] >= area[1])
    assert np.all(out.lat.values[mask1.values] <= area[3])


def test_time_invariant_subset_standard_name(load_esgf_test_data, tmpdir):

    result = subset(
        ds=CMIP6_MRSOFC,
        area=(5.0, 10.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="standard",
    )

    _check_output_nc(result, fname="mrsofc_fx_IPSL-CM6A-LR_ssp119_r1i1p1f1_gr.nc")


def test_longitude_and_latitude_coords_only(load_esgf_test_data, tmpdir):
    """Test subset suceeds when latitude and longitude are coordinates not dims and are not called lat/lon"""

    result = subset(
        ds=CMIP6_TOS,
        area=(10, -70, 260, 70),
        output_dir=tmpdir,
        output_type="nc",
    )

    _check_output_nc(
        result,
        fname="tos_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_18500116-18691216.nc",
    )


def test_time_invariant_subset_simple_name(load_esgf_test_data, tmpdir):

    result = subset(
        ds=CMIP6_MRSOFC,
        area=(5.0, 10.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    _check_output_nc(result)


def test_time_invariant_subset_with_time(load_esgf_test_data):

    with pytest.raises(AttributeError) as exc:
        subset(
            ds=CMIP6_MRSOFC,
            time=("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(5.0, 10.0, 360.0, 90.0),
            output_type="xarray",
        )
    assert str(exc.value) == "'Dataset' object has no attribute 'time'"


# test known bug
@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
@pytest.mark.skip(reason="bug no longer exists")
def test_cross_prime_meridian(tmpdir):
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp119/r1i1p1f1/day/tas/gn/v20191016"
        "/tas_day_MIROC6_ssp119_r1i1p1f1_gn_20150101-20241231.nc"
    )

    with pytest.raises(NotImplementedError) as exc:
        subset(
            ds=ds,
            area=(-5, 50, 30, 65),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )
    assert (
        str(exc.value)
        == "Input longitude bounds ([-5. 30.]) cross the 0 degree meridian "
        "but dataset longitudes are all positive."
    )


# test it works when not crossing 0 meridian
@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_do_not_cross_prime_meridian(tmpdir):
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/ScenarioMIP/MIROC/MIROC6/ssp119/r1i1p1f1/day/tas/gn/v20191016"
        "/tas_day_MIROC6_ssp119_r1i1p1f1_gn_20150101-20241231.nc"
    )

    result = subset(
        ds=ds,
        area=(10, 50, 30, 65),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    _check_output_nc(result)


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_0_360_no_cross(tmpdir):

    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803"
        "/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc"
    )
    result = subset(
        ds=ds,
        area=(10.0, -90.0, 200.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    _check_output_nc(result)


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
@pytest.mark.skip(reason="bug no longer exists")
def test_0_360_cross(tmpdir):
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/"
        "rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc"
    )

    with pytest.raises(NotImplementedError):
        subset(
            ds=ds,
            area=(-50.0, -90.0, 100.0, 90.0),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_300_60_no_cross(tmpdir):
    # longitude is -300 to 60
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Ofx/areacello/gn/v20190726/*.nc"
    )

    result = subset(
        ds=ds,
        area=(10.0, -90.0, 50.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    _check_output_nc(result)


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_300_60_cross(tmpdir):
    # longitude is -300 to 60
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Ofx/areacello/gn/v20190726/*.nc"
    )

    result = subset(
        ds=ds,
        area=(-100.0, -90.0, 50.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    _check_output_nc(result)


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_roll_positive_real_data():
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/rlds/gr/v20180803/"
        "rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc"
    )

    area = (-50.0, -90.0, 100.0, 90.0)

    result = subset(
        ds=ds,
        area=area,
        output_type="xarray",
    )

    assert result[0].lon.attrs == ds.lon.attrs

    assert area[0] <= all(result[0].lon.values) <= area[2]
    assert area[1] <= all(result[0].lat.values) <= area[3]

    # check array contains expected values
    assert np.array_equal(result[0].lon.values, np.arange(-50, 102.5, 2.5))


def test_roll_positive_mini_data():
    ds = _load_ds(CMIP6_TA)

    area = (-180.0, -90.0, 120.0, 90.0)

    result = subset(
        ds=ds,
        area=area,
        output_type="xarray",
    )

    assert result[0].lon.attrs == ds.lon.attrs

    assert area[0] <= all(result[0].lon.values) <= area[2]
    assert area[1] <= all(result[0].lat.values) <= area[3]

    # check array contains expected values
    assert np.array_equal(
        result[0].lon.values, [-170.15625, -106.875, -43.59375, 0.0, 63.28125]
    )


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_check_lon_alignment_curvilinear_grid():
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/ScenarioMIP/NCC/NorESM2-MM/ssp370/r1i1p1f1/Ofx/sftof/gn/v20191108/*.nc"
    )

    area = (-50.0, -90.0, 100.0, 90.0)

    with pytest.raises(Exception) as exc:
        subset(
            ds=ds,
            area=area,
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "The requested longitude subset (-50.0, 100.0) is not within the longitude bounds "
        "of this dataset and the data could not be converted to this longitude frame successfully. "
        "Please re-run your request with longitudes within the bounds of the dataset: (0.00, 359.99)"
    )


class TestSubset:
    def test_resolve_params(self, cmip5_tas_file):
        s = Subset(
            ds=cmip5_tas_file,
            time=("1999-01-01T00:00:00", "2100-12-30T00:00:00"),
            area=(-5.0, 49.0, 10.0, 65),
            level=(1000.0, 1000.0),
        )

        assert s.params["start_date"] == "1999-01-01T00:00:00"
        assert s.params["end_date"] == "2100-12-30T00:00:00"
        assert s.params["lon_bnds"] == (-5, 10)
        assert s.params["lat_bnds"] == (49, 65)

    def test_resolve_params_time(self, cmip5_tas_file):
        s = Subset(
            ds=cmip5_tas_file, time=("1999-01-01", "2100-12"), area=(0, -90, 360, 90)
        )
        assert s.params["start_date"] == "1999-01-01T00:00:00"
        assert s.params["end_date"] == "2100-12-30T00:00:00"

    def test_resolve_params_invalid_time(self, cmip5_tas_file):
        with pytest.raises(InvalidParameterValue):
            Subset(
                ds=cmip5_tas_file,
                time=("1999-01-01T00:00:00", "maybe tomorrow"),
                area=(0, -90, 360, 90),
            )
        with pytest.raises(InvalidParameterValue):
            Subset(ds=cmip5_tas_file, time=("", "2100"), area=(0, -90, 360, 90))

    def test_resolve_params_area(self, cmip5_tas_file):
        s = Subset(
            ds=cmip5_tas_file,
            area=(0, 10, 50, 60),
        )
        assert s.params["lon_bnds"] == (0, 50)
        assert s.params["lat_bnds"] == (10, 60)
        # allow also strings
        s = Subset(
            ds=cmip5_tas_file,
            area=("0", "10", "50", "60"),
        )
        assert s.params["lon_bnds"] == (0, 50)
        assert s.params["lat_bnds"] == (10, 60)

    def test_map_params_invalid_area(self, cmip5_tas_file):
        with pytest.raises(InvalidParameterValue):
            Subset(
                ds=cmip5_tas_file,
                area=(0, 10, 50),
            )
        with pytest.raises(InvalidParameterValue):
            Subset(
                ds=cmip5_tas_file,
                area=("zero", 10, 50, 60),
            )


def test_end_date_nudged_backwards():

    # use no leap dataset
    ds = _load_ds(CMIP6_SICONC_DAY)

    end_date = "2012-02-29T12:00:00"

    # check end date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, end_date))
    assert str(exc.value).startswith(
        "invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0"
    )

    result = subset(
        ds=CMIP6_SICONC_DAY,
        area=(20, 30.0, 150, 70.0),
        time=("2000-01-01T12:00:00", end_date),
        output_type="xarray",
    )

    # check end date of result is correct
    assert result[0].time.values[-1].strftime() == "2012-02-28 12:00:00"


def test_start_date_nudged_forwards():

    # use no leap dataset
    ds = _load_ds(CMIP6_SICONC_DAY)

    start_date = "2012-02-29T12:00:00"

    # check start date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, start_date))
    assert str(exc.value).startswith(
        "invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0"
    )

    result = subset(
        ds=CMIP6_SICONC_DAY,
        area=(20, 30.0, 150, 70.0),
        time=(start_date, "2014-07-29T12:00:00"),
        output_type="xarray",
    )

    # check start date of result is correct
    assert result[0].time.values[0].strftime() == "2012-03-01 12:00:00"


def test_end_date_nudged_backwards_monthly_data():

    # use no leap dataset
    ds = _load_ds(CMIP6_SICONC)

    end_date = "2012-02-29T12:00:00"

    # check end date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, end_date))
    assert str(exc.value).startswith(
        "invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0"
    )

    result = subset(
        ds=CMIP6_SICONC,
        area=(20, 30.0, 150, 70.0),
        time=("2000-01-01T12:00:00", end_date),
        output_type="xarray",
    )

    # check end date of result is correct
    assert result[0].time.values[-1].strftime() == "2012-02-15 00:00:00"


def test_start_date_nudged_backwards_monthly_data():

    # use no leap dataset
    ds = _load_ds(CMIP6_SICONC)

    start_date = "2012-02-29T12:00:00"

    # check start date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, start_date))
    assert str(exc.value).startswith(
        "invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0"
    )

    result = subset(
        ds=CMIP6_SICONC,
        area=(20, 30.0, 150, 70.0),
        time=(start_date, "2014-07-29T12:00:00"),
        output_type="xarray",
    )

    # check start date of result is correct
    assert result[0].time.values[0].strftime() == "2012-03-16 12:00:00"


def test_no_lon_in_range():

    with pytest.raises(Exception) as exc:
        subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(8.37, -90, 8.56, 90),
            time=("2006-01-01T00:00:00", "2099-12-30T00:00:00"),
            output_type="xarray",
        )

    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand "
        "the area covered by the bounding box, the time period or the level range you have selected."
    )


def test_no_lat_in_range():

    with pytest.raises(Exception) as exc:
        subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(0, 39.12, 360, 39.26),
            time=("2006-01-01T00:00:00", "2099-12-30T00:00:00"),
            output_type="xarray",
        )

    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand "
        "the area covered by the bounding box, the time period or the level range you have selected."
    )


def test_no_lat_lon_in_range():

    with pytest.raises(Exception) as exc:
        subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(8.37, 39.12, 8.56, 39.26),
            time=("2006-01-01T00:00:00", "2099-12-30T00:00:00"),
            output_type="xarray",
        )

    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand "
        "the area covered by the bounding box, the time period or the level range you have selected."
    )


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_curvilinear_ds_no_data_in_bbox_real_data():
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-CM6-1/ssp245/r1i1p1f2/Omon/tos/gn/v20190219/tos_Omon_CNRM-CM6-1_ssp245_r1i1p1f2_gn_201501-210012.nc"
    )
    with pytest.raises(ValueError) as exc:
        subset(
            ds=ds,
            area="1,40,2,4",
            time="2021-01-01/2050-12-31",
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand the area covered by the bounding box."
    )


@pytest.mark.skipif(Path("/badc").is_dir() is False, reason="data not available")
def test_curvilinear_ds_no_data_in_bbox_real_data_swap_lat():
    ds = _load_ds(
        "/badc/cmip6/data/CMIP6/ScenarioMIP/CNRM-CERFACS/CNRM-CM6-1/ssp245/r1i1p1f2/Omon/tos/gn/v20190219/tos_Omon_CNRM-CM6-1_ssp245_r1i1p1f2_gn_201501-210012.nc"
    )
    with pytest.raises(ValueError) as exc:
        subset(
            ds=ds,
            area="1,4,2,40",
            time="2021-01-01/2050-12-31",
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand the area covered by the bounding box."
    )


def test_curvilinear_ds_no_data_in_bbox():

    with pytest.raises(ValueError) as exc:
        subset(
            ds=CMIP6_TOS_ONE_TIME_STEP,
            area="1,5,1.2,4",
            time="2021-01-01/2050-12-31",
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand the area covered by the bounding box."
    )


def test_curvilinear_increase_lon_of_bbox():

    result = subset(
        ds=CMIP6_TOS_ONE_TIME_STEP,
        area="1,40,4,4",
        time="2021-01-01/2050-12-31",
        output_type="xarray",
    )

    assert result


class TestReverseBounds:
    def test_reverse_lat_regular(self, load_esgf_test_data):
        result = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(20, 45, 240, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lon_regular(self, load_esgf_test_data):
        result = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(240, -45, 20, 45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lon_cross_meridian_regular(self, load_esgf_test_data):
        result = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(-70, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(240, -45, -70, 45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lat_and_lon_regular(self, load_esgf_test_data):
        result = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(-70, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=CMIP6_RLDS_ONE_TIME_STEP,
            area=(240, 45, -70, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lat_curvilinear(self, load_esgf_test_data):
        result = subset(
            ds=CMIP6_TOS_ONE_TIME_STEP,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=CMIP6_TOS_ONE_TIME_STEP,
            area=(20, 45, 240, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].tos, result_rev[0].tos)

    def test_reverse_lon_curvilinear(self, load_esgf_test_data):
        result = subset(
            ds=CMIP6_TOS_ONE_TIME_STEP,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=CMIP6_TOS_ONE_TIME_STEP,
            area=(240, -45, 20, 45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].tos, result_rev[0].tos)

    def test_reverse_lon_cross_meridian_curvilinear(self, load_esgf_test_data):
        # can't roll because ds has a curvilinear grid
        with pytest.raises(Exception) as exc:
            subset(
                ds=CMIP6_TOS_ONE_TIME_STEP,
                area=(-70, -45, 240, 45),
                output_type="xarray",
            )

        # can't roll because ds has a curvilinear grid
        with pytest.raises(Exception) as exc_rev:
            subset(
                ds=CMIP6_TOS_ONE_TIME_STEP,
                area=(240, -45, -70, 45),
                output_type="xarray",
            )

        assert (
            str(exc.value)
            == "The requested longitude subset (-70.0, 240.0) is not within the longitude bounds of this dataset and the data could not be converted to this longitude frame successfully. Please re-run your request with longitudes within the bounds of the dataset: (0.01, 360.00)"
        )
        assert str(exc.value) == str(exc_rev.value)

    def test_reverse_lat_and_lon_curvilinear(self, load_esgf_test_data):
        result = subset(
            ds=CMIP6_TOS_ONE_TIME_STEP,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=CMIP6_TOS_ONE_TIME_STEP,
            area=(20, 45, 240, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].tos, result_rev[0].tos)

    def test_reverse_with_desc_lat_lon_regular(self):
        ds = _load_ds(CMIP6_RLDS_ONE_TIME_STEP)

        result = subset(
            ds=ds,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        # make lat and lon descending
        ds_rev = ds.sortby("lat", ascending=False).sortby("lon", ascending=False)

        result_rev = subset(
            ds=ds_rev,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        # return lat and lon to ascending
        result_rev = (
            result_rev[0].sortby("lat", ascending=True).sortby("lon", ascending=True)
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev.rlds)

    def test_reverse_with_desc_lat_lon_curvilinear(self):
        ds = _load_ds(CMIP6_TOS_ONE_TIME_STEP)

        result = subset(
            ds=ds,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        # make i and j descending
        ds_rev = ds.sortby("i", ascending=False).sortby("j", ascending=False)

        result_rev = subset(
            ds=ds_rev,
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        # return lat and lon to ascending
        result_rev = (
            result_rev[0].sortby("i", ascending=True).sortby("j", ascending=True)
        )

        np.testing.assert_array_equal(result[0].tos, result_rev.tos)

    def test_reverse_level(self, cmip6_o3):
        result = subset(
            ds=cmip6_o3,
            level="100000/100",
            output_type="xarray",
        )

        result_rev = subset(
            ds=cmip6_o3,
            level="100/100000",
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].o3, result_rev[0].o3)

    def test_reverse_time(self, load_esgf_test_data):

        result = subset(
            ds=CMIP5_TAS,
            time="2021-01-01/2050-12-31",
            output_type="xarray",
        )

        assert result[0].time.size == 360

        with pytest.raises(ValueError) as exc:
            subset(
                ds=CMIP5_TAS,
                time="2050-12-31/2021-01-01",
                output_type="xarray",
            )
        assert (
            str(exc.value)
            == 'Start date ("2051-01-16T00:00:00") is after end date ("2020-12-16T00:00:00").'
        )
