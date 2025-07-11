import os
import random
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from clisops import CONFIG
from clisops.exceptions import InvalidParameterValue
from clisops.ops.subset import Subset, subset
from clisops.parameter import (
    area_parameter,
    level_interval,
    level_series,
    time_components,
    time_interval,
    time_parameter,
    time_series,
)
from clisops.utils.dataset_utils import determine_lon_lat_range
from clisops.utils.output_utils import _format_time


def _load_ds(fpath: str | Path):
    if isinstance(fpath, (str, Path)):
        if str(fpath).endswith("*.nc"):
            return xr.open_mfdataset(fpath)
        else:
            return xr.open_dataset(fpath)
    return xr.open_mfdataset(fpath)


def test_subset_no_params(tmpdir, check_output_nc, nimbus):
    """Test subset without area param."""
    result = subset(
        ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_subset_time(nimbus, tmpdir, check_output_nc):
    """Tests clisops subset function with a time subset."""
    result = subset(
        ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
        time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_subset_args_as_parameter_classes(nimbus, tmpdir, check_output_nc):
    """Tests clisops subset function with a time subset with the arguments as parameter classes from roocs-utils."""
    time = time_parameter.TimeParameter(time_interval("2000-01-01T00:00:00", "2020-12-30T00:00:00"))
    area = area_parameter.AreaParameter((0, -90.0, 360.0, 90.0))

    result = subset(
        ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
        time=time,
        area=area,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


@pytest.mark.parametrize(
    "dset",
    [
        "ATLAS_v1_CMIP5",
        "ATLAS_v1_EOBS",
        "ATLAS_v1_ERA5",
        "ATLAS_v0_CORDEX_NAM",
        "ATLAS_v0_CMIP6",
    ],
)
def test_subset_ATLAS_datasets(tmpdir, dset, check_output_nc, mini_esgf_data):
    """Test temporal and spatial subset for several ATLAS datasets."""
    time = time_parameter.TimeParameter(time_interval("2000-01-01T00:00:00", "2020-12-30T00:00:00"))
    area = area_parameter.AreaParameter((0, -90.0, 360.0, 90.0))

    result = subset(
        ds=mini_esgf_data[dset],
        time=time,
        area=area,
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_subset_invalid_time(nimbus, tmpdir):
    """Tests subset with invalid time param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            time=time_interval("yesterday", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )


def test_subset_ds_is_none(tmpdir):
    """Tests subset with ds=None."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=None,
            time=time_interval("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
        )


def test_subset_no_ds(tmpdir):
    """Tests subset with no dataset provided."""
    with pytest.raises(TypeError):
        subset(
            time=time_interval("2020-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(0, -90.0, 360.0, 90.0),
            output_dir=tmpdir,
        )


def test_subset_area_simple_file_name(nimbus, tmpdir, check_output_nc):
    """Tests clisops subset function with an area subset (simple file name)."""
    result = subset(
        ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_subset_area_project_file_name_atlas(tmpdir, check_output_nc, mini_esgf_data):
    """Tests clisops subset function with an area subset (derived file name)."""
    result = subset(
        ds=mini_esgf_data["ATLAS_v1_EOBS_GRID"],
        area=(0.0, 10.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="standard",
    )
    check_output_nc(result, "t_E-OBS_no-expt_mon_19500101-19500101.nc")


def test_subset_area_project_file_name(nimbus, tmpdir, check_output_nc):
    """Tests clisops subset function with an area subset (derived file name)."""
    result = subset(
        ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="standard",
    )
    check_output_nc(result, "tas_mon_HadGEM2-ES_rcp85_r1i1p1_20051216-20301116.nc")


def test_subset_invalid_area(nimbus, tmpdir):
    """Tests subset with invalid area param."""
    with pytest.raises(InvalidParameterValue):
        subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            area=("zero", 49.0, 10.0, 65.0),
            output_dir=tmpdir,
        )


def test_subset_with_time_and_area(nimbus, tmpdir):
    """
    Tests clisops subset function with time and area subsets.

    On completion:
    - assert all dimensions have been reduced.
    """
    start_time, end_time = "2019-01-16", "2020-12-16"
    bbox = (0.0, -80, 170.0, 65.0)

    outputs = subset(
        ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
        time=time_interval(start_time, end_time),
        area=bbox,
        output_dir=tmpdir,
        output_type="xarray",
    )

    ds = outputs[0]

    assert _format_time(ds.time.values.min()) == start_time
    assert _format_time(ds.time.values.max()) == end_time

    assert ds.lon.values.tolist() == [0]
    assert ds.lat.values.tolist() == [35]


def test_subset_4D_data_all_argument_permutations(tmpdir, mini_esgf_data):
    """
    Tests clisops subset function with:
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
    time_input = time_interval("2022-01-01", "2022-06-01")
    level_input = level_interval(1000, 1000)
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
            ds=mini_esgf_data["CMIP6_TA"],
            time=tm,
            area=bbox,
            level=level,
            output_dir=tmpdir,
            output_type="xarray",
        )

        ds = outputs[0]
        assert ds.ta.shape == tuple(expected_shape)


def test_subset_with_multiple_files_tas(tmpdir, check_output_nc, mini_esgf_data):
    """Tests with multiple tas files."""
    result = subset(
        ds=mini_esgf_data["CMIP5_TAS"],
        time=time_interval("2001-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 0.0, 10.0, 65.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_subset_with_multiple_files_zostoga(tmpdir, check_output_nc, mini_esgf_data):
    """Tests with multiple zostoga files."""
    result = subset(
        ds=mini_esgf_data["CMIP5_ZOSTOGA"],
        time=time_interval("2000-01-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_subset_with_multiple_files_rh(tmpdir, check_output_nc, mini_esgf_data):
    """Tests with multiple rh files."""
    result = subset(
        ds=mini_esgf_data["CMIP5_RH"],
        time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0, -90.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_subset_with_tas_series(tmpdir, tas_series, check_output_nc):
    """Test with tas_series fixture."""
    result = subset(
        ds=tas_series(["20", "22", "25"]),
        time=time_interval("2000-07-01T00:00:00", "2020-12-30T00:00:00"),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_time_slices_in_subset_tas(mini_esgf_data):
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
        ds=mini_esgf_data["CMIP5_TAS"],
        time=time_interval(start_time, end_time),
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


def test_time_slices_in_subset_rh(mini_esgf_data):
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"

    time_slices = [
        ("2001-01-16", "2002-09-16"),
        ("2002-10-16", "2004-06-16"),
        ("2004-07-16", "2005-11-16"),
    ]

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size

    with xr.open_mfdataset(mini_esgf_data["CMIP5_RH"]) as ds:
        outputs = subset(
            ds=ds,
            time=time_interval(start_time, end_time),
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
def test_area_within_area_subset(mini_esgf_data):
    area = (0.0, 10.0, 175.0, 90.0)

    outputs = subset(
        ds=mini_esgf_data["CMIP5_TAS"],
        time=time_interval("2001-01-01T00:00:00", "2200-12-30T00:00:00"),
        area=area,
        output_type="xarray",
        file_namer="simple",
    )

    ds = outputs[0]
    assert area[0] <= ds.lon.data <= area[2]
    assert area[1] <= ds.lat.data <= area[3]


def test_area_within_area_subset_cmip6(mini_esgf_data):
    area = (20.0, 10.0, 250.0, 90.0)

    outputs = subset(
        ds=mini_esgf_data["CMIP6_RLDS"],
        time=time_interval("2001-01-01T00:00:00", "2002-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    ds = outputs[0]

    assert area[0] <= ds.lon.data <= area[2]
    assert area[1] <= ds.lat.data <= area[3]
    assert ds.lon.data[0] == 250
    assert np.isclose(ds.lat.data[0], 36.76056)


def test_subset_with_lat_lon_single_values(mini_esgf_data):
    """
    Creates subset where lat and lon only have one value. Then
    subsets that. This tests that the `lat_bnds` and `lon_bnds`
    are not being reversed by the `_check_desc_coords` function in
    `clisops.core.subset`.
    """
    area = (20.0, 10.0, 250.0, 90.0)

    outputs = subset(
        ds=mini_esgf_data["CMIP6_RLDS"],
        time=time_interval("2001-01-01T00:00:00", "2002-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    ds = outputs[0]

    outputs2 = subset(
        ds=ds,
        time=time_interval("2001-01-01T00:00:00", "2002-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    ds2 = outputs2[0]
    assert len(ds2.lat) == 1
    assert len(ds2.lon) == 1


def test_area_within_area_subset_chunked(mini_esgf_data):
    start_time, end_time = "2001-01-01T00:00:00", "2200-12-30T00:00:00"
    area = (0.0, 10.0, 175.0, 90.0)

    config_max_file_size = CONFIG["clisops:write"]["file_size_limit"]
    temp_max_file_size = "10KB"
    CONFIG["clisops:write"]["file_size_limit"] = temp_max_file_size
    outputs = subset(
        ds=mini_esgf_data["CMIP5_TAS"],
        time=time_interval(start_time, end_time),
        area=area,
        output_type="xarray",
        file_namer="simple",
    )
    CONFIG["clisops:write"]["file_size_limit"] = config_max_file_size

    for ds in outputs:
        assert area[0] <= ds.lon.data <= area[2]
        assert area[1] <= ds.lat.data <= area[3]


def test_subset_level(nimbus):
    """Tests clisops subset function with a level subset."""
    # Levels are: 100000, ..., 100
    cmip6_o3 = nimbus.fetch(
        "cmip6/o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc",
    )

    ds = _load_ds(cmip6_o3)

    result1 = subset(ds=cmip6_o3, level=level_interval("100000/100"), output_type="xarray")

    np.testing.assert_array_equal(result1[0].o3.values, ds.o3.values)

    result2 = subset(ds=cmip6_o3, level=level_interval("100/100"), output_type="xarray")

    np.testing.assert_array_equal(result2[0].o3.shape, (1200, 1, 2, 3))

    result3 = subset(ds=cmip6_o3, level=level_interval("101/-23.234"), output_type="xarray")

    np.testing.assert_array_equal(result3[0].o3.values, result2[0].o3.values)


def test_aux_variables():
    """
    Test auxiliary variables are remembered in output dataset
    Have to create a netcdf file with auxiliary variable
    """
    ds = _load_ds("tests/data/test_file.nc")

    assert "do_i_get_written" in ds.variables

    result = subset(
        ds=ds,
        time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_type="xarray",
    )

    assert "do_i_get_written" in result[0].variables


def test_coord_variables_exist(mini_esgf_data):
    """
    Check coord variables e.g. lat/lon when original data
    is on an irregular grid exist in output dataset
    """
    ds = _load_ds(mini_esgf_data["C3S_CMIP5_TSICE"])

    assert "lat" in ds.coords
    assert "lon" in ds.coords

    result = subset(
        ds=mini_esgf_data["C3S_CMIP5_TSICE"],
        time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=(0.0, 10.0, 10.0, 65.0),
        output_type="xarray",
    )

    assert "lat" in result[0].coords
    assert "lon" in result[0].coords


def test_coord_variables_subsetted_i_j(mini_esgf_data):
    """
    Check coord variables e.g. lat/lon when original data
    is on an irregular grid are subsetted correctly in output dataset
    """
    ds = _load_ds(mini_esgf_data["C3S_CMIP5_TSICE"])

    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "i" in ds.dims
    assert "j" in ds.dims

    area = (50, -65.0, 250.0, 65.0)

    result = subset(
        ds=mini_esgf_data["C3S_CMIP5_TSICE"],
        time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    out = result[0].tsice
    assert out.values.shape == (180, 7, 4)

    # all lats and lons (hence i and j) have been dropped in these ranges as they are all masked, only time dim remains
    assert out.where(np.logical_and(out.lon < area[0], out.lon > area[2]), drop=True).values.shape == (180, 0, 0)
    assert out.where(np.logical_and(out.lat < area[1], out.lat > area[3]), drop=True).values.shape == (180, 0, 0)

    mask1 = ~(np.isnan(out.sel(time=out.time[0])))

    assert np.all(out.lon.values[mask1.values] >= area[0])
    assert np.all(out.lon.values[mask1.values] <= area[2])
    assert np.all(out.lat.values[mask1.values] >= area[1])
    assert np.all(out.lat.values[mask1.values] <= area[3])


def test_coord_variables_subsetted_rlat_rlon(mini_esgf_data):
    """
    Check coord variables e.g. lat/lon when original data
    is on an irregular grid are subsetted correctly in output dataset
    """
    ds = _load_ds(mini_esgf_data["CMIP5_WRONG_CF_UNITS"])

    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert "rlat" in ds.dims
    assert "rlon" in ds.dims

    area = (5.0, 10.0, 20.0, 65.0)

    result = subset(
        ds=mini_esgf_data["CMIP5_WRONG_CF_UNITS"],
        time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
        area=area,
        output_type="xarray",
    )

    out = result[0].zos
    assert out.values.shape == (1, 65, 15)

    # all lats and lons (hence rlat and rlon) have been dropped in these ranges as they are all masked, only time dim remains
    assert out.where(np.logical_and(out.lon < area[0], out.lon > area[2]), drop=True).values.shape == (1, 0, 0)
    assert out.where(np.logical_and(out.lat < area[1], out.lat > area[3]), drop=True).values.shape == (1, 0, 0)

    mask1 = ~(np.isnan(out.sel(time=out.time[0])))

    assert np.all(out.lon.values[mask1.values] >= area[0])
    assert np.all(out.lon.values[mask1.values] <= area[2])
    assert np.all(out.lat.values[mask1.values] >= area[1])
    assert np.all(out.lat.values[mask1.values] <= area[3])


def test_time_invariant_subset_standard_name(tmpdir, check_output_nc, mini_esgf_data):
    result = subset(
        ds=mini_esgf_data["CMIP6_MRSOFC"],
        area=(5.0, 10.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="standard",
    )

    check_output_nc(result, fname="mrsofc_fx_IPSL-CM6A-LR_ssp119_r1i1p1f1_gr.nc")


@pytest.mark.xfail(
    reason="Dataset metadata are badly encoded and not easily handled by h5netcdf",
    strict=False,
)
def test_longitude_and_latitude_coords_only(tmpdir, check_output_nc, mini_esgf_data):
    """Test subset succeeds when latitude and longitude are coordinates not dims and are not called lat/lon"""
    result = subset(
        ds=mini_esgf_data["CMIP6_TOS"],
        area=(10, -70, 260, 70),
        output_dir=tmpdir,
        output_type="nc",
    )

    check_output_nc(
        result,
        fname="tos_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_18500116-18691216.nc",
    )


def test_time_invariant_subset_simple_name(tmpdir, check_output_nc, mini_esgf_data):
    result = subset(
        ds=mini_esgf_data["CMIP6_MRSOFC"],
        area=(5.0, 10.0, 360.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    check_output_nc(result)


def test_time_invariant_subset_with_time(mini_esgf_data):
    with pytest.raises(AttributeError) as exc:
        subset(
            ds=mini_esgf_data["CMIP6_MRSOFC"],
            time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
            area=(5.0, 10.0, 360.0, 90.0),
            output_type="xarray",
        )
    assert str(exc.value) == "'Dataset' object has no attribute 'time'"


def test_cross_prime_meridian(tmpdir, mini_esgf_data, check_output_nc):
    """Test subset with crossing prime meridian"""
    ds = _load_ds(mini_esgf_data["CMIP6_TAS_DAY"])

    result = subset(
        ds=ds,
        area=(-5, 50, 30, 65),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    check_output_nc(result)


def test_do_not_cross_prime_meridian(tmpdir, mini_esgf_data, check_output_nc):
    """Test subset without crossing prime meridian"""
    ds = _load_ds(mini_esgf_data["CMIP6_TAS_DAY"])

    result = subset(
        ds=ds,
        area=(10, 50, 30, 65),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    check_output_nc(result)


def test_0_360_no_cross(tmpdir, mini_esgf_data, check_output_nc):
    ds = _load_ds(mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"])

    result = subset(
        ds=ds,
        area=(10.0, -90.0, 200.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    check_output_nc(result)


def test_0_360_cross(tmpdir, mini_esgf_data, check_output_nc):
    ds = _load_ds(mini_esgf_data["CMIP6_RLDS"])

    result = subset(
        ds=ds,
        area=(-50.0, -90.0, 100.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )
    check_output_nc(result)


def test_300_60_no_cross(tmpdir, mini_esgf_data, check_output_nc):
    # longitude is -300 to 60
    ds = _load_ds(mini_esgf_data["CMIP6_AREACELLO"])

    result = subset(
        ds=ds,
        area=(10.0, -90.0, 50.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    check_output_nc(result)


def test_300_60_cross(tmpdir, mini_esgf_data, check_output_nc):
    # longitude is -300 to 60
    ds = _load_ds(mini_esgf_data["CMIP6_AREACELLO"])

    result = subset(
        ds=ds,
        area=(-100.0, -90.0, 50.0, 90.0),
        output_dir=tmpdir,
        output_type="nc",
        file_namer="simple",
    )

    check_output_nc(result)


def test_roll_positive_real_data(mini_esgf_data):
    ds = _load_ds(mini_esgf_data["CMIP6_RLUS_ONE_TIME_STEP"])

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


def test_roll_positive_mini_data(mini_esgf_data):
    ds = _load_ds(mini_esgf_data["CMIP6_TA"])

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
    assert np.array_equal(result[0].lon.values, [-170.15625, -106.875, -43.59375, 0.0, 63.28125])


def test_lon_alignment_curvilinear_grid(mini_esgf_data):
    ds = _load_ds(mini_esgf_data["CMIP6_SFTOF"])

    # Assert lon frame is (0, 360)
    x0, x1, *y = determine_lon_lat_range(
        ds,
        "longitude",
        "latitude",
        "vertices_longitude",
        "vertices_latitude",
        apply_fix=False,
    )
    assert np.isclose(x0, 0.0, atol=0.1)
    assert np.isclose(x1, 360.0, atol=0.1)

    # Define an area with another lon frame (-180, 180)
    area = (-50.0, -90.0, 100.0, 90.0)

    ds_subset = subset(
        ds=ds,
        area=area,
        output_type="xarray",
    )[0]

    # Assert data includes the desired area
    #  (lon_frame was converted to -180, 180 before subsetting)
    # Since it is a curvilinear grid, the minimum longitude is smaller than -50
    #  and the maximum longitude is larger than 100
    x0, x1, *y = determine_lon_lat_range(
        ds_subset,
        "longitude",
        "latitude",
        "longitude_bnds",
        "latitude_bnds",
        apply_fix=False,
    )
    assert np.isclose(x0, -155.0, atol=0.1)
    assert np.isclose(x1, 159.5, atol=0.1)

    # But we can confirm that the majority of points lie between -50 and 100
    #  (in this case more than 2/3 of the points)
    hist = np.histogram(ds_subset["longitude"], bins=(x0, -51, 101, x1))
    assert hist[0][1] > 2 * (hist[0][0] + hist[0][2])


class TestSubset:
    def test_resolve_params(self, nimbus):
        s = Subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            time=time_interval("1999-01-01T00:00:00", "2100-12-30T00:00:00"),
            area=(-5.0, 49.0, 10.0, 65),
            level=level_interval(1000.0, 1000.0),
        )

        assert s.params["start_date"] == "1999-01-01T00:00:00"
        assert s.params["end_date"] == "2100-12-30T00:00:00"
        assert s.params["lon_bnds"] == (-5, 10)
        assert s.params["lat_bnds"] == (49, 65)

    def test_resolve_params_time(self, nimbus):
        s = Subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            time=time_interval("1999-01-01", "2100-12"),
            area=(0, -90, 360, 90),
        )
        assert s.params["start_date"] == "1999-01-01T00:00:00"
        assert s.params["end_date"] == "2100-12-31T23:59:59"

    def test_resolve_params_invalid_time(self, nimbus):
        with pytest.raises(InvalidParameterValue):
            Subset(
                ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
                time=time_interval("1999-01-01T00:00:00", "maybe tomorrow"),
                area=(0, -90, 360, 90),
            )
        with pytest.raises(InvalidParameterValue):
            Subset(
                ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
                time=time_interval("", "2100"),
                area=(0, -90, 360, 90),
            )

    def test_resolve_params_area(self, nimbus):
        s = Subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            area=(0, 10, 50, 60),
        )
        assert s.params["lon_bnds"] == (0, 50)
        assert s.params["lat_bnds"] == (10, 60)
        # allow also strings
        s = Subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            area=("0", "10", "50", "60"),
        )
        assert s.params["lon_bnds"] == (0, 50)
        assert s.params["lat_bnds"] == (10, 60)

    def test_map_params_invalid_area(self, nimbus):
        with pytest.raises(InvalidParameterValue):
            Subset(
                ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
                area=(0, 10, 50),
            )
        with pytest.raises(InvalidParameterValue):
            Subset(
                ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
                area=("zero", 10, 50, 60),
            )


def test_end_date_nudged_backwards(mini_esgf_data):
    # use no leap dataset
    ds = _load_ds(mini_esgf_data["CMIP6_SICONC_DAY"])

    end_date = "2012-02-29T12:00:00"

    # check end date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, end_date))
    assert str(exc.value).startswith("invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0")

    result = subset(
        ds=mini_esgf_data["CMIP6_SICONC_DAY"],
        area=(20, 30.0, 150, 70.0),
        time=time_interval("2000-01-01T12:00:00", end_date),
        output_type="xarray",
    )

    # check end date of result is correct
    assert result[0].time.values[-1].strftime() == "2012-02-28 12:00:00"


def test_start_date_nudged_forwards(mini_esgf_data):
    # use no leap dataset
    ds = _load_ds(mini_esgf_data["CMIP6_SICONC_DAY"])

    start_date = "2012-02-29T12:00:00"

    # check start date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, start_date))
    assert str(exc.value).startswith("invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0")

    result = subset(
        ds=mini_esgf_data["CMIP6_SICONC_DAY"],
        area=(20, 30.0, 150, 70.0),
        time=time_interval(start_date, "2014-07-29T12:00:00"),
        output_type="xarray",
    )

    # check start date of result is correct
    assert result[0].time.values[0].strftime() == "2012-03-01 12:00:00"


def test_end_date_nudged_backwards_monthly_data(mini_esgf_data):
    # use no leap dataset
    ds = _load_ds(mini_esgf_data["CMIP6_SICONC"])

    end_date = "2012-02-29T12:00:00"

    # check end date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, end_date))
    assert str(exc.value).startswith("invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0")

    result = subset(
        ds=mini_esgf_data["CMIP6_SICONC"],
        area=(20, 30.0, 150, 70.0),
        time=time_interval("2000-01-01T12:00:00", end_date),
        output_type="xarray",
    )

    # check end date of result is correct
    assert result[0].time.values[-1].strftime() == "2012-02-15 00:00:00"


def test_start_date_nudged_backwards_monthly_data(mini_esgf_data):
    # use no leap dataset
    ds = _load_ds(mini_esgf_data["CMIP6_SICONC"])

    start_date = "2012-02-29T12:00:00"

    # check start date normally raises an error
    with pytest.raises(ValueError) as exc:
        ds.time.sel(time=slice(None, start_date))
    assert str(exc.value).startswith("invalid day number provided in cftime.DatetimeNoLeap(2012, 2, 29, 12, 0, 0, 0")

    result = subset(
        ds=mini_esgf_data["CMIP6_SICONC"],
        area=(20, 30.0, 150, 70.0),
        time=time_interval(start_date, "2014-07-29T12:00:00"),
        output_type="xarray",
    )

    # check start date of result is correct
    assert result[0].time.values[0].strftime() == "2012-03-16 12:00:00"


def test_no_lon_in_range(mini_esgf_data):
    with pytest.raises(Exception) as exc:
        subset(
            ds=mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"],
            area=(8.37, -90, 8.56, 90),
            time=time_interval("2006-01-01T00:00:00", "2099-12-30T00:00:00"),
            output_type="xarray",
        )

    assert (
        str(exc.value) == "There were no valid data points found in the requested subset. Please expand "
        "the area covered by the bounding box, the time period or the level range you have selected."
    )


def test_no_lat_in_range(mini_esgf_data):
    with pytest.raises(Exception) as exc:
        subset(
            ds=mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"],
            area=(0, 39.12, 360, 39.26),
            time=time_interval("2006-01-01T00:00:00", "2099-12-30T00:00:00"),
            output_type="xarray",
        )

    assert (
        str(exc.value) == "There were no valid data points found in the requested subset. Please expand "
        "the area covered by the bounding box, the time period or the level range you have selected."
    )


def test_no_lat_lon_in_range(mini_esgf_data):
    with pytest.raises(Exception) as exc:
        subset(
            ds=mini_esgf_data["CMIP6_RLDS_ONE_TIME_STEP"],
            area=(8.37, 39.12, 8.56, 39.26),
            time=time_interval("2006-01-01T00:00:00", "2099-12-30T00:00:00"),
            output_type="xarray",
        )

    assert (
        str(exc.value) == "There were no valid data points found in the requested subset. Please expand "
        "the area covered by the bounding box, the time period or the level range you have selected."
    )


def test_curvilinear_ds_no_data_in_bbox_real_data(mini_esgf_data):
    ds = _load_ds(mini_esgf_data["CMIP6_TOS_CNRM"])
    with pytest.raises(ValueError) as exc:
        subset(
            ds=ds,
            area="1,40,2,4",
            time=time_interval("2021-01-01/2050-12-31"),
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand the area covered by the bounding box."
    )


def test_curvilinear_ds_no_data_in_bbox_real_data_swap_lat(mini_esgf_data):
    ds = _load_ds(mini_esgf_data["CMIP6_TOS_CNRM"])
    with pytest.raises(ValueError) as exc:
        subset(
            ds=ds,
            area="1,4,2,40",
            time=time_interval("2015-01-01/2050-12-31"),
            output_type="xarray",
        )
    assert (
        str(exc.value)
        == "There were no valid data points found in the requested subset. Please expand the area covered by the bounding box."
    )


def test_curvilinear_ds_no_data_in_bbox(mini_esgf_data):
    with pytest.raises(ValueError) as exc:
        subset(
            ds=mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"],
            area="1,5,1.2,4",
            time=time_interval("2021-01-01/2050-12-31"),
            output_type="xarray",
        )
    assert (
        str(exc.value) == "There were no valid data points found in the requested subset. "
        "Please expand the area covered by the bounding box."
    )


def test_curvilinear_increase_lon_of_bbox(mini_esgf_data):
    result = subset(
        ds=mini_esgf_data["CMIP6_TOS_ONE_TIME_STEP"],
        area="1,40,4,4",
        time=time_interval("2021-01-01/2050-12-31"),
        output_type="xarray",
    )

    assert result


class TestReverseBounds:
    rlds = "CMIP6_RLDS_ONE_TIME_STEP"
    tos = "CMIP6_TOS_ONE_TIME_STEP"

    def test_reverse_lat_regular(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.rlds],
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.rlds],
            area=(20, 45, 240, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lon_regular(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.rlds],
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.rlds],
            area=(240, -45, 20, 45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lon_cross_meridian_regular(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.rlds],
            area=(-70, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.rlds],
            area=(240, -45, -70, 45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lat_and_lon_regular(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.rlds],
            area=(-70, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.rlds],
            area=(240, 45, -70, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].rlds, result_rev[0].rlds)

    def test_reverse_lat_curvilinear(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.tos],
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.tos],
            area=(20, 45, 240, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].tos, result_rev[0].tos)

    def test_reverse_lon_curvilinear(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.tos],
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.tos],
            area=(240, -45, 20, 45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].tos, result_rev[0].tos)

    def test_reverse_lon_cross_meridian_curvilinear(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.tos],
            area=(-70, -45, 120, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.tos],
            area=(120, -45, -70, 45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].tos, result_rev[0].tos)

    def test_reverse_lat_and_lon_curvilinear(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data[self.tos],
            area=(20, -45, 240, 45),
            output_type="xarray",
        )

        result_rev = subset(
            ds=mini_esgf_data[self.tos],
            area=(20, 45, 240, -45),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].tos, result_rev[0].tos)

    def test_reverse_with_desc_lat_lon_regular(self, mini_esgf_data):
        ds = _load_ds(mini_esgf_data[self.rlds])

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
        result_rev = result_rev[0].sortby("lat", ascending=True).sortby("lon", ascending=True)

        np.testing.assert_array_equal(result[0].rlds, result_rev.rlds)

    def test_reverse_with_desc_lat_lon_curvilinear(self, mini_esgf_data):
        ds = _load_ds(mini_esgf_data[self.tos])

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
        result_rev = result_rev[0].sortby("i", ascending=True).sortby("j", ascending=True)

        np.testing.assert_array_equal(result[0].tos, result_rev.tos)

    def test_reverse_level(self, nimbus):
        cmip6_o3 = nimbus.fetch(
            "cmip6/o3_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc",
        )

        result = subset(
            ds=cmip6_o3,
            level=level_interval("100000/100"),
            output_type="xarray",
        )

        result_rev = subset(
            ds=cmip6_o3,
            level=level_interval("100/100000"),
            output_type="xarray",
        )

        np.testing.assert_array_equal(result[0].o3, result_rev[0].o3)

    def test_reverse_time(self, mini_esgf_data):
        result = subset(
            ds=mini_esgf_data["CMIP5_TAS"],
            time=time_interval("2021-01-01/2050-12-31"),
            output_type="xarray",
        )

        assert result[0].time.size == 360

        with pytest.raises(ValueError) as exc:
            subset(
                ds=mini_esgf_data["CMIP5_TAS"],
                time=time_interval("2050-12-31/2021-01-01"),
                output_type="xarray",
            )
        assert str(exc.value) == 'Start date ("2051-01-16T00:00:00") is after end date ("2020-12-16T00:00:00").'


class TestByValues:
    @staticmethod
    def shuffle(lst):
        l_copy = lst[:]
        random.shuffle(l_copy)
        return l_copy

    @staticmethod
    def assert_vars_equal(var_id, *ds_list, extras=None):
        """
        Extract variable/DataArray `var_id` from each Dataset in the `ds_list`.

        Check they are all the same by comparing the arrays and common attributes.
        `extras` is an optional list of extra attributes to check.
        """
        if not extras:
            extras = []

        if len(ds_list) == 1:
            raise Exception("Only one Dataset passed to: _ds_var_check()")

        das = [ds[var_id] for ds in ds_list]
        ref_da = das[0]

        # Create a list of attributes to compare
        attrs = ["standard_name", "long_name", "units", "cell_methods"] + extras

        for da in das[1:]:
            assert da.values.tolist() == ref_da.values.tolist()

            for attr in attrs:
                if attr in ref_da.attrs:
                    assert ref_da.attrs[attr] == da.attrs.get(attr)

    def test_subset_level_by_values_all(self, tmpdir, mini_esgf_data):
        all_levels = [
            100000,
            92500,
            85000,
            70000,
            60000,
            50000,
            40000,
            30000,
            25000,
            20000,
            15000,
            10000,
            7000,
            5000,
            3000,
            2000,
            1000,
            500,
            100,
        ]

        shuffled_1 = self.shuffle(all_levels)
        shuffled_2 = self.shuffle(all_levels)

        # Get various outputs and compare they are the same
        ds_list = [
            subset(
                ds=mini_esgf_data["CMIP6_TA"],
                output_dir=tmpdir,
                output_type="xarray",
                level=level,
            )[0]
            for level in [
                None,
                level_series(all_levels),
                level_interval(all_levels[0], all_levels[-1]),
                level_series(list(reversed(all_levels))),
                level_series(shuffled_1),
                level_series(shuffled_2),
            ]
        ]

        self.assert_vars_equal("plev", *ds_list)

    def test_subset_level_by_values_partial(self, tmpdir, mini_esgf_data):
        some_levels = [
            60000,
            50000,
            40000,
            30000,
            25000,
            20000,
            15000,
            10000,
            7000,
            5000,
        ]

        shuffled_1 = self.shuffle(some_levels)
        shuffled_2 = self.shuffle(some_levels)

        # Get various outputs and compare they are the same
        ds_list = [
            subset(
                ds=mini_esgf_data["CMIP6_TA"],
                output_dir=tmpdir,
                output_type="xarray",
                level=level,
            )[0]
            for level in [
                level_series(some_levels),
                level_interval(some_levels[0], some_levels[-1]),
                level_series(list(reversed(some_levels))),
                level_series(shuffled_1),
                level_series(shuffled_2),
            ]
        ]

        self.assert_vars_equal("plev", *ds_list)

    def test_subset_level_by_values_with_gaps(self, tmpdir, mini_esgf_data):
        picked_levels = [60000, 30000, 25000, 20000, 7000, 5000]

        shuffled_1 = self.shuffle(picked_levels)
        shuffled_2 = self.shuffle(picked_levels)

        # Get various outputs and compare they are the same
        ds_list = [
            subset(
                ds=mini_esgf_data["CMIP6_TA"],
                output_dir=tmpdir,
                output_type="xarray",
                level=level,
            )[0]
            for level in [
                level_series(picked_levels),
                level_series(list(reversed(picked_levels))),
                level_series(shuffled_1),
                level_series(shuffled_2),
            ]
        ]

        self.assert_vars_equal("plev", *ds_list)

    def test_subset_level_by_values_and_bbox(self, tmpdir, mini_esgf_data):
        some_levels = [
            60000,
            50000,
            40000,
            30000,
            25000,
            20000,
            15000,
            10000,
            7000,
            5000,
        ]
        area = (20, 30.0, 150, 70.0)

        shuffled_1 = self.shuffle(some_levels)
        shuffled_2 = self.shuffle(some_levels)

        # Get various outputs and compare they are the same
        ds_list = [
            subset(
                ds=mini_esgf_data["CMIP6_TA"],
                output_dir=tmpdir,
                output_type="xarray",
                level=level,
                area=area,
            )[0]
            for level in [
                level_series(some_levels),
                level_interval(some_levels[0], some_levels[-1]),
                level_series(list(reversed(some_levels))),
                level_series(shuffled_1),
                level_series(shuffled_2),
            ]
        ]

        self.assert_vars_equal("plev", *ds_list)

    def test_subset_time_by_values_all(self, tmpdir, mini_esgf_data):
        all_times = [str(tm) for tm in xr.open_dataset(mini_esgf_data["CMIP6_TA"]).time.values]

        shuffled_1 = self.shuffle(all_times)
        shuffled_2 = self.shuffle(all_times)

        # Get various outputs and compare they are the same
        ds_list = [
            subset(
                ds=mini_esgf_data["CMIP6_TA"],
                output_dir=tmpdir,
                output_type="xarray",
                time=times,
            )[0]
            for times in [
                None,
                time_series(all_times),
                time_interval(all_times[0], all_times[-1]),
                time_series(list(reversed(all_times))),
                time_series(shuffled_1),
                time_series(shuffled_2),
            ]
        ]

        self.assert_vars_equal("time", *ds_list)

    def test_subset_time_by_values_partial(self, tmpdir, mini_esgf_data):
        all_times = [str(tm) for tm in xr.open_dataset(mini_esgf_data["CMIP6_TA"]).time.values]
        some_times = all_times[20:-15]

        shuffled_1 = self.shuffle(some_times)
        shuffled_2 = self.shuffle(some_times)

        # Get various outputs and compare they are the same
        ds_list = [
            subset(
                ds=mini_esgf_data["CMIP6_TA"],
                output_dir=tmpdir,
                output_type="xarray",
                time=times,
            )[0]
            for times in [
                time_interval(some_times[0], some_times[-1]),
                time_series(list(reversed(some_times))),
                time_series(shuffled_1),
                time_series(shuffled_2),
            ]
        ]

        self.assert_vars_equal("time", *ds_list)

    def test_subset_time_by_values_with_gaps(self, tmpdir, mini_esgf_data):
        all_times = [str(tm) for tm in xr.open_dataset(mini_esgf_data["CMIP6_TA"]).time.values]
        some_times = [
            all_times[0],
            all_times[100],
            all_times[4],
            all_times[33],
            all_times[9],
        ]

        shuffled_1 = self.shuffle(some_times)
        shuffled_2 = self.shuffle(some_times)

        # Get various outputs and compare they are the same
        ds_list = [
            subset(
                ds=mini_esgf_data["CMIP6_TA"],
                output_dir=tmpdir,
                output_type="xarray",
                time=times,
            )[0]
            for times in [
                time_series(list(reversed(some_times))),
                time_series(shuffled_1),
                time_series(shuffled_2),
            ]
        ]

        self.assert_vars_equal("time", *ds_list)


def test_subset_by_time_components_year_month(tmpdir, mini_esgf_data):
    # times = ("2015-01-16 12", "MANY MORE", "2024-12-16 12") [120]
    tc1 = time_components(year=(2021, 2022), month=["dec", "jan", "feb"])
    tc2 = time_components(year=(2021, 2022), month=[12, 1, 2])

    kwargs = {"output_dir": tmpdir, "output_type": "xarray"}

    for tc in (tc1, tc2):
        with subset(mini_esgf_data["CMIP6_TA"], time_components=tc, **kwargs)[0] as ds:
            assert set(ds.time.dt.year.values) == {2021, 2022}
            assert set(ds.time.dt.month.values) == {12, 1, 2}


def test_subset_empty(tmpdir, mini_esgf_data):
    # Monthly mean dataset with 360-day calendar
    with pytest.raises(Exception) as exc:
        subset(
            ds=mini_esgf_data["CMIP5_TAS"],
            output_type="nc",
            output_dir=tmpdir,
            time_components="month:12,1,2|day:1,2,3,4,5",
        )
    assert str(exc.value) == "'No timesteps are matching the selection criteria.'"


def test_subset_by_time_components_31_days_360_day(tmpdir, mini_esgf_data):
    # Dataset with 360-day calendar
    tmpdir30 = Path(tmpdir, "ds30")
    tmpdir31 = Path(tmpdir, "ds31")
    os.mkdir(tmpdir30)
    os.mkdir(tmpdir31)

    subset(
        ds=mini_esgf_data["CMIP5_MRSOS_MULTIPLE_TIME_STEPS"],
        output_type="nc",
        output_dir=tmpdir30,
        time_components="month:12,1,2|day:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
    )
    subset(
        ds=mini_esgf_data["CMIP5_MRSOS_MULTIPLE_TIME_STEPS"],
        output_type="nc",
        output_dir=tmpdir31,
        time_components="month:12,1,2|day:1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
    )
    with (
        xr.open_mfdataset(f"{tmpdir30}/*.nc") as ds30,
        xr.open_mfdataset(f"{tmpdir31}/*.nc") as ds31,
    ):
        assert ds30.identical(ds31)
        assert ds30.sizes["time"] == 8490


def test_subset_by_time_components_month_day(tmpdir, mini_esgf_data):
    # CMIP6_SICONC_DAY: 18500101-20141231 ;  n_times = 60225
    tc1 = time_components(month=["jul"], day=[1, 11, 21])
    tc2 = time_components(month=[7], day=[1, 11, 21])

    kwargs = {"output_dir": tmpdir, "output_type": "xarray"}

    for tc in (tc1, tc2):
        with subset(mini_esgf_data["CMIP6_SICONC_DAY"], time_components=tc, **kwargs)[0] as ds:
            assert set(ds.time.dt.month.values) == {7}
            assert set(ds.time.dt.day.values) == {1, 11, 21}
            assert len(ds.time.values) == (2014 - 1850 + 1) * 3


def test_subset_by_time_interval_and_components_month_day(tmpdir, mini_esgf_data):
    # CMIP6_SICONC_DAY: 18500101-20141231 ;  n_times = 60225
    ys, ye = 1850, 1869
    ti = time_interval(f"{ys}-01-01T00:00:00", f"{ye}-12-31T23:59:59")

    months = [3, 4, 5]
    days = [5, 6]

    tc1 = time_components(month=["mar", "apr", "may"], day=days)
    tc2 = time_components(month=months, day=days)

    kwargs = {"output_dir": tmpdir, "output_type": "xarray"}

    for tc in (tc1, tc2):
        with subset(mini_esgf_data["CMIP6_SICONC_DAY"], time=ti, time_components=tc, **kwargs)[0] as ds:
            assert set(ds.time.dt.month.values) == set(months)
            assert set(ds.time.dt.day.values) == set(days)
            assert len(ds.time.values) == (ye - ys + 1) * len(months) * len(days)


def test_subset_by_time_series_and_components_month_day(tmpdir, mini_esgf_data):
    # CMIP6_SICONC_DAY: 18500101-20141231 ;  n_times = 60225
    ys, ye = 1850, 1869
    req_times = [
        tm.isoformat() for tm in xr.open_dataset(mini_esgf_data["CMIP6_SICONC_DAY"]).time.values if ys <= tm.year <= ye
    ]

    ts = time_series(req_times)
    months = [3, 4, 5]
    days = [5, 6]

    tc1 = time_components(month=["mar", "apr", "may"], day=days)
    tc2 = time_components(month=months, day=days)

    kwargs = {"output_dir": tmpdir, "output_type": "xarray"}

    for tc in (tc1, tc2):
        with subset(mini_esgf_data["CMIP6_SICONC_DAY"], time=ts, time_components=tc, **kwargs)[0] as ds:
            assert set(ds.time.dt.month.values) == set(months)
            assert set(ds.time.dt.day.values) == set(days)
            assert len(ds.time.values) == (ye - ys + 1) * len(months) * len(days)


def test_subset_by_area_and_components_month_day(tmpdir, mini_esgf_data):
    # CMIP6_SICONC_DAY: 18500101-20141231 ;  n_times = 60225
    ys, ye = 1850, 1869
    ti = time_interval(f"{ys}-01-01T00:00:00", f"{ye}-12-31T23:59:59")

    months = [3, 4, 5]
    days = [5, 6]

    tc1 = time_components(month=["mar", "apr", "may"], day=days)
    tc2 = time_components(month=months, day=days)

    area = area_parameter.AreaParameter((20, 30.0, 150, 70.0))

    kwargs = {"output_dir": tmpdir, "output_type": "xarray"}

    for tc in (tc1, tc2):
        with subset(
            mini_esgf_data["CMIP6_SICONC_DAY"],
            time=ti,
            time_components=tc,
            area=area,
            **kwargs,
        )[0] as ds:
            assert set(ds.time.dt.month.values) == set(months)
            assert set(ds.time.dt.day.values) == set(days)
            assert len(ds.time.values) == (ye - ys + 1) * len(months) * len(days)


@pytest.mark.thread_unsafe
@pytest.mark.slow
class TestSubsetResultLocking:
    def test_subset_nc_no_fill_value(self, nimbus, tmpdir, mini_esgf_data):
        """Tests clisops subset function with a time subset."""
        # FIXME: The `subset` operation relies on opening a file with xarray as a reference.
        # This can cause issues when modifications are done to the file from other processes.
        result = subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )

        # check that with just opening the file with xarray, saving to netcdf, then opening again, these fill values get added
        with _load_ds(mini_esgf_data["CMIP5_TAS"]) as ds:
            ds.to_netcdf(f"{tmpdir}/test_fill_values.nc")

        with _load_ds(f"{tmpdir}/test_fill_values.nc") as ds:
            # assert np.isnan(float(ds.time.encoding.get("_FillValue")))
            assert np.isnan(float(ds.lat.encoding.get("_FillValue")))
            assert np.isnan(float(ds.lon.encoding.get("_FillValue")))
            assert np.isnan(float(ds.height.encoding.get("_FillValue")))

            assert np.isnan(float(ds.lat_bnds.encoding.get("_FillValue")))
            assert np.isnan(float(ds.lon_bnds.encoding.get("_FillValue")))
            assert np.isnan(float(ds.time_bnds.encoding.get("_FillValue")))

        # FIXME: The `subset` operation relies on opening a file with xarray as a reference.
        # This can cause issues when modifications are done to the file from other processes.
        # check that there is no fill value in encoding for coordinate variables and bounds
        with _load_ds(result) as res:
            assert "_FillValue" not in res.time.encoding
            assert "_FillValue" not in res.lat.encoding
            assert "_FillValue" not in res.lon.encoding
            assert "_FillValue" not in res.height.encoding

            assert "_FillValue" not in res.lat_bnds.encoding
            assert "_FillValue" not in res.lon_bnds.encoding
            assert "_FillValue" not in res.time_bnds.encoding

    def test_subset_cmip5_nc_consistent_bounds(self, nimbus, tmpdir):
        """Tests clisops subset function with a time subset and check the metadata"""
        # FIXME: The `subset` operation relies on opening a file with xarray as a reference.
        # This can cause issues when modifications are done to the file from other processes.
        result = subset(
            ds=nimbus.fetch("cmip5/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc"),
            time=time_interval("2005-01-01T00:00:00", "2020-12-30T00:00:00"),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )
        with _load_ds(result) as res:
            # check fill value in bounds
            assert "_FillValue" not in res.lat_bnds.encoding
            assert "_FillValue" not in res.lon_bnds.encoding
            assert "_FillValue" not in res.time_bnds.encoding
            # check fill value in coordinates
            assert "_FillValue" not in res.time.encoding
            assert "_FillValue" not in res.lat.encoding
            assert "_FillValue" not in res.lon.encoding
            assert "_FillValue" not in res.height.encoding
            # check coordinates in bounds
            assert "coordinates" not in res.lat_bnds.encoding
            assert "coordinates" not in res.lon_bnds.encoding
            assert "coordinates" not in res.time_bnds.encoding

    def test_subset_cmip6_nc_consistent_bounds(self, nimbus, tmpdir, mini_esgf_data):
        """Tests clisops subset function with a time subset and check the metadata"""
        # FIXME: The `subset` operation relies on opening a file with xarray as a reference.
        # This can cause issues when modifications are done to the file from other processes.
        result = subset(
            ds=mini_esgf_data["CMIP6_TASMIN"],
            time=time_interval("2010-01-01T00:00:00", "2010-12-31T00:00:00"),
            output_dir=tmpdir,
            output_type="nc",
            file_namer="simple",
        )
        with _load_ds(result) as res:
            # check fill value in bounds
            assert "_FillValue" not in res.lat_bnds.encoding
            assert "_FillValue" not in res.lon_bnds.encoding
            assert "_FillValue" not in res.time_bnds.encoding
            # check fill value in coordinates
            assert "_FillValue" not in res.time.encoding
            assert "_FillValue" not in res.lat.encoding
            assert "_FillValue" not in res.lon.encoding
            assert "_FillValue" not in res.height.encoding
            # check coordinates in bounds
            assert "coordinates" not in res.lat_bnds.encoding
            assert "coordinates" not in res.lon_bnds.encoding
            assert "coordinates" not in res.time_bnds.encoding

    def test_subset_cmip6_issue_308_fillvalue(self, tmpdir, capsys, mini_esgf_data):
        """
        Tests clisops subset function with a time subset and check the metadata.

        Notes
        -----
        This test is used for fillvalue issues. See: https://github.com/roocs/clisops/issues/308

        """
        from clisops.utils.common import enable_logging
        from clisops.utils.testing import ContextLogger

        with ContextLogger():
            enable_logging()
            # FIXME: The `subset` operation relies on opening a file with xarray as a reference.
            # This can cause issues when modifications are done to the file from other processes.
            result = subset(
                ds=mini_esgf_data["CMIP6_FILLVALUE"],
                time=time_interval("2000-01-01T00:00:00", "2000-12-31T00:00:00"),
                output_dir=tmpdir,
                output_type="nc",
                file_namer="simple",
            )
            with _load_ds(result) as res:
                # check fill value in bounds
                assert "_FillValue" not in res.lat_bnds.encoding
                assert "_FillValue" not in res.lon_bnds.encoding
                assert "_FillValue" not in res.time_bnds.encoding
                # xarray should set the appropriate dtype
                assert res.tas.encoding["_FillValue"].dtype == np.float32
                assert res.tas.encoding["missing_value"].dtype == np.float32
                captured = capsys.readouterr()
                assert (
                    "The defined _FillValue and missing_value for 'tas' are not the same '1.0000000200408773e+20' != '1e+20'. Setting '1e+20' for both."
                    in captured.err
                )
