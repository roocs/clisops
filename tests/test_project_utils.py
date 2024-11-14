import os

import pytest
import xarray as xr

from clisops import config, project_utils


@pytest.fixture(scope="module")
def cds_domain():
    return "https://data.mips.climate.copernicus.eu"


class TestProjectUtils:

    def test_get_project_name(self, mini_esgf_data):
        # cmip5
        dset = "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga"
        project = project_utils.get_project_name(dset)
        assert project == "cmip5"

        dset = "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc"
        # project = project_utils.get_project_name(dset)
        # assert project == "cmip5"

        with xr.open_mfdataset(
            mini_esgf_data["CMIP5_TAS"],
            use_cftime=True,
            combine="by_coords",
        ) as ds:
            project = project_utils.get_project_name(ds)
            assert project == "cmip5"

        # cmip6
        dset = "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.SImon.siconc.gn.latest"
        project = project_utils.get_project_name(dset)
        assert project == "cmip6"

        with xr.open_mfdataset(
            mini_esgf_data["CMIP6_SICONC"],
            use_cftime=True,
            combine="by_coords",
        ) as ds:
            project = project_utils.get_project_name(ds)
            assert project == "cmip6"

        # tests default for cmip6 path is c3s-cmip6
        dset = "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc/gn/latest/*.nc"
        # project = project_utils.get_project_name(dset)
        # assert project == "c3s-cmip6"

        # c3s-cmip6-decadal
        dset = "c3s-cmip6-decadal.DCPP.MOHC.HadGEM3-GC31-MM.dcppA-hindcast.s1995-r1i1p1f2.Amon.tas.gn.v20200417"
        project = project_utils.get_project_name(dset)
        assert project == "c3s-cmip6-decadal"

        # TODO: This needs to be cleaned up by introducing aliases and/or multiple mapping facets.
        #       Each project should be defined only once in the roocs.ini.
        #       Currently, without the possibility to define aliases, this is not possible
        #       for the ATLAS projects, because the web prefix, the project name in the DRS and the
        #       project name in the netCDF metadata differ from one another ...
        #
        # c3s-cica-atlas
        dset = "c3s-cica-atlas.cd.CMIP6.historical.yr"
        project = project_utils.get_project_name(dset)
        assert project == "c3s-cica-atlas"

        # c3s-cica-atlas 2
        dset = "/pool/data/c3s-cica-atlas/ERA5/psl_ERA5_mon_194001-202212.nc"
        # project = project_utils.get_project_name(dset)
        # assert project == "c3s-cica-atlas"

        # c3s-ipcc-ar6-atlas
        dset = "c3s-ipcc-ar6-atlas.t.CORDEX-ANT.rcp45.mon"
        project = project_utils.get_project_name(dset)
        assert project == "c3s-ipcc-ar6-atlas"

        # c3s-ipcc-ar6-atlas
        dset = "/pool/data/c3s-ipcc-ar6-atlas/CORDEX-ANT/rcp45/pr_CORDEX-ANT_rcp45_mon_200601-210012.nc"
        project = project_utils.get_project_name(dset)
        assert project in ["c3s-ipcc-ar6-atlas", "c3s-ipcc-atlas"]

    @pytest.mark.xfail(reason="outdated")
    def test_get_project_name_badc(self):
        dset = "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc"
        project = project_utils.get_project_name(dset)
        assert project == "cmip5"

    def test_get_project_base_dir(self):
        cmip5_base_dir = project_utils.get_project_base_dir("cmip5")
        assert cmip5_base_dir == "/mnt/lustre/work/kd0956/CMIP5/data/cmip5"

        c3s_cordex_base_dir = project_utils.get_project_base_dir("c3s-cordex")
        assert (
            c3s_cordex_base_dir == "/mnt/lustre/work/ik1017/C3SCORDEX/data/c3s-cordex"
        )

        with pytest.raises(Exception) as exc:
            project_utils.get_project_base_dir("test")
        assert str(exc.value) == "The project supplied is not known."


class TestDatasetMapper:
    dset = "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.SImon.siconc.gn.latest"

    def test_raw(self):

        assert (
            project_utils.DatasetMapper(self.dset).raw
            == "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.SImon.siconc.gn.latest"
        )

    def test_data_path(self):
        assert (
            project_utils.DatasetMapper(self.dset).data_path
            == "/mnt/lustre/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/SImon/siconc/gn/latest"
        )

    def test_ds_id(self):
        assert (
            project_utils.DatasetMapper(self.dset).ds_id
            == "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.SImon.siconc.gn.latest"
        )

    def test_base_dir(self):
        assert (
            project_utils.DatasetMapper(self.dset).base_dir
            == "/mnt/lustre/work/ik1017/CMIP6/data/CMIP6"
        )

    @pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
    def test_files(self):
        assert project_utils.DatasetMapper(self.dset).files == [
            "/badc/cmip6/data/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/SImon/siconc/gn/latest"
            "/siconc_SImon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"
        ]

    @pytest.mark.xfail(reason="outdated")
    def test_fixed_path_mappings(self, write_roocs_cfg, monkeypatch):
        # reload the roocs_config
        monkeypatch.setenv("ROOCS_CONFIG", write_roocs_cfg)
        project_utils.CONFIG = config.reload_config()

        dsm = project_utils.DatasetMapper("proj_test.my.first.test")
        assert dsm._data_path == "/projects/test/proj/first/test/something.nc"
        assert dsm.files == []  # because these do not exist when globbed

        dsm = project_utils.DatasetMapper("proj_test.my.second.test")
        assert dsm._data_path == "/projects/test/proj/second/test/data_*.txt"
        assert dsm.files == []  # because these do not exist when globbed

        dsm = project_utils.DatasetMapper("proj_test.my.unknown")
        assert dsm._data_path == "/projects/test/proj/my/unknown"

        # reset the config
        monkeypatch.delenv("ROOCS_CONFIG")
        project_utils.CONFIG = config.reload_config()

    @pytest.mark.xfail(reason="outdated")
    def test_fixed_path_modifiers(self, write_roocs_cfg, monkeypatch):
        """Tests how modifiers can change the fixed path mappings."""
        # reload the roocs_config
        monkeypatch.setenv("ROOCS_CONFIG", write_roocs_cfg)
        project_utils.CONFIG = config.reload_config()

        dsm = project_utils.DatasetMapper("proj_test.another.sun.test")
        assert dsm._data_path == "/projects/test/proj/good/test/sun.nc"

        # reset the config
        monkeypatch.delenv("ROOCS_CONFIG")
        project_utils.CONFIG = config.reload_config()


@pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
def test_get_filepaths():
    dset = "c3s-cmip6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.SImon.siconc.gn.latest"

    files = project_utils.dset_to_filepaths(dset)
    assert files == [
        "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc"
        "/gn/latest/siconc_SImon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc",
        "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc"
        "/gn/latest/siconc_SImon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc",
    ]

    dset = "c3s-cmip6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.SImon.siconc.gn.latest"

    files_force = project_utils.dset_to_filepaths(dset, force=True)
    assert files_force == [
        "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc"
        "/gn/latest/siconc_SImon_MIROC6_historical_r1i1p1f1_gn_185001-194912.nc",
        "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc"
        "/gn/latest/siconc_SImon_MIROC6_historical_r1i1p1f1_gn_195001-201412.nc",
    ]

    dset = "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc"

    files = project_utils.dset_to_filepaths(dset)
    assert (
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon"
        "/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_217412-219911.nc"
        in files
    )

    dset = "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/*.nc"

    files_force = project_utils.dset_to_filepaths(dset, force=True)
    assert (
        "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon"
        "/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_217412-219911.nc"
        in files_force
    )


class TestDset:
    @pytest.mark.xfail(reason="outdated")
    def test_derive_dset(self):
        from clisops.project_utils import derive_dset

        # c3s-cmip6
        dset = "c3s-cmip6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.SImon.siconc.gn.latest"
        ds_id = derive_dset(dset)

        assert (
            ds_id
            == "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc/gn/latest"
        )

        # cmip5
        dset = (
            "cmip5.output1.ICHEC.EC-EARTH.historical.day.atmos.day.r1i1p1.tas.v20131231"
        )
        ds_id = derive_dset(dset)

        assert (
            ds_id
            == "/badc/cmip5/data/cmip5/output1/ICHEC/EC-EARTH/historical/day/atmos/day/r1i1p1/tas/v20131231"
        )

        # c3s-cmip6-decadal
        dset = "c3s-cmip6-decadal.DCPP.MOHC.HadGEM3-GC31-MM.dcppA-hindcast.s1995-r1i1p1f2.Amon.tas.gn.v20200417"
        ds_id = derive_dset(dset)

        assert (
            ds_id
            == "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1995-r1i1p1f2/Amon/tas/gn/v20200417"
        )

        # c3s-cica-atlas
        dset = "c3s-cica-atlas.cd.CMIP6.historical.yr"
        ds_id = derive_dset(dset)

        assert ds_id == "/pool/data/c3s-cica-atlas/cd/CMIP6/historical/yr"

        # c3s-ipcc-ar6-atlas
        dset = "c3s-ipcc-ar6-atlas.cd.CMIP6.historical.yr"
        ds_id = derive_dset(dset)

        assert ds_id == "/pool/data/c3s-ipcc-ar6-atlas/cd/CMIP6/historical/yr"

    @pytest.mark.xfail(reason="outdated")
    def test_switch_dset(self):
        from clisops.project_utils import switch_dset

        dset = "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc/gn/latest/*.nc"
        ds_id = switch_dset(dset)

        assert (
            ds_id
            == "c3s-cmip6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.SImon.siconc.gn.latest"
        )

    @pytest.mark.xfail(reason="outdated")
    def test_switch_dset_modified_config(self, write_roocs_cfg, monkeypatch):
        # reload the roocs_config
        monkeypatch.setenv("ROOCS_CONFIG", write_roocs_cfg)
        project_utils.CONFIG = config.reload_config()

        dset = "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/SImon/siconc/gn/latest/*.nc"
        ds_id = project_utils.switch_dset(dset)

        # The first match is returned when parsing the projects within the roocs.ini file
        assert (
            ds_id
            == "c3s-cmip6-decadal.CMIP.MIROC.MIROC6.historical.r1i1p1f1.SImon.siconc.gn.latest"
        )

        # reset the config
        monkeypatch.delenv("ROOCS_CONFIG")
        project_utils.CONFIG = config.reload_config()


def test_unknown_fpath_force():
    dset = "/tmp/tmpxi6d78ng/subset_tttaum9d/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_19850116-20141216.nc"

    dm_force = project_utils.DatasetMapper(dset, force=True)

    assert dm_force.files == [
        "/tmp/tmpxi6d78ng/subset_tttaum9d/"
        "rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_19850116-20141216.nc"
    ]
    assert dm_force.data_path == "/tmp/tmpxi6d78ng/subset_tttaum9d"
    assert dm_force.ds_id is None

    files = project_utils.dset_to_filepaths(dset, force=True)
    assert files == [
        "/tmp/tmpxi6d78ng/subset_tttaum9d/"
        "rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_19850116-20141216.nc"
    ]


class TestExceptions:
    from clisops.exceptions import InvalidProject

    def test_unknown_fpath_no_force(self):
        dset = "/tmp/tmpxi6d78ng/subset_tttaum9d/rlds_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_19850116-20141216.nc"

        with pytest.raises(self.InvalidProject) as exc:
            project_utils.DatasetMapper(dset)
        assert (
            str(exc.value)
            == "The project could not be identified and force was set to false"
        )

    def test_unknown_project_no_force(self):
        dset = "unknown_project.data1.data2.data3.data4"

        with pytest.raises(self.InvalidProject) as exc:
            project_utils.DatasetMapper(dset)
        assert (
            str(exc.value)
            == "The project could not be identified and force was set to false"
        )


class TestFileMapper:
    from clisops.utils.file_utils import FileMapper

    @pytest.mark.skipif(os.path.isdir("/badc") is False, reason="data not available")
    def test_filemapper(self):
        file_paths = [
            "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
            "tas_day_MIROC6_amip_r1i1p1f1_gn_19790101-19881231.nc",
            "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest/"
            "tas_day_MIROC6_amip_r1i1p1f1_gn_19890101-19981231.nc",
        ]
        dset = self.FileMapper(file_paths)
        dm = project_utils.DatasetMapper(dset)

        assert dm.files == [
            "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest"
            "/tas_day_MIROC6_amip_r1i1p1f1_gn_19790101-19881231.nc",
            "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest"
            "/tas_day_MIROC6_amip_r1i1p1f1_gn_19890101-19981231.nc",
        ]
        assert (
            dm.data_path
            == "/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/amip/r1i1p1f1/day/tas/gn/latest"
        )
        assert dm.ds_id == "c3s-cmip6.CMIP.MIROC.MIROC6.amip.r1i1p1f1.day.tas.gn.latest"


@pytest.mark.xfail(reason="outdated")
def test_url_to_file_path(cds_domain):
    from clisops.project_utils import url_to_file_path

    url = (
        f"{cds_domain}/thredds/fileServer/esg_c3s-cmip6/CMIP/E3SM-Project/E3SM-1-1"
        "/historical/r1i1p1f1/Amon/rlus/gr/v20191211/rlus_Amon_E3SM-1-1_historical_r1i1p1f1_gr_200001-200912.nc"
    )
    fpath = url_to_file_path(url)

    assert (
        fpath == "/badc/cmip6/data/CMIP6/CMIP/E3SM-Project/E3SM-1-1"
        "/historical/r1i1p1f1/Amon/rlus/gr/v20191211"
        "/rlus_Amon_E3SM-1-1_historical_r1i1p1f1_gr_200001-200912.nc"
    )
