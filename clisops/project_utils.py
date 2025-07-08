"""Project utilities for CLISOPS."""

import glob
import os

import xarray as xr
from loguru import logger

from clisops import CONFIG
from clisops.exceptions import InvalidProject
from clisops.utils.file_utils import FileMapper


class DatasetMapper:  # noqa: E501
    r"""
    Class to map to data path, dataset ID and files from any dataset input.

    Dset must be a string and can be input as:
      - A dataset ID: e.g. "cmip5.output1.INM.inmcm4.rcp45.mon.ocean.Omon.r1i1p1.latest.zostoga".
      - A file path:
        e.g. "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/tas_Amon_HadGEM2-ES_rcp85_r1i1p1_200512-203011.nc".
      - A path to a group of files:
        e.g. "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas/\*.nc".
      - A directory e.g. "/badc/cmip5/data/cmip5/output1/MOHC/HadGEM2-ES/rcp85/mon/atmos/Amon/r1i1p1/latest/tas".
      - An instance of the FileMapper class (that represents a set of files within a single directory).

    When force=True, if the project cannot be identified, any attempt to use the base_dir of a project
    to resolve the data path will be ignored. Any of data_path, ds_id, and files that can be set will be set.

    Parameters
    ----------
    dset : str or FileMapper
        The dataset input, which can be a Dataset/DataArray, a string representing a dataset ID or file path,
        or an instance of FileMapper.
    project : str, optional
        The project name to use for mapping the dataset. If not provided, it will be deduced from the dataset input.
    force : bool, optional
        If True, the function will attempt to find files even if the project of the input dataset cannot be identified.
        Default is False.
    """

    SUPPORTED_EXTENSIONS = (".nc", ".gz")

    def __init__(self, dset, project=None, force=False):
        self._project = project
        self.dset = dset

        self._base_dir = None
        self._ds_id = None
        self._data_path = None
        self._files = []

        self._parse(force)

    @staticmethod
    def _get_base_dirs_dict():
        projects = get_projects()
        base_dirs = {project: CONFIG[f"project:{project}"]["base_dir"] for project in projects}
        return base_dirs

    @staticmethod
    def _is_ds_id(dset):
        return dset.count(".") > 1

    def _deduce_project(self, dset):
        if isinstance(dset, str):
            if dset.startswith("/"):
                # by default this returns c3s-cmip6 not cmip6 (as they have the same base_dir)
                base_dirs_dict = self._get_base_dirs_dict()
                for project, base_dir in base_dirs_dict.items():
                    if dset.startswith(base_dir) and CONFIG[f"project:{project}"].get("is_default_for_path") is True:
                        return project

            elif self._is_ds_id(dset):
                return dset.split(".")[0].lower()

            # this will not return c3s project names
            elif dset.endswith(".nc") or os.path.isfile(dset):
                dset = xr.open_dataset(dset, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
                return get_project_from_ds(dset)

        else:
            raise InvalidProject(f"The format of {dset} is not known and the project name could not be found.")

    def _parse(self, force):
        # if instance of FileMapper
        if isinstance(self.dset, FileMapper):
            dset = self.dset.dirpath
        else:
            dset = self.dset

        # set project and base_dir
        if not self._project:
            try:
                self._project = self._deduce_project(dset)
                self._base_dir = get_project_base_dir(self._project)
            except InvalidProject:
                logger.info("The project could not be identified")
                if not force:
                    raise InvalidProject("The project could not be identified and force was set to false")

        # get base_dir in the case where project has been supplied
        if not self._base_dir and self._project:
            self._base_dir = get_project_base_dir(self._project)

        # if a file, group of files or directory to files - find files
        if dset.startswith("/") or dset.endswith(".nc"):
            # if instance of FileMapper
            if isinstance(self.dset, FileMapper):
                self._files = self.dset.file_paths
                self._data_path = self.dset.dirpath

            if os.path.splitext(dset)[-1] in self.SUPPORTED_EXTENSIONS:
                if "*" in dset:
                    self._files = sorted(glob.glob(dset))
                else:
                    self._files.append(dset)

                # remove file extension to create data_path
                self._data_path = "/".join(dset.split("/")[:-1])

            # if base_dir identified, insert into data_path
            if self._base_dir:
                self._ds_id = ".".join(self._data_path.replace(self._base_dir, self._project).strip("/").split("/"))

        # test if dataset id
        elif self._is_ds_id(dset):
            self._ds_id = dset

            mappings = CONFIG.get(f"project:{self.project}", {}).get("fixed_path_mappings", {})

            # If the dataset uses a fixed path mapping (from the config file) then use it
            if self._ds_id in mappings:
                data_path = mappings[self._ds_id]
                self._data_path = os.path.join(self._base_dir, data_path)

                # Use pattern of fixed file mapping as glob pattern
                self._files = sorted(glob.glob(self._data_path))

            # Default mapping is done by converting '.' characters to '/' separators in path
            else:
                self._data_path = os.path.join(self._base_dir, "/".join(dset.split(".")[1:]))

        # use to data_path to find files if not set already
        if len(self._files) < 1:
            self._files = sorted(glob.glob(os.path.join(self._data_path, "*.nc")))
            print(self._data_path)
            print(self._files)

    @property
    def raw(self):
        """
        Raw dataset input.

        Returns
        -------
        xarray.Dataset or xarray.DataArray or str or FileMapper
            The original dataset input provided to the DatasetMapper.
        """
        return self.dset

    @property
    def data_path(self):
        """
        Dataset input converted to a data path.

        Returns
        -------
        str
            The data path derived from the input dataset, which can be a file path or a directory path.
        """
        return self._data_path

    @property
    def ds_id(self):
        """
        Dataset input converted to a ds id.

        Returns
        -------
        str
            The dataset ID derived from the input dataset, which is typically in the format "project.dataset_id".
        """
        return self._ds_id

    @property
    def base_dir(self):
        """
        The base directory of the input dataset.

        Returns
        -------
        str
            The base directory where the dataset files are located, derived from the project configuration.
        """
        return self._base_dir

    @property
    def files(self):
        """
        The files found from the input dataset.

        Returns
        -------
        list
            A list of file paths deduced from the input dataset. If the dataset is a directory or a file pattern,
            it will return all matching files. If the dataset is a dataset ID, it will return files based on the
            dataset ID mapping.
        """
        return self._files

    @property
    def project(self):
        """
        The project of the dataset input.

        Returns
        -------
        str
            The project name derived from the input dataset. If the project cannot be identified, it will return None.
        """
        return self._project


def derive_dset(dset: xr.Dataset | xr.DataArray | str | FileMapper) -> str:
    """
    Derive the dataset path of the provided dset.

    Parameters
    ----------
    dset : xarray.Dataset or xarray.DataArray or str or FileMapper
        The dataset input, which can be a Dataset/DataArray, a string representing a dataset ID or file path,
        or an instance of FileMapper.

    Returns
    -------
    str
        The dataset path derived from the input dataset.
    """
    return DatasetMapper(dset).data_path


def derive_ds_id(dset: xr.Dataset | xr.DataArray | str | FileMapper) -> str:
    """
    Derive the dataset id of the provided dset.

    Parameters
    ----------
    dset : xarray.Dataset or xarray.DataArray or str or FileMapper
        The dataset input, which can be a Dataset/DataArray, a string representing a dataset ID or file path,
        or an instance of FileMapper.

    Returns
    -------
    str
        The dataset id derived from the input dataset.
    """
    return DatasetMapper(dset).ds_id


def datapath_to_dsid(datapath: xr.Dataset | xr.DataArray | str | FileMapper) -> str:
    """
    Switch from dataset path to ds id.

    Parameters
    ----------
    datapath : xarray.Dataset or xarray.DataArray or str or FileMapper
        The dataset input, which can be a Dataset/DataArray, a string representing a dataset ID or file path,
        or an instance of FileMapper.

    Returns
    -------
    str
        The dataset id derived from the input dataset path.
    """
    return DatasetMapper(datapath).ds_id


def dsid_to_datapath(dsid: str) -> str:
    """
    Switch from ds id to dataset path.

    Parameters
    ----------
    dsid : str
        The dataset ID, which should be in the format "project.dataset_id".

    Returns
    -------
    str
        The dataset path derived from the input dataset ID.
    """
    return DatasetMapper(dsid).data_path


def dset_to_filepaths(dset, force=False):
    """
    Get filepaths deduced from input dset.

    Parameters
    ----------
    dset : xarray.Dataset or xarray.DataArray or str or FileMapper
        The dataset input, which can be a Dataset/DataArray, a string representing a dataset ID or file path,
        or an instance of FileMapper.
    force : bool, optional
        If True, the function will attempt to find files even if the project of the input dset cannot be identified.
        Default is False.

    Returns
    -------
    list
        A list of file paths deduced from the input dataset.
    """
    mapper = DatasetMapper(dset, force=force)
    return mapper.files


def switch_dset(dset: xr.Dataset | xr.DataArray | str | FileMapper) -> str:
    """
    Switch between dataset path and ds id.

    Parameters
    ----------
    dset : xarray.Dataset or xarray.DataArray or str or FileMapper
        The dataset input, which can be a Dataset/DataArray, a string representing a dataset ID or file path,
        or an instance of FileMapper.

    Returns
    -------
    str
        The dataset path or dataset ID derived from the input dataset, switched from the input.
    """
    if dset.startswith("/"):
        return datapath_to_dsid(dset)
    else:
        return dsid_to_datapath(dset)


def get_projects() -> list[str]:
    """
    Get all the projects available in the config.

    Returns
    -------
    list of str
        A list of project names derived from the configuration.
    """
    return [_.split(":")[1] for _ in CONFIG.keys() if _.startswith("project:")]


def get_project_from_ds(ds: xr.Dataset | xr.DataArray) -> str | None:
    """
    Get the project from an xarray Dataset/DataArray.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The xarray Dataset or DataArray from which to derive the project.

    Returns
    -------
    str | None
        The project derived from the input dataset.
    """
    for project in get_projects():
        key = map_facet("project", project)
        if ds.attrs.get(key, "").lower() == project:
            return project


def get_project_name(dset: xr.Dataset | xr.DataArray | str | FileMapper) -> str | None:
    """
    Get the project from an input dset.

    Parameters
    ----------
    dset : xarray.Dataset or xarray.DataArray or str or FileMapper
        The dataset input, which can be a Dataset/DataArray, a string representing a dataset ID or file path,
        or an instance of FileMapper.

    Returns
    -------
    str | None
        The project name derived from the input dataset, or None if the project cannot be identified.
    """
    if type(dset) in (xr.core.dataarray.DataArray, xr.core.dataset.Dataset):
        return get_project_from_ds(dset)  # will not return c3s dataset

    else:
        return DatasetMapper(dset).project


def map_facet(facet: str, project: str) -> str:
    """
    Return mapped facet value from config or facet name if not found.

    Parameters
    ----------
    facet : str
        The facet name to map.
    project : str
        The project name to use for mapping.

    Returns
    -------
    str
        The mapped facet value or the original facet name if no mapping is found.
    """
    # Return mapped value or the same facet name
    proj_mappings = CONFIG[f"project:{project}"]["mappings"]
    return proj_mappings.get(facet, facet)


def get_facet(facet_name: str, facets: dict, project: str) -> str:
    """
    Get facet from project config.

    Parameters
    ----------
    facet_name : str
        The name of the facet to retrieve.
    facets : dict
        A dictionary of facets from the project configuration.
    project : str
        The project name to use for mapping the facet.

    Returns
    -------
    str
        The mapped facet value from the project configuration.
    """
    return facets[map_facet(facet_name, project)]


def get_project_base_dir(project: str) -> str:
    """
    Get the base directory of a project from the config.

    Parameters
    ----------
    project : str
        The name of the project for which to retrieve the base directory.

    Returns
    -------
    str
        The base directory of the specified project.
    """
    try:
        return CONFIG[f"project:{project}"]["base_dir"]
    except KeyError:
        raise InvalidProject("The project supplied is not known.")


def get_data_node_dirs_dict() -> dict[str, str]:
    """
    Get a dictionary of the data node roots used for retrieving original files.

    Returns
    -------
    dict
        A dictionary where keys are project names and values are the data node root directories.
    """
    projects = get_projects()
    data_node_dirs = {
        project: CONFIG[f"project:{project}"].get("data_node_root")
        for project in projects
        if CONFIG[f"project:{project}"].get("data_node_root")
    }
    return data_node_dirs


def get_project_from_data_node_root(url: str) -> str:
    """
    Identify the project from data node root by identifying the data node root in the input url.

    Parameters
    ----------
    url : str
        The URL of the original file, which should contain the data node root.

    Returns
    -------
    str
        The project name derived from the data node root in the input URL.

    Raises
    ------
    InvalidProject
        If the project cannot be identified from the URL.
    """
    data_node_dict = get_data_node_dirs_dict()
    project = None

    for proj, data_node_root in data_node_dict.items():
        if data_node_root in url:
            project = proj

    if not project:
        raise InvalidProject(
            f"The project could not be identified from the URL {url} so it could not be mapped to a file path."
        )
    return project


def url_to_file_path(url: str) -> str:
    """
    Convert the input url of an original file to a file path.

    Parameters
    ----------
    url : str
        The URL of the original file, which should contain the data node root.

    Returns
    -------
    str
        The file path derived from the input URL, based on the project's base directory and data node root.
    """
    project = get_project_from_data_node_root(url)

    data_node_root = CONFIG.get(f"project:{project}", {}).get("data_node_root")
    base_dir = CONFIG.get(f"project:{project}", {}).get("base_dir")
    file_path = os.path.join(base_dir, url.partition(data_node_root)[2])

    return file_path
