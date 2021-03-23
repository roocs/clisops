"""Regrid module."""
import warnings
from pathlib import Path
from typing import Tuple, Union

import cf_xarray as cfxr
import numpy as np
import scipy
import xarray as xr
import xesmf as xe
from roocs_utils.exceptions import InvalidParameterValue

# from roocs_utils.xarray_utils.xarray_utils import get_coord_by_type
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_attr

from clisops.utils import dataset_utils


def regrid(ds, Regridder, adaptive_masking_threshold=0.5):
    # if adaptive_masking_threshold>1. or adaptive_masking_thresold<0.:
    #    adaptive_masking_threshold=False
    # ds_out=Regridder(ds,
    #                 adaptive_masking_threshold=adaptive_masking_threshold,
    #                 keep_attrs=True)

    # adaptive-masking will be supported in xesmf from version 0.6+ and make this lines
    #   obsolete
    if (
        Regridder.method in ["conservative", "conservative_normed", "patch"]
        and adaptive_masking_threshold >= 0.0
        and adaptive_masking_threshold < 1.0
    ):
        return adaptive_masking(ds, Regridder, adaptive_masking_threshold)
    else:
        return Regridder(ds, keep_attrs=True)


def adaptive_masking(ds_in, regridder, min_norm_contribution=0.5):
    """Performs regridding incl. renormalization for conservative weights"""
    validi = ds_in.notnull().astype("d")
    valido = regridder(validi, keep_attrs=True)
    tempi0 = ds_in.fillna(0)
    tempo0 = regridder(tempi0)
    # min_norm_contribution factor could prevent values for cells that should be masked.
    # It prevents the renormalization for cells that get less than min_norm_contribution
    #  from source cells. If the factor==0.66 it means that at most one third of the source cells' area
    #  contributing to the target cell is masked. This factor has however to be tweaked manually for each
    #  pair of source and destination grid.
    if min_norm_contribution < 1:
        valido = xr.where(valido < min_norm_contribution, np.nan, valido)
    ds_out = xr.where(valido != 0, tempo0 / valido, np.nan)
    return ds_out


class Weights:
    # ToDo:
    # - Doc-Strings
    # - Think about whether to extend xesmf.Regridder class?
    # - Load weight file from cache or disk
    # - Save weight file to cache or disk
    # - Reformat to other weight-file formats when loading/saving from disk
    def __init__(
        self, grid_in=None, grid_out=None, from_id=None, from_disk=None, method=None
    ):
        """
        Generate weights for grid_in, grid_out for method or
        read weights from cache (from_id) or
        read weights from disk (from_disk, method).
        In the latter case, the weight file format has to be detected and supported,
        to reformat it to xESMF format.
        """
        self.grid_in = grid_in
        self.grid_out = grid_out
        self.method = method
        self.id = None

        if not grid_in and not grid_out:
            if from_id:
                self.id = from_id
                self.Regridder, self.method = self.load_from_cache(self.id)
            elif from_disk and method:
                self.Regridder = self.load_from_disk(from_disk)
                self.method = method
            else:
                raise Exception(
                    "Not all necessary input parameters have been specified."
                )
        else:
            # Might allow datasets as well, then they can either be used to create Grid objects
            #  or they can be passed on to xesmf.Regridder without any further checking.
            if not isinstance(grid_in, Grid) and not isinstance(grid_out, Grid):
                raise Exception(
                    "Input and output grids have to be instances of clisops.core.Grid!"
                )
            self.id = self.generate_id()
            self.Regridder = self.load_from_cache(self.id)
            if not self.Regridder:
                self.Regridder = self.compute()
                # Masking out-of-domain values and missing values
                #  will be obsolete with future xesmf versions (0.6+)
                self.add_matrix_NaNs()

    def compute(self, ignore_degenerate=None):
        """
        Method to generate or load the weights
        If grids have problems of degenerated cells near the poles
        there is the ignore_degenerate option.
        """

        # Is grid periodic in longitude
        periodic = False
        try:
            if self.grid_in.extent == "global":
                periodic = True
        except AttributeError:
            if self.method not in ["conservative", "conservative_normed"]:
                warnings.warn(
                    "The grid extent could not be accessed. "
                    "It will be assumed that the input grid is not periodic in longitude."
                )

        # Call xesmf.Regridder
        return xe.Regridder(
            self.grid_in.ds,
            self.grid_out.ds,
            self.method,
            periodic=periodic,
            ignore_degenerate=ignore_degenerate,
        )

    def _reformat(self, format_from, format_to):
        raise NotImplementedError

    def _detect_format(self, ds):
        raise NotImplementedError

    def generate_id(self):
        # A unique id could maybe consist of:
        #  - method
        #  - hash/checksum of input and output grid (lat, lon, lat_bnds, lon_bnds)
        return

    def save_to_cache(self):
        # Create folder dependent on id
        # Store weight file in xESMF-format
        # Further information will be stored in a json file in the same folder
        # weightfile=regridderHR.to_netcdf(regridder.filename)
        raise NotImplementedError

    def load_from_cache(self, id):
        # Identify folder by id
        # Load weightfile with xesmf.Regridder
        # Load additional info from the json file: setattr(self, key, initial_data[key])
        # self.Regridder=xe.Regridder(method=method, reuse_weights=True, weights=filename)
        return

    def save_to_disk(self, filename=None, format="xESMF"):
        raise NotImplementedError

    def load_from_disk(self, filename=None, format=None):
        # if format == "xESMF":
        # weightfile=regridderHR.to_netcdf(regridder.filename)
        # else:
        # read file, reformat to xESMF sparse matrix and initialize xesmf.Regridder
        raise NotImplementedError

    def add_matrix_NaNs(self):
        """
        Add Nans to matrices, which makes any output cell with a weight from a NaN input cell = NaN
        Will likely be obsolete with future versions of xesmf (0.6+)
        """
        X = self.Regridder.weights
        M = scipy.sparse.csr_matrix(X)
        # indptr: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
        # Creates array with length nrows+1 with information about non-zero values,
        #  with np.diff calculating how many non-zero elements there are in each row
        num_nonzeros = np.diff(M.indptr)
        # Setting rows with only zeros to NaN
        M[num_nonzeros == 0, 0] = np.NaN
        self.Regridder.weights = scipy.sparse.coo_matrix(M)


class Grid:
    def __init__(self, ds=None, grid_id=None, grid_instructor=tuple()):
        "Initialise the Grid object. Supporting only 2D horizontal grids."

        # TODO: Doc-Strings

        # Some of the methods might be useful outside clisops.core.regrid?
        # -> @staticmethod?
        # -> define outside the class?
        # 2nd option preferred

        # All attributes - defaults
        self.type = None
        self.format = None
        self.extent = None
        self.nlat = 0
        self.nlon = 0
        self.ncells = 0
        self.lat = None
        self.lon = None
        self.lat_bnds = None
        self.lon_bnds = None
        self.mask = None
        self.source = None

        # Grid from dataset/dataarray, grid_instructor or grid_id
        if isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray):
            self.ds = ds
            self.format = self.detect_format()
            self.source = "Dataset"
        elif len(grid_instructor) > 0:
            self.grid_from_instructor(grid_instructor)
        elif grid_id:
            self.grid_from_id(grid_id)
        else:
            raise Exception(
                "xarray.Dataset, grid_id or grid_instructor have to be specified as input."
            )

        # Force format CF
        if self.format not in ["CF"]:
            self.ds = self.grid_reformat(grid_format="CF")

        # Detect latitude and longitude coordinates
        self.lat = self.detect_coordinate("latitude")
        self.lon = self.detect_coordinate("longitude")
        self.lat_bnds = self.detect_bounds(self.lat)
        self.lon_bnds = self.detect_bounds(self.lon)

        # Detect type
        if not self.type:
            self.type = self.detect_type()

        # Unstagger the grid if necessary
        self.grid_unstagger()

        # Cut off halo (duplicate grid rows / columns)
        self.grid_remove_halo()

        # Lon/Lat dimension sizes
        self.nlat, self.nlon, self.ncells = self.detect_shape()

        # Compute bounds if not specified and if possible
        if not self.lat_bnds or not self.lon_bnds:
            self.compute_bounds()

        # Extent of the grid (global or regional)
        if not self.extent:
            self.extent = self.detect_extent()

        # Get a permanent mask if there is
        # self.mask = self.detect_mask()

        # Temporary fix for cf_xarray bug that identifies lat_bnds/lon_bnds as latitude/longitude
        #  if lat_bnds and lon_bnds are registered as coordinates of the xarray.Dataset
        if self.lat_bnds and self.lon_bnds:
            if self.lat_bnds in self.ds.coords and self.lon_bnds in self.ds.coords:
                self.ds = self.ds.reset_coords(
                    [self.lat_bnds, self.lon_bnds], drop=False
                )

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        info = (
            "clisops Grid\n"
            + (
                "Lat x Lon:      {} x {}\n".format(self.nlat, self.nlon)
                if self.type != "irregular"
                else ""
            )
            + "Gridcells:      {}\n".format(self.ncells)
            + "Format:         {}\n".format(self.format)
            + "Type:           {}\n".format(self.type)
            + "Extent:         {}\n".format(self.extent)
            + "Source:         {}\n".format(self.source)
            + "Bounds?         {}\n".format(
                self.lat_bnds is not None and self.lon_bnds is not None
            )
            + "Permanent Mask: {}".format(self.mask)
        )
        return info

    def grid_from_id(self, grid_id):
        grid_dict = {
            "0pt25deg": "cmip6_720x1440_scrip.20181001.nc",  # one cell center @ 0.125E,0.125N
            "World_Ocean_Atlas": "cmip6_180x360_scrip.20181001.nc",  # one cell center @ 0.5E,0.5N
            "1deg": "cmip6_180x360_scrip.20181001.nc",  # one cell center @ 0.5E,0.5N
            "2pt5deg": "cmip6_72x144_scrip.20181001.nc",
            "MERRA-2": "cmip6_361x576_scrip.20181001.nc",
            "0pt625x0pt5deg": "cmip6_361x576_scrip.20181001.nc",
            "ERA-Interim": "cmip6_241x480_scrip.20181001.nc",
            "0pt75deg": "cmip6_241x480_scrip.20181001.nc",
            "ERA-40": "cmip6_145x288_scrip.20181001.nc",
            "1pt25deg": "cmip6_145x288_scrip.20181001.nc",
            "ERA5": "cmip6_721x1440_scrip.20181001.nc",
            "0pt25deg_era5": "cmip6_721x1440_scrip.20181001.nc",
            "0pt25deg_era5_lsm": "land_sea_mask_025degree_ERA5.nc",
            "0pt5deg_lsm": "land_sea_mask_05degree.nc4",
            "1deg_lsm": "land_sea_mask_1degree.nc4",
            "2deg_lsm": "land_sea_mask_2degree.nc4",
        }
        try:
            grid = xr.open_dataset(grid_dict[grid_id])
        except KeyError:
            raise KeyError("The grid_id '%s' you specified does not exist!" % grid_id)
        self.ds = grid
        self.source = "Predefined_" + grid_id
        self.type = "regular_lat_lon"
        self.format = self.detect_format()

    def grid_from_instructor(self, grid_instructor):
        if len(grid_instructor) not in [1, 2, 3, 6]:
            raise Exception(
                "The grid_instructor has to be a tuple of length 1, 2, 3 or 6!"
            )
        elif len(grid_instructor) in [1, 2]:
            grid = xe.util.grid_global(grid_instructor[0], grid_instructor[-1])
        elif len(grid_instructor) in [3, 6]:
            grid = xe.util.grid_2d(
                grid_instructor[0],
                grid_instructor[1],
                grid_instructor[2],
                grid_instructor[-3],
                grid_instructor[-2],
                grid_instructor[-1],
            )
        self.ds = grid
        self.source = "xESMF"
        self.type = "regular_lat_lon"
        self.format = "xESMF"

    def grid_store(self, grid_format):
        if self.format != grid_format:
            self.reformat(grid_format)
        # TODO: Use filenamer? Use a hash or date? Output-Folder?
        filename = (
            self.source + "_" + "x".join([str(self.nlat), str(self.nlon)])
            if self.type != "irregular"
            else str(self.ncells) + "_" + self.type + "_" + self.format + ".nc"
        )
        self.ds.to_netcdf(filename)

    def grid_reformat(self, grid_format):
        # TODO: Extend for formats CF, xESMF, ESMF, UGRID, SCRIP
        #      If CF and self.type=="regular_lat_lon":
        #        assure lat/lon are 1D each and bounds are nlat,2 and nlon,2
        #      -> that might have to be executed after the regridding
        #####################
        # Plan: Start with if-else-tree and later switch to a dictionary
        ###################################
        # if self.format=="CF":
        #      if grid_format=="SCRIP":
        #          do sth
        # else:
        #    raise NotImplementedError
        ###################################
        # def SCRIP_to_CF: ...
        # reformat_dict["SCRIP_to_CF"]=SCRIP_to_CF
        # reformat_dict[self.format+"_"+grid_format]()
        ###################################
        SCRIP_vars = [
            "grid_center_lat",
            "grid_center_lon",
            "grid_corner_lat",
            "grid_corner_lon",
            "grid_dims",
            "grid_area",
            "grid_imask",
        ]
        if self.format == "SCRIP":
            if not (
                all([var in SCRIP_vars for var in self.ds.data_vars])
                and all([coord in SCRIP_vars for coord in self.ds.coords])
            ):
                raise Exception(
                    "Converting the grid format from %s to %s is not yet possible for data variables."
                    % (self.format, grid_format)
                )
            if grid_format == "CF":
                lat = self.ds.grid_center_lat.values.reshape(
                    (self.ds.grid_dims.values[1], self.ds.grid_dims.values[0])
                )
                lon = self.ds.grid_center_lon.values.reshape(
                    (self.ds.grid_dims.values[1], self.ds.grid_dims.values[0])
                )

                if all(
                    [
                        np.array_equal(lat[:, i], lat[:, i + 1], equal_nan=True)
                        for i in range(self.ds.grid_dims.values[0] - 1)
                    ]
                ) and all(
                    [
                        np.array_equal(lon[i, :], lon[i + 1, :], equal_nan=True)
                        for i in range(self.ds.grid_dims.values[1] - 1)
                    ]
                ):
                    # regular_lat_lon grid type:
                    # Reshape vertices from (n,2) to (n+1) for lat and lon axis
                    lat = lat[:, 0]
                    lon = lon[0, :]
                    lat_b = self.ds.grid_corner_lat.values.reshape(
                        (
                            self.ds.grid_dims.values[1],
                            self.ds.grid_dims.values[0],
                            self.ds.dims["grid_corners"],
                        )
                    )
                    lon_b = self.ds.grid_corner_lon.values.reshape(
                        (
                            self.ds.grid_dims.values[1],
                            self.ds.grid_dims.values[0],
                            self.ds.dims["grid_corners"],
                        )
                    )
                    lat_bnds = np.zeros(
                        (self.ds.grid_dims.values[1], 2), dtype="double"
                    )
                    lon_bnds = np.zeros(
                        (self.ds.grid_dims.values[0], 2), dtype="double"
                    )
                    lat_bnds[:, 0] = np.min(lat_b[:, 0, :], axis=1)
                    lat_bnds[:, 1] = np.max(lat_b[:, 0, :], axis=1)
                    lon_bnds[:, 0] = np.min(lon_b[0, :, :], axis=1)
                    lon_bnds[:, 1] = np.max(lon_b[0, :, :], axis=1)
                    ds_ref = xr.Dataset(
                        data_vars={},
                        coords={
                            "lat": (["lat"], lat),
                            "lon": (["lon"], lon),
                            "lat_bnds": (["lat", "bnds"], lat_bnds),
                            "lon_bnds": (["lon", "bnds"], lon_bnds),
                        },
                    )
                    # ToDo: Case of other units (rad)
                    # ToDo: Reformat data variables if in ds, apply imask on data variables
                    # ToDo: vertical axis, time axis, ... ?!
                    ds_ref["lat"].attrs = {
                        "bounds": "lat_bnds",
                        "units": "degrees_north",
                        "long_name": "latitude",
                        "standard_name": "latitude",
                        "axis": "Y",
                    }
                    ds_ref["lon"].attrs = {
                        "bounds": "lon_bnds",
                        "units": "degrees_east",
                        "long_name": "longitude",
                        "standard_name": "longitude",
                        "axis": "X",
                    }
                    ds_ref["lat_bnds"].attrs = {
                        "long_name": "latitude_bounds",
                        "units": "degrees_north",
                    }
                    ds_ref["lon_bnds"].attrs = {
                        "long_name": "longitude_bounds",
                        "units": "degrees_east",
                    }
                    self.format = "CF"
                    return ds_ref

                else:
                    raise Exception(
                        "Converting the grid format from %s to %s is yet only possible for regular latitude longitude grids."
                        % (self.format, grid_format)
                    )
            else:
                raise Exception(
                    "Converting the grid format from %s to %s is not yet supported."
                    % (self.format, grid_format)
                )
        elif self.format == "xESMF":
            if grid_format == "CF":
                # ToDo: Check if it is regular_lat_lon, Check dimension sizes
                # Define lat, lon, lat_bnds, lon_bnds
                lat = self.ds.lat[:, 0]
                lon = self.ds.lon[0, :]
                lat_bnds = np.zeros((lat.shape[0], 2), dtype="double")
                lon_bnds = np.zeros((lon.shape[0], 2), dtype="double")
                # From (N+1, M+1) shaped bounds to (N, M, 4) shaped vertices
                lat_vertices = cfxr.vertices_to_bounds(
                    self.ds.lat_b, ("bnds", "lat", "lon")
                ).values
                lon_vertices = cfxr.vertices_to_bounds(
                    self.ds.lon_b, ("bnds", "lat", "lon")
                ).values
                lat_vertices = np.moveaxis(lat_vertices, 0, -1)
                lon_vertices = np.moveaxis(lon_vertices, 0, -1)
                # From (N, M, 4) shaped vertices to (N, 2)  and (M, 2) shaped bounds
                lat_bnds[:, 0] = np.min(lat_vertices[:, 0, :], axis=1)
                lat_bnds[:, 1] = np.max(lat_vertices[:, 0, :], axis=1)
                lon_bnds[:, 0] = np.min(lon_vertices[0, :, :], axis=1)
                lon_bnds[:, 1] = np.max(lon_vertices[0, :, :], axis=1)
                # Create dataset
                ds_ref = xr.Dataset(
                    data_vars={},
                    coords={
                        "lat": (["lat"], lat),
                        "lon": (["lon"], lon),
                        "lat_bnds": (["lat", "bnds"], lat_bnds),
                        "lon_bnds": (["lon", "bnds"], lon_bnds),
                    },
                )

                # ToDo: Case of other units (rad)
                # ToDo: Reformat data variables if in ds, apply imask on data variables
                # ToDo: vertical axis, time axis, ... ?!
                # Add variable attributes to the coordinate variables
                ds_ref["lat"].attrs = {
                    "bounds": "lat_bnds",
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "axis": "Y",
                }
                ds_ref["lon"].attrs = {
                    "bounds": "lon_bnds",
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "axis": "X",
                }
                ds_ref["lat_bnds"].attrs = {
                    "long_name": "latitude_bounds",
                    "units": "degrees_north",
                }
                ds_ref["lon_bnds"].attrs = {
                    "long_name": "longitude_bounds",
                    "units": "degrees_east",
                }
                self.format = "CF"
                return ds_ref
            else:
                raise Exception(
                    "Converting the grid format from %s to %s is yet only possible for regular latitude longitude grids."
                    % (self.format, grid_format)
                )
        else:
            raise Exception(
                "Converting the grid format from %s to %s is not yet supported."
                % (self.format, grid_format)
            )

    def grid_unstagger(self):
        # TODO
        # Plan:
        # Check if it is vectoral and not scalar data (eg. by variable name? No other idea yet.)
        # Unstagger if needed.
        # a) Provide the unstaggered grid (from another dataset with scalar variable) or provide
        #    the other vector component? One of both might be required.
        # b) Rotate the vector in latitudinal / longitudinal direction and interpolate to
        #    cell center of unstaggered grid
        # c) Flux direction seems to be important for the rotation (see cdo mrotuvb), how to infer that?
        # d) Grids staggered in vertical direction, w-component? Is that important at all for
        #    horizontal regridding, maybe only for 3D-irregular grids?
        # All in all a quite impossible task to automatise this process.
        pass

    def grid_remove_halo(self):
        # TODO
        # Plan:
        # Detect duplicated cells and check if they occupy entire rows / columns
        # If single duplicated cells are found, raise Error
        # If duplicated rows/columns are found, remove them with xarray.Dataset.isel()
        pass

    def detect_format(self):
        # TODO: Extend for formats CF, xESMF, ESMF, UGRID, SCRIP
        # Add more conditions (dimension sizes, ...)
        SCRIP_vars = [
            "grid_center_lat",
            "grid_center_lon",
            "grid_corner_lat",
            "grid_corner_lon",
            # "grid_imask", "grid_area"
        ]
        SCRIP_dims = ["grid_corners", "grid_size", "grid_rank"]
        xESMF_vars = [
            "lat",
            "lon",
            "lat_b",
            "lon_b",
            # "mask",
        ]
        xESMF_dims = ["x", "y", "x_b", "y_b"]

        # Test if SCRIP
        if all([var in self.ds for var in SCRIP_vars]) and all(
            [dim in self.ds for dim in SCRIP_dims]
        ):
            return "SCRIP"
        # Test if xESMF
        elif all([var in self.ds.coords for var in xESMF_vars]) and all(
            [dim in self.ds.dims for dim in xESMF_dims]
        ):
            return "xESMF"
        # Test if CF standard_names latitude and longitude can be found
        elif (
            cfxr.accessor._get_with_standard_name(self.ds, "latitude") != []
            and cfxr.accessor._get_with_standard_name(self.ds, "longitude") != []
        ):
            return "CF"
        else:
            raise Exception("The grid format is not supported.")

    def detect_type(self):
        # TODO: Extend for other formats for regular_lat_lon, curvilinear / rotated_pole, irregular
        if self.format == "CF":
            if self.ds[self.lat].ndim == 1 and self.ds[self.lon].ndim == 1:
                lat_1D = self.ds[self.lat].dims[0]
                lon_1D = self.ds[self.lon].dims[0]
                # if lat_1D in ds[var].dims and lon_1D in ds[var].dims:
                if not self.lat_bnds or not self.lon_bnds:
                    if lat_1D == lon_1D:
                        return "irregular"
                    else:
                        return "regular_lat_lon"
                else:
                    if (
                        lat_1D == lon_1D
                        and all(
                            [
                                self.ds[bnds].ndim == 2
                                for bnds in [self.lon_bnds, self.lat_bnds]
                            ]
                        )
                        and all(
                            [
                                self.ds.dims[dim] > 2
                                for dim in [
                                    self.ds[self.lon_bnds].dims[-1],
                                    self.ds[self.lat_bnds].dims[-1],
                                ]
                            ]
                        )
                    ):
                        return "irregular"
                    elif all(
                        [
                            self.ds[bnds].ndim == 2
                            for bnds in [self.lon_bnds, self.lat_bnds]
                        ]
                    ) and all(
                        [
                            self.ds.dims[dim] == 2
                            for dim in [
                                self.ds[self.lon_bnds].dims[-1],
                                self.ds[self.lat_bnds].dims[-1],
                            ]
                        ]
                    ):
                        return "regular_lat_lon"
                    else:
                        raise Exception("The grid type is not supported.")
                # else:
                #    raise Exception("The grid type is not supported.")
            elif self.ds[self.lat].ndim == 2 and self.ds[self.lon].ndim == 2:
                # Test for curvilinear or restructure lat/lon coordinate variables
                # ToDo: Check if regular_lat_lon despite 2D
                #  - requires additional function checking
                #      lat[:,i]==lat[:,j] for all i,j
                #      lon[i,:]==lon[j,:] for all i,j
                #  - and if that is the case to extract lat/lon and *_bnds
                #      lat[:]=lat[:,j], lon[:]=lon[j,:]
                #      lat_bnds[:, 2]=[min(lat_bnds[:,j, :]), max(lat_bnds[:,j, :])]
                #      lon_bnds similar
                if not self.ds[self.lat].shape == self.ds[self.lon].shape:
                    raise Exception("The grid type is not supported.")
                else:
                    if not self.lat_bnds or not self.lon_bnds:
                        return "curvilinear"
                    else:
                        # Shape of curvilinear bounds either [nlat, nlon, 4] or [nlat+1, nlon+1]
                        if list(self.ds[self.lat].shape) + [4] == list(
                            self.ds[self.lat_bnds].shape
                        ) and list(self.ds[self.lon].shape) + [4] == list(
                            self.ds[self.lon_bnds].shape
                        ):
                            return "curvilinear"
                        elif [si + 1 for si in self.ds[self.lat].shape] == list(
                            self.ds[self.lat_bnds].shape
                        ) and [si + 1 for si in self.ds[self.lon].shape] == list(
                            self.ds[self.lon_bnds].shape
                        ):
                            return "curvilinear"
                        else:
                            raise Exception("The grid type is not supported.")
            else:
                raise Exception("The grid type is not supported.")
        else:
            raise Exception(
                "Grid type can only be determined for datasets following the CF conventions."
            )

    def detect_extent(self):
        "Determine if grid is global in terms of its latitudinal/east-west extent."
        # TODO
        # Plan:
        # Check lat/lon_bnds or lat/lon if not available
        # Consider different longitude ranges like 0,360 -180,180
        # -> see subset roll functionality (and issue related to it about the gridspacing)
        # Check approx resolution in x-direction (abs(lon1-lon0))
        # Check eastmost/westmost (np.amax/np.amin)
        # Create histogram to see if the grid has global extent in x-direction
        #
        # Support Units "rad" rather than "degree ..."
        #
        # Additionally check that leftmost and rightmost lon_bnds touch for each row?
        #
        # Decide on the supported longitude range: elif lon_min<-720. or lon_max>720.:
        #
        # Perform a roll if necessary in case the longitude values are not in the range (0,360)
        # - Grids that range for example from (-1. , 359.)
        # - Grids that are totally out of range, like GFDL (-300, 60)
        # ds=dataset_utils.check_lon_alignment(ds, (0,360)) # does not work yet for this purpose

        # Approximate the resolution in x direction
        if self.ds[self.lon].ndim == 2:
            approx_xres = np.amax(
                [
                    np.average(
                        np.absolute(
                            self.ds[self.lon].values[:, 1:]
                            - self.ds[self.lon].values[:, :-1]
                        )
                    ),
                    np.average(
                        np.absolute(
                            self.ds[self.lon].values[1:, :]
                            - self.ds[self.lon].values[:-1, :]
                        )
                    ),
                ]
            )
        elif self.ds[self.lon].ndim == 1:
            approx_xres = np.average(
                np.absolute(
                    self.ds[self.lon].values[1:] - self.ds[self.lon].values[:-1]
                )
            )
        else:
            raise Exception("Only 1D and 2D longitude coordinate variables supported.")

        # Generate a histogram with bins for zonal sections,
        #  width of the bins/sections dependent on the resolution in x-direction
        atol = 2.0 * approx_xres
        # Check the range of the lon values
        lon_max = np.amax(self.ds[self.lon].values)
        lon_min = np.amin(self.ds[self.lon].values)
        if lon_min < -atol and lon_min > -180.0 - atol and lon_max < 180.0 + atol:
            min_range, max_range = (-180.0, 180.0)
        elif lon_min > -atol and lon_max < 360.0 + atol:
            min_range, max_range = (0.0, 360.0)
        elif lon_min < -720.0 or lon_max > 720.0:
            raise Exception(
                "The longitude values have to be within the range (-180, 360)!"
            )
        elif lon_max - lon_min > 360.0 - atol and lon_max - lon_min < 360.0 + atol:
            min_range, max_range = (
                lon_min - approx_xres / 2.0,
                lon_max + approx_xres / 2.0,
            )
        else:
            raise Exception(
                "The longitude values have to be within the range (-180, 360)!"
            )
        # Execute numpy.histogram
        extent_hist = np.histogram(
            self.ds[self.lon],
            bins=np.arange(min_range - approx_xres, max_range + approx_xres, atol),
        )

        # If the counts for all bins are greater than zero, the grid is considered global in x-direction
        # Yet, this information is only needed for xesmf.Regridder, as "periodic in longitude"
        # and hence, the extent in y-direction does not matter.
        # If at some point the qualitative extent in y-direction has to be checked, one needs to
        # take into account that global ocean grids often tend to end at the antarctic coast and do not
        # reach up to -90°S.
        if np.all(extent_hist[0]):
            return "global"
        else:
            return "regional"

    def detect_mask(self):
        "Yet to be implemented, if at all necessary (eg. for reformating to SCRIP etc.)."
        # TODO
        # Plan:
        # Depending on the format, the mask is stored as extra variable.
        # If self.format=="CF": An extra variable mask could be generated from missing values?
        # This could be an extra function of the reformatter with target format xESMF/SCRIP/...
        # For CF as target format, this mask could be applied to mask the data for all variables that
        # are not coordinate or auxilliary variables (infer from attributes if possible).
        # If a vertical dimension is present, this should not be done.
        # In general one might be better off with the adaptive masking and this would be
        # just a nice to have thing in case of reformatting and storing the grid on disk.

        # ds_in_LR_mask["mask"]=xr.where(~np.isnan(ds_LR['tos'].isel(time=0)), 1, 0).astype(int)
        return

    def detect_shape(self):
        if self.ds[self.lon].ndim != self.ds[self.lon].ndim:
            raise Exception(
                "The coordinate variables %s and %s have not the same number of dimensions!"
                % (self.lat, self.lon)
            )
        elif self.ds[self.lat].ndim == 2:
            nlat = self.ds[self.lat].shape[0]
            nlon = self.ds[self.lon].shape[1]
            ncells = nlat * nlon
        elif self.ds[self.lat].ndim == 1:
            if (
                self.ds[self.lat].shape == self.ds[self.lon].shape
                and self.type == "irregular"
            ):
                nlat = self.ds[self.lat].shape[0]
                nlon = nlat
                ncells = nlat
            else:
                nlat = self.ds[self.lat].shape[0]
                nlon = self.ds[self.lon].shape[0]
                ncells = nlat * nlon
        else:
            raise Exception(
                "The coordinate variables %s and %s are not 1- or 2-dimensional!"
                % (self.lat, self.lon)
            )
        return (nlat, nlon, ncells)

    def detect_coordinate(self, coordinate):
        """
        Using cf_xarray function. Might as well use a roocs_utils function, like:
        roocs_utils.xarray_utils.get_coord_by_attr(ds, attr, value)
        roocs_utils.xarray_utils.get_coord_type(coord)
        roocs_utils.xarray_utils.xarray_utils.get_coord_by_type(ds, coord_type, ignore_aux_coords=True)
        """
        # coordinates = self.ds.cf[coordinate].name
        coordinates = get_coord_by_attr(self.ds, "standard_name", coordinate).name
        return coordinates
        """
        coordinates = cfxr.accessor._get_with_standard_name(self.ds, coordinate)

        if coordinates == []:
            raise Exception(
                "A %s coordinate cannot be identified in the dataset!" % coordinate
            )
        elif len(coordinates) > 1:
            raise Exception(
                "More than one %s coordinate has been identified in the dataset!"
                % coordinate
            )
        else:
            return coordinates[0]
        """

    def detect_bounds(self, coordinate):
        "The coordinate variable must have a 'bounds' attribute."
        try:
            # return self.ds.cf.get_bounds(coordinate).name
            return self.ds[coordinate].attrs["bounds"]
        except KeyError:
            warnings.warn(
                "The coordinate variable '%s' does not have a 'bounds' attribute."
                % coordinate
            )
            return

    def compute_bounds(self):
        # TODO
        # Plan:
        # + xESMF / cf_xarray functions:
        #   - ds.cf.add_bounds([lon_name, lat_name]))
        #   - functions in ESMPy?
        # + 34d function
        #   https://github.com/SantanderMetGroup/ATLAS/blob/mai-devel/scripts/ATLAS-data/bash-interpolation-scripts/AtlasCDOremappeR_CORDEX/grid_bounds_calc.py
        #
        # Computation may fail, in this case, raise Warning
        # Without bounds, regrid method 'conservative' cannot be used
        #
        if self.format == "CF":
            if self.type == "regular_lat_lon":
                if (
                    np.amin(self.ds[self.lat].values) < -90.0
                    or np.amax(self.ds[self.lat].values) > 90.0
                ):
                    warnings.warn("At least one latitude value exceeds [-90,90].")
                    return
                if self.nlat < 3 or self.nlon < 3:
                    warnings.warn(
                        "The latitude and longitude axes need at least 3 entries"
                        " to be able to calculate the bounds."
                    )
                    return
                # Assuming lat / lon values are strong monotonically decreasing/increasing
                # Latitude / Longitude bounds shaped (nlat, 2) / (nlon, 2)
                lat_bnds = np.zeros((self.ds[self.lat].shape[0], 2), dtype="double")
                lon_bnds = np.zeros((self.ds[self.lon].shape[0], 2), dtype="double")
                # lat_bnds
                lat_bnds[1:, 0] = (
                    self.ds[self.lat].values[:-1] + self.ds[self.lat].values[1:]
                ) / 2.0
                lat_bnds[:-1, 1] = lat_bnds[1:, 0]
                lat_bnds[0, 0] = self.ds[self.lat].values[0] - lat_bnds[1, 0]
                lat_bnds[-1, 1] = (
                    self.ds[self.lat].values[-1]
                    - lat_bnds[-1, 0]
                    + self.ds[self.lat].values[-1]
                )
                lat_bnds = np.where(lat_bnds < -90.0, -90.0, lat_bnds)
                lat_bnds = np.where(lat_bnds > 90.0, 90.0, lat_bnds)
                # lon_bnds
                lon_bnds[1:, 0] = (
                    self.ds[self.lon].values[:-1] + self.ds[self.lon].values[1:]
                ) / 2.0
                lon_bnds[:-1, 1] = lon_bnds[1:, 0]
                lon_bnds[0, 0] = self.ds[self.lon].values[0] - lon_bnds[1, 0]
                lon_bnds[-1, 1] = (
                    self.ds[self.lon].values[-1]
                    - lon_bnds[-1, 0]
                    + self.ds[self.lon].values[-1]
                )
                # Add to the dataset as coordinates and attach variable attributes
                self.ds["lat_bnds"] = ((self.ds[self.lat].dims[0], "bnds"), lat_bnds)
                self.ds["lon_bnds"] = ((self.ds[self.lon].dims[0], "bnds"), lon_bnds)
                self.ds = self.ds.set_coords(["lat_bnds", "lon_bnds"])
                self.ds[self.lat].attrs["bounds"] = "lat_bnds"
                self.ds[self.lon].attrs["bounds"] = "lon_bnds"
                self.ds["lat_bnds"].attrs = {
                    "long_name": "latitude_bounds",
                    "units": "degrees_north",
                }
                self.ds["lon_bnds"].attrs = {
                    "long_name": "longitude_bounds",
                    "units": "degrees_east",
                }
                # Set the Class attributes
                self.lat_bnds = "lat_bnds"
                self.lon_bnds = "lon_bnds"
                # Issue warning
                warnings.warn(
                    "Successfully calculated a set of latitude and longitude bounds."
                    " They might, however, differ from the actual bounds"
                    " of the model grid!"
                )
            else:
                warnings.warn(
                    "The bounds can not be calculated for grid_type '%s' and format '%s'."
                    % (self.type, self.format)
                )
        else:
            warnings.warn(
                "The bounds can not be calculated for grid_type '%s' and format '%s'."
                % (self.type, self.format)
            )


"""
#Functions and code from the regrid-prototype notebooks that might
#be useful at some point in setting this up.


def _reravel(vertex_bounds, bounds, M, N):

    Helper function to go from the M+1, N+1 style to
    the vertex style M, N, 4 of lat/lon bounds.

    Basically inverted _unravel as found on
    https://nbviewer.jupyter.org/gist/bradyrx/421627385666eefdb0a20567c2da9976
    Using cf_xarray.vertices_to_bounds instead,
    the following solution yet leaves out gridpoints

    vertex_bounds[:, :, 0] = bounds[0:N, 0:M]

    # fill in missing row
    vertex_bounds[N - 1, :, 1] = bounds[N, 0:M]
    # fill in missing column
    vertex_bounds[:, M - 1, 2] = bounds[0:N, M]
    # fill in remaining element
    vertex_bounds[N - 1, M - 1, 3] = bounds[N, M]
    return vertex_bounds







# Calculate the bounds of the target grid
# The bnds cannot be in CF format, as xESMF conservative regridding requires
#    certain format of the bnds. See eg.:
#    https://github.com/JiaweiZhuang/xESMF/issues/5
#    https://github.com/JiaweiZhuang/xESMF/issues/74
#    https://github.com/JiaweiZhuang/xESMF/issues/14#issuecomment-369686779
#
# lat_bnds with shape (nlat+1)
lat_bnds=np.zeros(ds_out.lat.shape[0]+1, dtype="double")
lat_bnds[0]=-90.
lat_bnds[-1]=90.
lat_bnds[1:-1]=(ds_out.lat.values[:-1]+ds_out.lat.values[1:])/2.

# lon_bnds with shape (nlon+1)
lon_bnds=np.zeros(ds_out.lon.shape[0]+1, dtype="double")
lon_bnds[0]=-180.
lon_bnds[-1]=180.
lon_bnds[1:-1]=(ds_out.lon.values[:-1]+ds_out.lon.values[1:])/2.

# Create dataset with mask
ds_out_mask=xr.Dataset(data_vars={"mask":(["lat", "lon"], xr.where(ds_out['sftlf']==0, 1, 0)),
                                  "lat_b":(["lat1"], lat_bnds),
                                  "lon_b":(["lon1"], lon_bnds)},
                       coords={"lat":(["lat"], ds_out.lat),
                               "lon":(["lon"], ds_out.lon)})

# Create output dataset unmasked
ds_out=xr.Dataset(data_vars={"lat_b":(["lat1"], lat_bnds),
                             "lon_b":(["lon1"], lon_bnds)},
                  coords={"lat":(["lat"], ds_out.lat),
                          "lon":(["lon"], ds_out.lon)})

# Variable attributes
lat_attrs={"bounds":"lat_b",
           "units":"degrees_north",
           "long_name":"latitude",
           "standard_name":"latitude"}
lon_attrs={"bounds":"lon_b",
           "units":"degrees_east",
           "long_name":"longitude",
           "standard_name":"longitude"}

ds_out["lat"].attrs=lat_attrs
ds_out["lon"].attrs=lon_attrs




















def SCRIP_to_xESMF_lat_lon(ds):

    Return xarray.Dataset containing coordinate variables interpretable by xESMF.

    Parameters
    ----------
    ds : xarray.Dataset
        Regular horizontal lat-lon grid in SCRIP format (grid_center_lat/lon
        of shape (nat, nlon), grid_corner_lat/lon of shape (nlat, nlon, 4)).

    Returns
    -------
    xarray.Dataset
        Regular lat-lon grid that can be interpreted by xESMF:
            lat - 1D latitude array of length nlat
            lon - 1D longitude array of length nlon
            lat_b - 1D latitude bounds for lat, array of length nlat+1
            lon_b - 1D longitude bounds for lon, array of length nlon+1

    # SCRIP format seems not to be supported by xESMF (though it is by ESMF)
    # -> converting manually to (almost) CF format for a rectilinear grid following
    #    https://github.com/JiaweiZhuang/xESMF/issues/5
    #    https://github.com/JiaweiZhuang/xESMF/issues/74
    #    https://github.com/JiaweiZhuang/xESMF/issues/14#issuecomment-369686779
    # The bnds cannot be in CF format, as xESMF conservative regridding requires
    #    certain format of the bnds (see links above)
    #
    #  [:,3]     [:,2]
    #
    #    x---------x
    #    |         |
    #    |    o    |
    #    |         |
    #    x---------x
    #
    #  [:,0]     [:,1]
    #
    # x - grid cell corners
    # o - grid cell center
    #
    # lat/lon
    lat=ds["grid_center_lat"].values.reshape((180,360))[:, 0]
    lon=ds["grid_center_lon"].values.reshape((180,360))[0, :]

    # lower and upper bounds
    latb_l=ds["grid_corner_lat"].values[:, 0].reshape((180,360))[:, 0]
    latb_u=ds["grid_corner_lat"].values[:, 3].reshape((180,360))[:, 0]
    lonb_l=ds["grid_corner_lon"].values[:, 0].reshape((180,360))[0, :]
    lonb_u=ds["grid_corner_lon"].values[:, 1].reshape((180,360))[0, :]

    # reshape from (nlat,2) to (nlat+1)
    lat_bnds=np.zeros(lat.shape[0]+1, dtype="double")
    lat_bnds[:-1]=latb_l[:]
    lat_bnds[-1]=latb_u[-1]

    # reshape from (nlon,2) to (nlon+1)
    lon_bnds=np.zeros(lon.shape[0]+1, dtype="double")
    lon_bnds[:-1]=lonb_l[:]
    lon_bnds[-1]=lonb_u[-1]

    # Create dataset
    ds_out=xr.Dataset(data_vars={"lat_b":(["lat1"], lat_bnds),
                                 "lon_b":(["lon1"], lon_bnds)},
                      coords={"lat":(["lat"], lat),
                              "lon":(["lon"], lon)})
    ds_out["lat"].attrs={"bounds":"lat_b",
                         "units":"degrees_north",
                         "long_name":"latitude",
                         "standard_name":"latitude",
                         "axis":"Y"}
    ds_out["lon"].attrs={"bounds":"lon_b",
                         "units":"degrees_east",
                         "long_name":"longitude",
                         "standard_name":"longitude",
                         "axis":"X"}
    ds_out["lat_b"].attrs={"long_name":"latitude_bounds",
                           "units":"degrees_north"}
    ds_out["lon_b"].attrs={"long_name":"longitude_bounds",
                           "units":"degrees_east"}

    return ds_out





















def translate_cf_reglatlon_for_xESMF(ds,
                                     lat_name="lat",
                                     lon_name="lon",
                                     lat_vert_name="lat_bnds",
                                     lon_vert_name="lon_bnds"):

    Return bounds in xESMF interpretable format (bounds with shape (nlat+1) instead of (nlat,2).

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with CF conformal regular lat/lon Grid incl. bounds.
    lat_name : string, optional
        Name of the latitude coordinate array. The default is "lat".
    lon_name : string, optional
        Name of the longitude coordinate array. The default is "lon".
    lat_vert_name : string, optional
        Name of the latitude bounds coordinate array. The default is "lat_bnds".
    lon_vert_name : string, optional
        Name of the longitude bounds coordinate array. The default is "lon_bnds".

    Returns
    -------
    xarray.Dataset
        containing the coordinate variables lat, lon, lat_bnds, lon_bnds.

    # Specify lat_b(nlat+1) and lon_b(nlat+1) from lat_bnds(nlat,2)
    # and lon_bnds(nlon+2) required by ESMF for conservative regridding.
    # Following:
    # https://github.com/JiaweiZhuang/xESMF/issues/74
    # https://github.com/JiaweiZhuang/xESMF/issues/14#issuecomment-369686779
    # Renaming alone is not sufficient
    #rename_dict={"lat_bnds": "lat_b", "lon_bnds": "lon_b"}
    #ds=ds.rename(rename_dict)

    # Reshape vertices from (n,2) to (n+1) for lat and lon axis
    lat_b=np.zeros(ds[lat_vert_name].shape[0]+1, dtype="double")
    lat_b[:-1]=ds[lat_vert_name][:,0].values
    lat_b[-1]=ds[lat_vert_name][-1,1].values
    lon_b=np.zeros(ds[lon_vert_name].shape[0]+1, dtype="double")
    lon_b[:-1]=ds[lon_vert_name][:,0].values
    lon_b[-1]=ds[lon_vert_name][-1,1].values

    ds_xesmf=xr.Dataset(data_vars={"lat_b":(["y1","x1"], lat_b),
                                   "lon_b":(["y1","x1"], lon_b)},
                        coords={"lat":(["y","x"], ds[lat_name].values),
                                "lon":(["y","x"], ds[lon_name].values)})
    ds_xesmf["lat"].attrs={"bounds":"lat_b",
                           "units":"degrees_north",
                           "long_name":"latitude",
                           "standard_name":"latitude",
                           "axis":"Y"}
    ds_xesmf["lon"].attrs={"bounds":"lon_b",
                           "units":"degrees_east",
                           "long_name":"longitude",
                           "standard_name":"longitude",
                           "axis":"X"}
    ds_xesmf["lat_b"].attrs={"long_name":"latitude_bounds",
                             "units":"degrees_north"}
    ds_xesmf["lon_b"].attrs={"long_name":"longitude_bounds",
                             "units":"degrees_east"}
    return ds_xesmf


















# 1st option
def translate_cf_curvilinear_for_xESMF(ds,
                                       lat_name="latitude",
                                       lon_name="longitude",
                                       lat_vert_name="vertices_latitude",
                                       lon_vert_name="vertices_longitude"):

    Reshapes vertices from (nlat,nlon,4) to (nlat+1,nlon+1).
    Returns xarray.dataset.

    # Calculate bounds for input grid (assumes variables vertices_latitude, vertices_longitude)
    # reshape from (nlat,nlon,4) to (nlat+1,nlon+1)
    lat_bnds=np.zeros(tuple([el+1 for el in list(ds[lat_name].shape)]), dtype="double")
    lat_bnds[:-1, :-1]=ds[lat_vert_name][:,:,3]
    lat_bnds[-1, :-1]=ds[lat_vert_name][-1,:,2]
    lat_bnds[:-1, -1]=ds[lat_vert_name][:,-1,1]
    lat_bnds[-1, -1]=ds[lat_vert_name][-1,-1,0]

    lon_bnds=np.zeros(tuple([el+1 for el in list(ds[lon_name].shape)]), dtype="double")
    lon_bnds[:-1, :-1]=ds[lon_vert_name][:,:,3]
    lon_bnds[-1, :-1]=ds[lon_vert_name][-1,:,2]
    lon_bnds[:-1, -1]=ds[lon_vert_name][:,-1,1]
    lon_bnds[-1, -1]=ds[lon_vert_name][-1,-1,0]

    ds_xesmf=xr.Dataset(data_vars={"lat_b":(["y1","x1"], lat_bnds),
                                   "lon_b":(["y1","x1"], lon_bnds)},
                        coords={"lat":(["y","x"], ds[lat_name].values),
                                "lon":(["y","x"], ds[lon_name].values)})
    return ds_xesmf



# 2nd option
def compress_vertices(ds,
                      lat_name="latitude",
                      lon_name="longitude",
                      lat_bnds_name='vertices_latitude',
                      lon_bnds_name='vertices_longitude'):

    Converts (M, N, 4) (lat/lon/vertex) bounds to
    (M+1, N+1) bounds for xESMF.

    # Calculate corners
    # reshape from (nlat,nlon,4) to (nlat+1,nlon+1)
    #
    # Altered from
    # https://nbviewer.jupyter.org/gist/bradyrx/421627385666eefdb0a20567c2da9976
    #
    M = ds[lat_name].shape[1] # i - x - 1st dimension size
    N = ds[lat_name].shape[0] # j - y - 2nd dimension size

    # create arrays for 2D bounds info
    lat_b = np.zeros((N+1, M+1))
    lon_b = np.zeros((N+1, M+1))

    # unravel nvertices to 2D style
    lat_b = _unravel(lat_b, ds[lat_bnds_name], M, N)
    lon_b = _unravel(lon_b, ds[lon_bnds_name], M, N)

    # get rid of old coordinates
    del ds[lat_bnds_name], ds[lon_bnds_name]
    ds=ds.rename({lat_name:"lat", lon_name:"lon"})
    ds["lat"].attrs["bounds"]=lat_bnds_name
    ds["lon"].attrs["bounds"]=lon_bnds_name
    ds = ds.squeeze()

    # assign new coordinates
    ds.coords['lat_b'] = (('y_b', 'x_b'), lat_b)
    ds.coords['lon_b'] = (('y_b', 'x_b'), lon_b)
    return ds
def _unravel(new_bounds, vertex_bounds, M, N):

    Helper function to go from the vertex style to
    the M+1, N+1 style of lat/lon bounds.

    new_bounds[0:N, 0:M] = vertex_bounds[:, :, 0]

    # fill in missing row
    new_bounds[N, 0:M] = vertex_bounds[N-1, :, 1]
    # fill in missing column
    new_bounds[0:N, M] = vertex_bounds[:, M-1, 2]
    # fill in remaining element
    new_bounds[N, M] = vertex_bounds[N-1, M-1, 3]
    return new_bounds

# In case of problems, activate ESMF verbose mode
import ESMF
ESMF.Manager(debug=True)

def add_matrix_NaNs(regridder):
    Add Nans to matrices, which makes any output cell with a weight from a NaN input cell = NaN
    X = regridder.weights
    M = scipy.sparse.csr_matrix(X)
    # indptr: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    # Creates array with length nrows+1 with information about non-zero values,
    #  with np.diff calculating how many non-zero elements there are in each row
    num_nonzeros = np.diff(M.indptr)
    # Setting rows with only zeros to NaN
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.weights = scipy.sparse.coo_matrix(M)
    return regridder

def adaptive_masking(ds_in, regridder, min_norm_contribution=1):
    Performs regridding incl. renormalization for conservative weights
    validi = ds_in.notnull().astype('d')
    valido = regridder(validi)
    tempi0 = ds_in.fillna(0)
    tempo0 = regridder(tempi0)
    # min_norm_contribution factor could prevent values for cells that should be masked.
    # It prevents the renormalization for cells that get less than min_norm_contribution
    #  from source cells. If the factor==0.66 it means that at most one third of the source cells' area
    #  contributing to the target cell is masked. This factor has however to be tweaked manually for each
    #  pair of source and destination grid.
    if min_norm_contribution<1:
        valido = xr.where(valido < min_norm_contribution, np.nan, valido)
    ds_out = xr.where(valido != 0, tempo0 / valido, np.nan)
    return ds_out



# Function to generate the weights
#   If grids have problems of degenerated cells near the poles there is the ignore_degenerate option
def regrid(ds_in, ds_out, method, periodic, ignore_degenerate=None):
    Convenience function for calculating regridding weights
    return xe.Regridder(ds_in, ds_out, method, periodic=periodic, ignore_degenerate=ignore_degenerate)

"""
