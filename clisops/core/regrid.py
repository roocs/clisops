"""Regrid module."""
# from pathlib import Path
# from typing import Tuple, Union
import functools
import json
import os
import warnings
from hashlib import md5

import cf_xarray as cfxr
import numpy as np
import roocs_grids
import xarray as xr
from pkg_resources import parse_version

# Try importing xesmf and set to None if not found at correct version
# If set to None, the `require_xesmf` decorator will check this
XESMF_MINIMUM_VERSION = "0.6.0"
try:
    import xesmf as xe

    if parse_version(xe.__version__) < parse_version(XESMF_MINIMUM_VERSION):
        raise ValueError()
except Exception:
    xe = None

# from roocs_utils.exceptions import InvalidParameterValue
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_attr, get_coord_by_type

# from daops import functions to load and save weights from/to a central weight store
try:
    from daops.regrid import weights_load, weights_save
except Exception:
    # Use local weight store
    weights_save = None
    weights_load = None

# from clisops.utils import dataset_utils
from clisops import CONFIG

weights_dir = CONFIG["clisops:grid_weights"]["local_weights_dir"]
weights_svc = CONFIG["clisops:grid_weights"]["remote_weights_svc"]
coord_precision_hor = int(CONFIG["clisops:coordinate_precision"]["hor_coord_decimals"])


def require_xesmf(func):
    "Decorator to ensure that xesmf is installed before function/method is called."

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        if xe is None:
            raise Exception(
                f"Package xesmf >= {XESMF_MINIMUM_VERSION} is required to use the regridding functionality."
            )
        return func(*args, **kwargs)

    return wrapper_func


def check_dir(func, dr):
    "Decorator to ensure that a directory exists."
    if not os.path.isdir(dr):
        os.makedirs(dr)

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_func


check_weights_dir = functools.partial(check_dir, dr=weights_dir)

# Initialize weights_dic (local weights storage)
try:
    with open(weights_dir + "/weights.json") as weights_dic_path:
        weights_dic = json.load(weights_dic_path)
except FileNotFoundError:
    weights_dic = {}


@require_xesmf
def regrid(grid_in, grid_out, weights, adaptive_masking_threshold=0.5):
    # if adaptive_masking_threshold>1. or adaptive_masking_thresold<0.:
    #    adaptive_masking_threshold=False
    # ds_out=Regridder(ds,
    #                 adaptive_masking_threshold=adaptive_masking_threshold,
    #                 keep_attrs=True)
    if not isinstance(grid_out.ds, xr.Dataset):
        raise Exception(
            "The target Grid object 'grid_out' has to be built from an xarray.Dataset"
            " and not an xarray.DataArray!"
        )
    grid_out._drop_vars()  # Remove all unnecessary coords and data_vars
    if isinstance(grid_in.ds, xr.Dataset):
        grid_out._transfer_coords(grid_in)
    regridded_ds = grid_out.ds

    # Copy attrs
    regridded_ds.attrs.update(grid_in.ds.attrs)  # todo: or overwrite them completely
    regridded_ds.attrs.update(
        {
            "regrid_operation": weights.regridder.filename.split(".")[0],
            "regrid_tool": weights.tool,
            "regrid_weights_uid": weights.id,
        }
    )

    # It might in general be sufficient to always act as if the threshold was
    #  set correctly and let xesmf handle it. But then we might not allow it
    #  the bilinear method, as the results do not look too great and I am still
    #  not sure/convinced adaptive_masking makes sense for this method.
    if isinstance(grid_in.ds, xr.Dataset):
        for data_var in grid_in.ds.data_vars:
            if (
                weights.regridder.method
                in ["conservative", "conservative_normed", "patch"]
                and adaptive_masking_threshold >= 0.0
                and adaptive_masking_threshold < 1.0
            ):
                regridded_ds[data_var] = weights.regridder(
                    grid_in.ds[data_var],
                    skipna=True,
                    na_thres=adaptive_masking_threshold,
                    keep_attrs=True,
                )
            else:
                regridded_ds[data_var] = weights.regridder(
                    grid_in.ds[data_var], skipna=False, keep_attrs=True
                )
        return regridded_ds
    else:
        if (
            weights.regridder.method in ["conservative", "conservative_normed", "patch"]
            and adaptive_masking_threshold >= 0.0
            and adaptive_masking_threshold < 1.0
        ):
            regridded_ds[grid_in.ds.name] = weights.regridder(
                grid_in.ds,
                skipna=True,
                na_thres=adaptive_masking_threshold,
                keep_attrs=True,
            )
        else:
            regridded_ds[grid_in.ds.name] = weights.regridder(
                grid_in.ds, skipna=False, keep_attrs=True
            )
        return regridded_ds[grid_in.ds.name]


class Weights:
    # todo:
    # - Doc-Strings
    # - Think about whether to extend xesmf.Regridder class?
    # - Load weight file from cache or disk
    # - Save weight file to cache or disk
    # - Reformat to other weight-file formats when loading/saving from disk

    @require_xesmf
    def __init__(
        self, grid_in, grid_out, from_disk=None, method="nearest_s2d", format="xESMF"
    ):
        """
        Generate weights / read from cache (if present locally or retreivable from central
        regrid weight store) for grid_in, grid_out for method or alternatively
        read weights from disk (from_disk, method).
        In the latter case, the weight file format has to be detected and supported,
        to reformat it to xESMF format.
        """
        if not isinstance(grid_in, Grid) or not isinstance(grid_out, Grid):
            raise Exception(
                "Input and output grids have to be instances of clisops.core.Grid!"
            )
        self.grid_in = grid_in
        self.grid_out = grid_out
        if grid_in.hash == grid_out.hash:
            raise Exception(
                "The selected source and target grids are the same. "
                "No regridding operation required."
            )
        self.method = method
        # todo: properly test / check the periodic attribute of the xESMF Regridder.
        #  The grid.extent check done here might not be suitable to set the periodic att.
        # Is grid periodic in longitude
        self.periodic = False
        try:
            if self.grid_in.extent == "global":
                self.periodic = True
        except AttributeError:
            # forced to False for conservative regridding in xesmf
            if self.method not in ["conservative", "conservative_normed"]:
                warnings.warn(
                    "The grid extent could not be accessed. "
                    "It will be assumed that the input grid is not periodic in longitude."
                )

        # activate ignore degenerate cells setting if collapsing cells are found within the grid
        self.ignore_degenerate = (
            True
            if (
                self.grid_in.contains_collapsing_cells
                or self.grid_out.contains_collapsing_cells
            )
            else None
        )

        self.id = self._generate_id()
        self.filename = self.id + ".nc"

        # todo:
        # - Load weights from disk
        #   + read
        #   + detect format & reformat to xESMF
        #   + generate_id, set ignore_degenerate, periodic, method to unknown if cannot be determined
        #     as id maybe generate hash of the weight matrix
        # - Load weights from cache
        #   + query local store for weights file
        #   +if not file: query central store
        #   +if not file: generate and store locally
        #   +make sure format is xESMF
        #   (+at some point generated weights will be added to central DB)

        if from_disk:
            if not method:
                raise Exception(
                    "'method' has to be specified when reading weights from disk!"
                )
        else:
            self.load_from_cache()
            self.tool = "xESMF_v" + xe.__version__
            self.regridder = self.compute()

        if self.tool.startswith("xESMF"):
            self.format = "xESMF"
            self.save_to_cache()
        else:
            self.format = self._detect_format()
            self._reformat("xESMF")
        self.regridder.filename = self.regridder._get_default_filename()

    def compute(self, ignore_degenerate=None):
        """
        Method to generate or load the weights
        If grids have problems of degenerated cells near the poles
        there is the ignore_degenerate option.
        """
        # Call xesmf.Regridder
        if os.path.isfile(weights_dir + "/" + self.filename):
            reuse_weights = True
        else:
            reuse_weights = False
        return xe.Regridder(
            self.grid_in.ds,
            self.grid_out.ds,
            self.method,
            periodic=self.periodic,
            ignore_degenerate=self.ignore_degenerate,
            unmapped_to_nan=True,
            filename=weights_dir + "/" + self.filename,
            reuse_weights=reuse_weights,
        )
        # The default filename is important for later use, so reset it.
        # xESMF writes weights to disk when filename is given and reuse_weights=False
        # (latter is default) else it will create a default filename and weights can be manually
        # written to disk with Regridder.to_netcdf(filename)

    def _reformat(self, format_from, format_to):
        raise NotImplementedError

    def _detect_format(self, ds):
        raise NotImplementedError

    def _generate_id(self):
        # A unique id could maybe consist of:
        #  - method
        #  - hash/checksum of input and output grid (lat, lon, lat_bnds, lon_bnds)
        wid = "_".join(
            [
                self.grid_in.hash,
                self.grid_out.hash,
                str(self.periodic),
                str(self.ignore_degenerate),
                self.method,
            ]
        )
        return wid

    @check_weights_dir
    def save_to_cache(self):
        # Create folder dependent on id
        # Store weight file in xESMF-format
        # Further information will be stored in a json file in the same folder
        # todo: parallel write mechanism to json file and weights file?
        # todo: store the horizontal input and output grid in an extra netcdf file?
        if self.id not in weights_dic:
            self.regridder.to_netcdf(weights_dir + "/" + self.filename)
            weights_dic.update(
                {
                    self.id: {
                        "source_uid": self.grid_in.hash,
                        "target_uid": self.grid_out.hash,
                        "source_lat": self.grid_in.lat,
                        "source_lon": self.grid_in.lon,
                        "source_lat_bnds": self.grid_in.lat_bnds,
                        "source_lon_bnds": self.grid_in.lon_bnds,
                        "source_nlat": self.grid_in.nlat,
                        "source_nlon": self.grid_in.nlon,
                        "source_ncells": self.grid_in.ncells,
                        "source_type": self.grid_in.type,
                        "source_format": self.grid_in.format,
                        "source_extent": self.grid_in.extent,
                        "target_lat": self.grid_out.lat,
                        "target_lon": self.grid_out.lon,
                        "target_lat_bnds": self.grid_out.lat_bnds,
                        "target_lon_bnds": self.grid_out.lon_bnds,
                        "target_nlat": self.grid_out.nlat,
                        "target_nlon": self.grid_out.nlon,
                        "target_ncells": self.grid_out.ncells,
                        "target_type": self.grid_out.type,
                        "target_format": self.grid_out.format,
                        "target_extent": self.grid_out.extent,
                        "format": self.format,
                        "ignore_degenerate": str(self.ignore_degenerate),
                        "periodic": str(self.periodic),
                        "method": self.method,
                        "uid": self.id,
                        "filename": self.filename,
                        "def_filename": self.regridder.filename,
                        "tool": self.tool,
                    }
                }
            )
            with open(weights_dir + "/weights.json", "w") as weights_dic_path:
                json.dump(weights_dic, weights_dic_path, sort_keys=True, indent=4)

    @check_weights_dir
    def load_from_cache(self):
        # Load additional info from the json file: setattr(self, key, initial_data[key])
        if self.id not in weights_dic:
            if weights_load and weights_svc:
                weights_load(
                    weights_svc, weights_dir, weights_dic, self.id, self.filename
                )
        if self.id in weights_dic:
            self.tool = weights_dic[self.id]["tool"]

    def save_to_disk(self, filename=None, wformat="xESMF"):
        raise NotImplementedError

    def load_from_disk(self, filename=None, format=None):
        # if format == "xESMF":
        # weightfile = regridderHR.to_netcdf(regridder.filename)
        # else:
        # read file, reformat to xESMF sparse matrix and initialize xesmf.Regridder
        raise NotImplementedError


class Grid:
    def __init__(self, ds=None, grid_id=None, grid_instructor=None):
        "Initialise the Grid object. Supporting only 2D horizontal grids."

        # todo: Doc-Strings

        # todo: DataArrays cannot have bounds?! Therefore not allow DataArrays?

        # Some of the methods might be useful outside clisops.core.regrid?
        # -> @staticmethod?
        # -> define outside the class?
        # 2nd option preferred
        grid_instructor = grid_instructor or tuple()

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
        self.hash = None

        # Grid from dataset/dataarray, grid_instructor or grid_id
        if isinstance(ds, (xr.Dataset, xr.DataArray)):
            if grid_id in ["auto", "adaptive"]:
                self.grid_from_ds_adaptive(ds)
            else:
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

        # xESMF will need standard names set on the coordinate objects
        # - set them just in case they don't exist in the input grid
        for coord_type in ("latitude", "longitude"):
            self.set_standard_name(coord_type)

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
            if isinstance(self.ds, xr.Dataset):
                self.compute_bounds()

        # Extent of the grid (global or regional)
        if not self.extent:
            self.extent = self.detect_extent()

        # Get a permanent mask if there is
        # self.mask = self.detect_mask()

        # Clean coordinate variables out of data_vars
        if isinstance(self.ds, xr.Dataset):
            self._set_data_vars_and_coords()

        # todo: possible step to use np.around(in_array, decimals [, out_array])
        # 6 decimals corresponds to precision of ~ 0.1m (deg), 6m (rad)
        self._cap_precision(coord_precision_hor)

        # Create md5 hash of the coordinate variable arrays
        # Takes into account lat/lon + bnds + mask (if defined)
        self.hash = self.compute_hash()

        # Detect collapsing grid cells
        if self.lat_bnds and self.lon_bnds:
            self.detect_collapsed_grid_cells()
        else:
            self.coll_mask = None
            self.contains_collapsing_cells = None

    def __str__(self):
        if self.type == "irregular":
            grid_str = str(self.ncells) + "_cells_grid"
        else:
            grid_str = str(self.nlat) + "x" + str(self.nlon) + "_cells_grid"
        return grid_str

    def __repr__(self):
        info = (
            "clisops {}\n".format(self.__str__())
            + (
                "Lat x Lon:      {} x {}\n".format(self.nlat, self.nlon)
                if self.type != "irregular"
                else ""
            )
            + "Gridcells:        {}\n".format(self.ncells)
            + "Format:           {}\n".format(self.format)
            + "Type:             {}\n".format(self.type)
            + "Extent:           {}\n".format(self.extent)
            + "Source:           {}\n".format(self.source)
            + "Bounds?           {}\n".format(
                self.lat_bnds is not None and self.lon_bnds is not None
            )
            + "Collapsing cells? {}\n".format(self.contains_collapsing_cells)
            + "Permanent Mask:   {}\n".format(self.mask)
            + "md5 hash:         {}".format(self.hash)
        )
        return info

    def grid_from_id(self, grid_id):
        try:
            grid_file = roocs_grids.get_grid_file(grid_id)
            grid = xr.open_dataset(grid_file)
        except KeyError:
            raise KeyError(f"The grid_id '{grid_id}' you specified does not exist!")

        self.ds = grid
        self.source = "Predefined_" + grid_id
        self.type = "regular_lat_lon"
        self.format = self.detect_format()

    @require_xesmf
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

    @require_xesmf
    def grid_from_ds_adaptive(self, ds):
        grid_tmp = Grid(ds=ds)
        if grid_tmp.type == "irregular":
            raise Exception("The grid type is not supported.")
            # One could distribute the number of grid cells to nlat and nlon,
            # in proportion to extent in latitudinal and longitudinal direction
        else:
            # todo: filter out missing values is done by xarray if
            # attribute is set, if not, should the attribute be set
            # or let dachar/daops deal with this?
            xsize = grid_tmp.nlon
            ysize = grid_tmp.nlat
            xfirst = float(grid_tmp.ds[grid_tmp.lon].min())
            yfirst = float(grid_tmp.ds[grid_tmp.lat].min())
            xlast = float(grid_tmp.ds[grid_tmp.lon].max())
            ylast = float(grid_tmp.ds[grid_tmp.lat].max())
            xinc = (xlast - xfirst) / (xsize - 1)
            yinc = (ylast - yfirst) / (ysize - 1)
            xrange = [0.0, 360.0] if xlast > 180 else [-180.0, 180.0]
            xfirst = xfirst - xinc / 2.0
            xlast = xlast + xinc / 2.0
            xfirst = xfirst if xfirst > xrange[0] - xinc / 2.0 else xrange[0]
            xlast = xlast if xlast < xrange[1] + xinc / 2.0 else xrange[1]
            yfirst = yfirst - yinc / 2.0
            ylast = ylast + yinc / 2.0
            yfirst = yfirst if yfirst > -90.0 else -90.0
            ylast = ylast if ylast < 90.0 else 90.0
            self.grid_from_instructor((xfirst, xlast, xinc, yfirst, ylast, yinc))

    def grid_store(self, grid_format):
        if self.format != grid_format:
            self.reformat(grid_format)

        # todo: Use filenamer? Use a hash or date? Output-Folder?
        filename = (
            self.source + "_" + "x".join([str(self.nlat), str(self.nlon)])
            if self.type != "irregular"
            else str(self.ncells) + "_" + self.type + "_" + self.format + ".nc"
        )
        self.ds.to_netcdf(filename)

    def grid_reformat(self, grid_format):
        # todo: Extend for formats CF, xESMF, ESMF, UGRID, SCRIP
        #      If CF and self.type=="regular_lat_lon":
        #        assure lat/lon are 1D each and bounds are nlat,2 and nlon,2
        #      -> that might have to be executed after the regridding
        # todo: When 2D coordinates will be changed to 1D index coordinates
        #       xarray.assign_coords might be necessary, or alternatively,
        #       define a new Dataset and move all data_vars and aux. coords across.
        #       Might introduce drop_vars=True/False to get rid of other than horizonal
        #       coordinate variables if required.
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

        if not isinstance(self.ds, xr.Dataset):
            raise Exception(
                "Reformat is only possible for Datasets."
                " DataArrays have to be CF conformal coordinate variables defined."
            )

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
                ).astype(np.float32)
                lon = self.ds.grid_center_lon.values.reshape(
                    (self.ds.grid_dims.values[1], self.ds.grid_dims.values[0])
                ).astype(np.float32)

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
                    ).astype(np.float32)
                    lon_b = self.ds.grid_corner_lon.values.reshape(
                        (
                            self.ds.grid_dims.values[1],
                            self.ds.grid_dims.values[0],
                            self.ds.dims["grid_corners"],
                        )
                    ).astype(np.float32)
                    lat_bnds = np.zeros(
                        (self.ds.grid_dims.values[1], 2), dtype=np.float32
                    )
                    lon_bnds = np.zeros(
                        (self.ds.grid_dims.values[0], 2), dtype=np.float32
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
                    # todo: Case of other units (rad)
                    # todo: Reformat data variables if in ds, apply imask on data variables
                    # todo: vertical axis, time axis, ... ?!
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
                # todo: Check if it is regular_lat_lon, Check dimension sizes
                # Define lat, lon, lat_bnds, lon_bnds
                lat = self.ds.lat[:, 0]
                lon = self.ds.lon[0, :]
                lat_bnds = np.zeros((lat.shape[0], 2), dtype=np.float32)
                lon_bnds = np.zeros((lon.shape[0], 2), dtype=np.float32)

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
                        "lat": (["lat"], lat.data),
                        "lon": (["lon"], lon.data),
                        "lat_bnds": (["lat", "bnds"], lat_bnds.data),
                        "lon_bnds": (["lon", "bnds"], lon_bnds.data),
                    },
                )

                # todo: Case of other units (rad)
                # todo: Reformat data variables if in ds, apply imask on data variables
                # todo: vertical axis, time axis, ... ?!
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
        # todo
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
        # todo
        # If dimension is not named after the coordinate variable, get dimension name and then isel.
        # Always assuming for 2D coordinate variables, the first dimension is latitude, the second longitude
        # Plan:
        # Detect duplicated cells and check if they occupy entire rows / columns
        # If single duplicated cells are found, raise Error
        # If duplicated rows/columns are found, remove them with xarray.Dataset.isel()

        # This might be moved out of this class to be a general util function,
        # as something similar is required for subsetting and averaging.

        # In this class, as a single Grid object does not have the info whether
        #  it is an input or an output grid, it is not checked whether the extent
        #  of the output grid domain requires the halo to be removed or if partial row/column
        #  halos would prevent the remapping process. For each duplicated grid point one would
        #  have to check if it falls into the domain of the output grid / subset_bbox / area to
        #  average over.

        # Create array of (ilat, ilon) tuples
        if self.ds[self.lon].ndim == 2 or (
            self.ds[self.lon].ndim == 1 and self.type == "irregular"
        ):
            latlon_halo = np.array(
                list(
                    zip(
                        self.ds[self.lat].values.ravel(),
                        self.ds[self.lon].values.ravel(),
                    )
                ),
                dtype=("float32,float32"),
            ).reshape(self.ds[self.lon].values.shape)
        else:
            latlon_halo = list()

        # For 1D regular_lat_lon - find duplicates - remove them assuming dimensions and coordinate variables have the same name
        if isinstance(latlon_halo, list):
            dup_rows = [
                i
                for i in list(range(self.ds[self.lat].shape[0]))
                if i not in np.unique(self.ds[self.lat], return_index=True)[1]
            ]
            dup_cols = [
                i
                for i in list(range(self.ds[self.lon].shape[0]))
                if i not in np.unique(self.ds[self.lon], return_index=True)[1]
            ]
            if dup_cols != []:
                lat_dim = self.ds[self.lat].dims[0]
                self.ds = self.ds.isel(
                    {
                        lat_dim: [
                            i
                            for i in range(0, self.ds[self.lat].shape[0])
                            if i not in dup_cols
                        ]
                    }
                )
                warnings.warn(
                    "The selected dataset contains duplicate grid cells. "
                    "The following %i duplicated columns will be removed: %s"
                    % (len(dup_cols), ", ".join([str(i) for i in dup_cols]))
                )
            if dup_rows != []:
                lon_dim = self.ds[self.lon].dims[0]
                self.ds = self.ds.isel(
                    {
                        lon_dim: [
                            i
                            for i in range(0, self.ds[self.lon].shape[0])
                            if i not in dup_rows
                        ]
                    }
                )
                warnings.warn(
                    "The selected dataset contains duplicate grid cells. "
                    "The following %i duplicated rows will be removed: %s"
                    % (len(dup_rows), ", ".join([str(i) for i in dup_rows]))
                )
            return
        # For 1D irregular - find duplicates - remove them using xarray.Dataset.isel
        elif self.type == "irregular" and self.ds[self.lon].ndim == 1:
            mask_duplicates = self._create_duplicate_mask(latlon_halo)
            dup_cells = np.where(mask_duplicates is True)[0]
            if dup_cells != []:
                # ncells dimension name:
                ncells_dim = self.ds[self.lon].dims[0]
                self.ds = self.ds.isel(
                    {
                        ncells_dim: [
                            i
                            for i in range(0, self.ds[self.lon].shape[0])
                            if i not in dup_cells
                        ]
                    }
                )
                warnings.warn(
                    "The selected dataset contains duplicate grid cells. "
                    "The following %i duplicated cells will be removed: %s"
                    % (len(dup_cells), self._list_ten(dup_cells))
                )
            return
        # For 2D coordinate variables - find duplicates - remove them using xarray.Dataset.isel
        #    ... assuming lat is the first dimension and lon is the second
        #        dimension of the 2D coordinate variables
        else:
            mask_duplicates = self._create_duplicate_mask(latlon_halo)
            # All duplicate rows indices:
            dup_rows = list()
            for i in range(mask_duplicates.shape[0]):
                if all(mask_duplicates[i, :]):
                    dup_rows.append(i)
            # All duplicate columns indices:
            dup_cols = list()
            for j in range(mask_duplicates.shape[1]):
                if all(mask_duplicates[:, j]):
                    dup_cols.append(j)
            for i in dup_rows:
                mask_duplicates[i, :] = False
            for j in dup_cols:
                mask_duplicates[:, j] = False
            # All duplicate rows indices:
            dup_part_rows = list()
            for i in range(mask_duplicates.shape[0]):
                if any(mask_duplicates[i, :]):
                    dup_part_rows.append(i)
            # All duplicate columns indices:
            dup_part_cols = list()
            for j in range(mask_duplicates.shape[1]):
                if any(mask_duplicates[:, j]):
                    dup_part_cols.append(j)
            if 1 == 2 and dup_part_cols != [] or dup_part_rows != [] and 1 == 2:
                raise Exception(
                    "The selected dataset contains dupliated grid cells. "
                    "Several rows or columns of the grid are partially duplicated and thus cannot be removed!"
                )
            else:
                if dup_cols != []:
                    warnings.warn(
                        "The selected dataset contains duplicate grid cells. "
                        "The following %i duplicated columns will be removed: %s"
                        % (len(dup_cols), ", ".join([str(i) for i in dup_cols]))
                    )
                    lon_dim = self.ds[self.lon].dims[1]
                    self.ds = self.ds.isel(
                        {
                            lon_dim: [
                                i
                                for i in range(0, self.ds[self.lon].shape[1])
                                if i not in dup_cols
                            ]
                        }
                    )
                if dup_rows != []:
                    warnings.warn(
                        "The selected dataset contains duplicate grid cells. "
                        "The following %i duplicated rows will be removed: %s"
                        % (len(dup_rows), ", ".join([str(i) for i in dup_rows]))
                    )
                    lat_dim = self.ds[self.lat].dims[0]
                    self.ds = self.ds.isel(
                        {
                            lat_dim: [
                                i
                                for i in range(0, self.ds[self.lat].shape[0])
                                if i not in dup_rows
                            ]
                        }
                    )
            return

    def _create_duplicate_mask(self, arr):
        "Create duplicate mask helper function"
        arr_flat = arr.ravel()
        mask = np.zeros_like(arr_flat, dtype=bool)
        mask[np.unique(arr_flat, return_index=True)[1]] = True
        mask_duplicates = np.where(mask, False, True).reshape(arr.shape)
        return mask_duplicates

    def _list_ten(self, list1d):
        """
        List up to 10 list elements equally distributed to beginning and end of list.
        Helper function.
        """
        if len(list1d) < 11:
            return ", ".join(str(i) for i in list1d)
        else:
            return (
                ", ".join(str(i) for i in list1d[0:5])
                + " ... "
                + ", ".join(str(i) for i in list1d[-5:])
            )

    def detect_format(self):
        # todo: Extend for formats CF, xESMF, ESMF, UGRID, SCRIP
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
            [dim in self.ds.dims for dim in SCRIP_dims]
        ):
            return "SCRIP"

        # Test if xESMF
        elif all([var in self.ds.coords for var in xESMF_vars]) and all(
            [dim in self.ds.dims for dim in xESMF_dims]
        ):
            return "xESMF"

        # Test if CF standard_names latitude and longitude can be found
        elif (
            get_coord_by_type(self.ds, "latitude", ignore_aux_coords=False) is not None
            and get_coord_by_type(self.ds, "longitude", ignore_aux_coords=False)
            is not None
            # cfxr.accessor._get_with_standard_name(self.ds, "latitude") != []
            # and cfxr.accessor._get_with_standard_name(self.ds, "longitude") != []
        ):
            return "CF"

        else:
            raise Exception("The grid format is not supported.")

    def detect_type(self):
        # todo: Extend for other formats for regular_lat_lon, curvilinear / rotated_pole, irregular
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
                # todo: Check if regular_lat_lon despite 2D
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
        # todo
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
            xsize = self.nlon
            ysize = self.nlat
            xfirst = float(self.ds[self.lon].min())
            yfirst = float(self.ds[self.lat].min())
            xlast = float(self.ds[self.lon].max())
            ylast = float(self.ds[self.lat].max())
            xinc = (xlast - xfirst) / (xsize - 1)
            yinc = (ylast - yfirst) / (ysize - 1)
            approx_xres = (xinc + yinc) / 2.0
        elif self.ds[self.lon].ndim == 1:
            if self.type == "irregular":
                raise Exception("The grid type is not supported.")
                # todo: One could distribute the number of grid cells to nlat and nlon,
                # in proportion to extent in latitudinal and longitudinal direction
                # Alternatively one can use the kdtree method to calculate the approx. resolution
                # once it is implemented here.
            else:
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
        lon_max = float(self.ds[self.lon].max())
        lon_min = float(self.ds[self.lon].min())
        if lon_min < -atol and lon_min > -180.0 - atol and lon_max < 180.0 + atol:
            min_range, max_range = (-180.0, 180.0)
        elif lon_min > -atol and lon_max < 360.0 + atol:
            min_range, max_range = (0.0, 360.0)
        # todo: for shifted longitudes, eg. (-300,60)? I forgot what it was for but likely it is irrelevant
        # elif lon_min < -180.0 - atol or lon_max > 360.0 + atol:
        #    raise Exception(
        #        "The longitude values have to be within the range (-180, 360)!"
        #    )
        # elif lon_max - lon_min > 360.0 - atol and lon_max - lon_min < 360.0 + atol:
        #    min_range, max_range = (
        #        lon_min - approx_xres / 2.0,
        #        lon_max + approx_xres / 2.0,
        #    )
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
        # todo
        # Plan:
        # Depending on the format, the mask is stored as extra variable.
        # If self.format=="CF": An extra variable mask could be generated from missing values?
        # This could be an extra function of the reformatter with target format xESMF/SCRIP/...
        # For CF as target format, this mask could be applied to mask the data for all variables that
        # are not coordinate or auxiliary variables (infer from attributes if possible).
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

    def detect_coordinate(self, coord_type):
        """
        Using cf_xarray function. Might as well use a roocs_utils function, like:
        roocs_utils.xarray_utils.get_coord_by_attr(ds, attr, value)
        roocs_utils.xarray_utils.get_coord_type(coord)
        roocs_utils.xarray_utils.xarray_utils.get_coord_by_type(ds, coord_type, ignore_aux_coords=True)
        """
        # coordinates = self.ds.cf[coordinate].name
        coord = get_coord_by_type(self.ds, coord_type, ignore_aux_coords=False)
        try:
            return coord.name
        except AttributeError:
            raise Exception(
                "A %s coordinate cannot be identified in the dataset!" % coord_type
            )

    def set_standard_name(self, coord_type):
        coord = get_coord_by_type(self.ds, coord_type, ignore_aux_coords=False)
        coord.attrs["standard_name"] = coord_type

    def detect_bounds(self, coordinate):
        "The coordinate variable must have a 'bounds' attribute."
        try:
            return self.ds.cf.bounds[coordinate][0]
        except (KeyError, AttributeError):
            warnings.warn(
                "For coordinate variable '%s' no bounds can be identified." % coordinate
            )
            return

    def detect_collapsed_grid_cells(self):
        "Detect collapsing grid cells. Requires defined bounds."
        mask_lat = self._create_collapse_mask(self.ds[self.lat_bnds].data)
        mask_lon = self._create_collapse_mask(self.ds[self.lon_bnds].data)
        # for regular lat-lon grids, create 2D coordinate arrays
        if (
            mask_lat.shape != mask_lon.shape
            and mask_lat.ndim == 1
            and mask_lon.ndim == 1
        ):
            mask_lon, mask_lat = np.meshgrid(mask_lon, mask_lat)
        self.coll_mask = mask_lat | mask_lon
        self.contains_collapsing_cells = np.any(self.coll_mask)

    def _create_collapse_mask(self, arr):
        "Grid cells collapsing to lines or points"
        orig_shape = arr.shape[:-1]  # [nlon, nlat, nbnds] -> [nlon, nlat]
        arr_flat = arr.reshape(-1, arr.shape[-1])  # -> [nlon x nlat, nbnds]
        arr_set = np.apply_along_axis(lambda x: len(set(x)), -1, arr_flat)
        mask = np.zeros(arr_flat.shape[:-1], dtype=bool)
        mask[arr_set == 1] = True
        return mask.reshape(orig_shape)

    def _cap_precision(self, decimals):
        # todo: extend for vertical axis for vertical interpolation usecase
        # 6 decimals corresponds to hor. precision of ~ 0.1m (deg), 6m (rad)
        coord_dict = {}
        attr_dict = {}
        encoding_dict = {}
        for coord in [self.lat_bnds, self.lon_bnds, self.lat, self.lon]:
            if coord:
                attr_dict.update({coord: self.ds[coord].attrs})
                encoding_dict.update({coord: self.ds[coord].encoding})
                coord_dict.update(
                    {
                        coord: (
                            self.ds[coord].dims,
                            np.around(self.ds[coord].data.astype(np.float64), decimals),
                        )
                    }
                )
        if coord_dict:
            self.ds = self.ds.assign_coords(coord_dict)
            # Restore attrs and encoding - is there a proper way to do this?? (todo)
            for coord in [self.lat_bnds, self.lon_bnds, self.lat, self.lon]:
                if coord:
                    self.ds[coord].attrs = attr_dict[coord]
                    self.ds[coord].encoding = encoding_dict[coord]

    def compute_hash(self):
        hash_arr = list()
        for cvar in [self.lat, self.lon, self.lat_bnds, self.lon_bnds, self.mask]:
            if cvar:
                hash_arr.append(
                    md5(str(self.ds[cvar].values.tobytes()).encode("utf-8")).hexdigest()
                )
            # else:
            #    hash_arr.append(md5("undefined".encode('utf-8')).hexdigest())
        print(hash_arr)
        return md5("".join(hash_arr).encode("utf-8")).hexdigest()

    def compare_grid(self, ds_or_Grid):
        if isinstance(ds_or_Grid, xr.Dataset) or isinstance(ds_or_Grid, xr.DataArray):
            grid_tmp = Grid(ds=ds_or_Grid)
        elif isinstance(ds_or_Grid, Grid):
            grid_tmp = ds_or_Grid
        else:
            raise TypeError(
                "The provided input has to be of one of the types [xarray.DataArray, xarray.Dataset, clisops.core.Grid]!"
            )
        return grid_tmp.hash == self.hash

    def _drop_vars(self):
        "Remove all non necessary (=non-horizontal) coords and data_vars"
        to_keep = [
            var for var in [self.lat, self.lon, self.lat_bnds, self.lon_bnds] if var
        ]
        to_drop = [
            var
            for var in list(self.ds.data_vars) + list(self.ds.coords)
            if var not in to_keep
        ]
        self.ds = self.ds.drop(labels=to_drop)

    def _transfer_coords(self, source_grid):
        "Transfer all non-horizontal coordinates to a dataset"
        to_skip = [
            var
            for var in [
                source_grid.lat,
                source_grid.lon,
                source_grid.lat_bnds,
                source_grid.lon_bnds,
            ]
            if var
        ]
        to_transfer = [var for var in list(source_grid.ds.coords) if var not in to_skip]
        coord_dict = {}
        for coord in to_transfer:
            coord_dict.update({coord: source_grid.ds[coord]})
        self.ds = self.ds.assign_coords(coord_dict)

    def _set_data_vars_and_coords(self):
        "Set all non data vars as coordinates."
        to_coord = []
        to_datavar = []

        # Set as coord for auxiliary coord. variables not supposed to be remapped
        if self.ds[self.lat].ndim == 2:
            for var in self.ds.data_vars:
                if self.ds[var].ndim < 2:
                    to_coord.append(var)
                elif self.ds[var].shape[-2:] != self.ds[self.lat].shape:
                    to_coord.append(var)
        elif self.ds[self.lat].ndim == 1:
            for var in self.ds.data_vars:
                if self.type == "irregular":
                    if (
                        len(self.ds[var].shape) > 0
                        and (self.ds[var].shape[-1],) != self.ds[self.lat].shape
                    ):
                        to_coord.append(var)
                else:
                    if not (
                        self.ds[var].shape[-2:] == (self.nlat, self.nlon)
                        or self.ds[var].shape[-2:] == (self.nlon, self.nlat)
                    ):
                        to_coord.append(var)

        # Set bound variables to coord
        for var in [bnd for bnds in self.ds.cf.bounds.values() for bnd in bnds]:
            if var in self.ds.data_vars:
                to_coord.append(var)

        # Reset coords for variables supposed to be remapped (eg. ps)
        for var in self.ds.coords:
            if var not in [self.lat, self.lon] + [
                bnd for bnds in self.ds.cf.bounds.values() for bnd in bnds
            ]:
                if self.type == "irregular":
                    if len(self.ds[var].shape) > 0 and (
                        self.ds[var].shape[-1] == self.ncells
                        and self.ds[var].dims[-1] in self.ds[self.lat].dims
                        and var not in self.ds.dims
                    ):
                        to_datavar.append(var)
                else:
                    if (
                        len(self.ds[var].shape) > 0
                        and (
                            self.ds[var].shape[-2:] == (self.nlat, self.nlon)
                            or self.ds[var].shape[-2:] == (self.nlon, self.nlat)
                        )
                        and all(
                            [
                                dim in self.ds[var].dims
                                for dim in list(self.ds[self.lat].dims)
                                + list(self.ds[self.lon].dims)
                            ]
                        )
                    ):
                        to_datavar.append(var)

        if to_coord:
            self.ds = self.ds.set_coords(list(set(to_coord)))
        if to_datavar:
            self.ds = self.ds.reset_coords(list(set(to_datavar)))

    def compute_bounds(self):
        # todo
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
        # ! Duplicated cells should be removed before computing bounds
        #   or the possiblity of duplicated cells has to be considered
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
                lat_bnds = np.zeros((self.ds[self.lat].shape[0], 2), dtype=np.float32)
                lon_bnds = np.zeros((self.ds[self.lon].shape[0], 2), dtype=np.float32)
                # lat_bnds
                #  positive<0 for strong monotonically increasing
                #  positive>0 for strong monotonically decreasing
                positive = self.ds[self.lat].values[0] - self.ds[self.lat].values[1]
                gspacingl = abs(positive)
                gspacingu = abs(
                    self.ds[self.lat].values[-1] - self.ds[self.lat].values[-2]
                )
                if positive < 0:
                    lat_bnds[1:, 0] = (
                        self.ds[self.lat].values[:-1] + self.ds[self.lat].values[1:]
                    ) / 2.0
                    lat_bnds[:-1, 1] = lat_bnds[1:, 0]
                    lat_bnds[0, 0] = self.ds[self.lat].values[0] - gspacingl / 2.0
                    lat_bnds[-1, 1] = self.ds[self.lat].values[-1] + gspacingu / 2.0
                elif positive > 0:
                    lat_bnds[1:, 1] = (
                        self.ds[self.lat].values[:-1] + self.ds[self.lat].values[1:]
                    ) / 2.0
                    lat_bnds[:-1, 0] = lat_bnds[1:, 1]
                    lat_bnds[0, 1] = self.ds[self.lat].values[0] + gspacingl / 2.0
                    lat_bnds[-1, 0] = self.ds[self.lat].values[-1] - gspacingu / 2.0
                else:
                    warnings.warn(
                        "The bounds could not be calculated since the latitude and/or longitude "
                        "values are not strong monotonically decreasing/increasing."
                    )
                    return
                lat_bnds = np.where(lat_bnds < -90.0, -90.0, lat_bnds)
                lat_bnds = np.where(lat_bnds > 90.0, 90.0, lat_bnds)
                # lon_bnds
                positive = self.ds[self.lon].values[0] - self.ds[self.lon].values[1]
                gspacingl = abs(positive)
                gspacingu = abs(
                    self.ds[self.lon].values[-1] - self.ds[self.lon].values[-2]
                )
                if positive < 0:
                    lon_bnds[1:, 0] = (
                        self.ds[self.lon].values[:-1] + self.ds[self.lon].values[1:]
                    ) / 2.0
                    lon_bnds[:-1, 1] = lon_bnds[1:, 0]
                    lon_bnds[0, 0] = self.ds[self.lon].values[0] - gspacingl / 2.0
                    lon_bnds[-1, 1] = self.ds[self.lon].values[-1] + gspacingu / 2.0
                elif positive > 0:
                    lon_bnds[1:, 1] = (
                        self.ds[self.lon].values[:-1] + self.ds[self.lon].values[1:]
                    ) / 2.0
                    lon_bnds[:-1, 0] = lon_bnds[1:, 1]
                    lon_bnds[0, 1] = self.ds[self.lon].values[0] + gspacingl / 2.0
                    lon_bnds[-1, 0] = self.ds[self.lon].values[-1] - gspacingu / 2.0
                else:
                    warnings.warn(
                        "The bounds could not be calculated since the latitude and/or longitude "
                        "values are not strong monotonically decreasing/increasing."
                    )
                    return
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
                    "The bounds cannot be calculated for grid_type '%s' and format '%s'."
                    % (self.type, self.format)
                )
        else:
            warnings.warn(
                "The bounds cannot be calculated for grid_type '%s' and format '%s'."
                % (self.type, self.format)
            )


"""
#Functions and code from the regrid-prototype notebooks that might
#be useful at some point in setting this up.


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

# In case of problems, activate ESMF verbose mode
import ESMF
ESMF.Manager(debug=True)

"""
