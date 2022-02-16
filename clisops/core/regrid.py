"""Regrid module."""
import functools
import json
import os
import warnings
from collections import ChainMap, OrderedDict
from glob import glob
from hashlib import md5
from math import sqrt
from pathlib import Path
from typing import Optional, Union

import cf_xarray as cfxr
import numpy as np
import roocs_grids
import xarray as xr
from pkg_resources import parse_version
from roocs_utils.exceptions import InvalidParameterValue

# from roocs_utils.exceptions import InvalidParameterValue
from roocs_utils.xarray_utils.xarray_utils import get_coord_by_type

from clisops import CONFIG
from clisops import __version__ as clversion
from clisops.utils.common import check_dir, require_module
from clisops.utils.output_utils import FileLock, create_lock

# from clisops.utils import dataset_utils

# Try importing xesmf and set to None if not found at correct version
# If set to None, the `require_module` decorator will throw an exception
XESMF_MINIMUM_VERSION = "0.6.2"
try:
    import xesmf as xe

    if parse_version(xe.__version__) < parse_version(XESMF_MINIMUM_VERSION):
        raise ValueError()
except Exception:
    xe = None


# Read coordinate variable precision from the clisops configuration (roocs.ini)
#  All horizontal coordinate variables will be rounded to this precision
coord_precision_hor = int(CONFIG["clisops:coordinate_precision"]["hor_coord_decimals"])

# Check if xESMF module is imported - decorator, used below
require_xesmf = functools.partial(
    require_module, module=xe, module_name="xESMF", min_version=XESMF_MINIMUM_VERSION
)


def weights_cache_init(weights_dir: str):
    """
    Initialize global variable `weights_dir` as used by the Weights class.

    Parameters
    ----------
    weights_dir: str
        Directory name to initalize the local weights cache in.
        Will be created if it does not exist.
        Per default, this function is called upon import with weights_dir as defined in roocs.ini.

    Returns
    -------
    None.
    """
    # Overwrite CONFIG entry with new value
    CONFIG["clisops:grid_weights"]["local_weights_dir"] = weights_dir

    # Create directory tree if required
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)


# Initialize weights cache as defined in the clisops configuration (roocs.ini)
weights_cache_init(CONFIG["clisops:grid_weights"]["local_weights_dir"])

# Ensure local weight storage directory exists - decorator, used below
check_weights_dir = functools.partial(
    check_dir, dr=CONFIG["clisops:grid_weights"]["local_weights_dir"]
)


def weights_cache_flush(
    weights_dir_init: Optional[Union[str, Path]] = "",
    dryrun: Optional[bool] = False,
    verbose: Optional[bool] = False,
):
    """
    Flush and reinitialize the local weights cache.

    Parameters
    ----------
    weights_dir_init : str, optional
        Directory name to reinitalize the local weights cache in.
        Will be created if it does not exist.
        The default is CONFIG["clisops:grid_weights"]["local_weights_dir"] as defined in roocs.ini
        (or as redefined by a manual weights_cache_init call).
    dryrun : bool, optional
        If True, it will only print all files that would get deleted. The default is False.
    verbose : bool, optional
        If True, and dryrun is False, will print all files that are getting deleted.
        The default is False.

    Returns
    -------
    None.

    """
    # Read weights_dir from CONFIG
    weights_dir = CONFIG["clisops:grid_weights"]["local_weights_dir"]

    if dryrun:
        print(f"Flushing the clisops weights cache ('{weights_dir}')would remove:")
    elif verbose:
        print(f"Flushing the clisops weights cache ('{weights_dir}'). Removing ...")

    # Find and delete/report weight files, grid files and the json files containing the metadata
    if os.path.isdir(weights_dir):
        flist_weights = glob(f"{weights_dir}/weights_{'?'*32}_{'?'*32}_*.nc")
        flist_meta = glob(f"{weights_dir}/weights_{'?'*32}_{'?'*32}_*.json")
        flist_grids = glob(f"{weights_dir}/grid_{'?'*32}.nc")
        if flist_weights != [] or flist_grids != [] or flist_meta != []:
            for f in flist_meta + flist_weights + flist_grids:
                if dryrun or verbose:
                    print(f" - {f}")
                if not dryrun:
                    os.remove(f)
        else:
            if dryrun or verbose:
                print("No weight or grid files found. Cache empty?")
    elif dryrun:
        print("No weight or grid files found. Cache empty?")

    # Reinitialize local weights cache
    if not dryrun:
        if not weights_dir_init:
            weights_dir_init = weights_dir
        weights_cache_init(weights_dir_init)
        if verbose:
            print(f"Initialized new weights cache at {weights_dir_init}")


class Grid:
    """
    Create a Grid object that is suitable to serve as source or target grid of the Weights class.

    Pre-processes coordinate variables of input dataset (eg. create or read dataset from input,
    reformat, generate bounds, identify duplicated and collapsing cells, determine longitudinal extent).

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray, optional
        Uses horizontal coordinates of an xarray.Dataset or xarray.DataArray to create a Grid object.
        The default is None.
    grid_id : str, optional
        Create the Grid object from a selection of pre-defined grids, eg. "1deg" or "2pt5deg".
        The grids are provided via the roocs_grids package (https://github.com/roocs/roocs-grids).
        A special setting is "adaptive"/"auto", which requires the parameter 'ds' to be specified as well,
        and creates a regular lat-lon grid of the same extent and approximate resolution as the grid
        described by 'ds'. The default is None.
    grid_instructor : tuple, float or int, optional
        Create a regional or global regular lat-lon grid using xESMF utility functions.
        - Global grid: grid_instructor = (lon_step, lat_step) or grid_instructor = step
        - Regional grid: grid_instructor = (lon_start, lon_end, lon_step, lat_start, lat_end, lat_step)
                       or grid_instructor = (start, end, step)
        The default is None.
    compute_bounds : bool, optional
        Compute latitude and longitude bounds if the dataset has none defined.
        The default is False.
    """

    def __init__(
        self,
        ds: Optional[Union[xr.Dataset, xr.DataArray]] = None,
        grid_id: Optional[str] = None,
        grid_instructor: Optional[Union[tuple, float, int]] = None,
        compute_bounds: Optional[bool] = False,
    ):
        """Initialise the Grid object. Supporting only 2D horizontal grids."""
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
        self.coll_mask = None
        self.contains_collapsing_cells = None

        # Create grid_instructor as empty tuple if None
        grid_instructor = grid_instructor or tuple()

        # Grid from dataset/dataarray, grid_instructor or grid_id
        if isinstance(ds, (xr.Dataset, xr.DataArray)):
            if grid_id in ["auto", "adaptive"]:
                self._grid_from_ds_adaptive(ds)
            else:
                self.ds = ds
                self.format = self.detect_format()
                self.source = "Dataset"
        elif grid_instructor:
            self._grid_from_instructor(grid_instructor)
        elif grid_id:
            self._grid_from_id(grid_id)
        else:
            raise InvalidParameterValue(
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

        # Make sure standard_names are set for the coordinates
        self.ds[self.lat].attrs["standard_name"] = "latitude"
        self.ds[self.lon].attrs["standard_name"] = "longitude"

        # Detect type
        if not self.type:
            self.type = self.detect_type()

        # Unstagger the grid if necessary
        self.grid_unstagger()

        # Cut off halo (duplicate grid rows / columns)
        self.grid_remove_halo()

        # Lon/Lat dimension sizes
        self.nlat, self.nlon, self.ncells = self.detect_shape()

        # Extent of the grid (global or regional)
        if not self.extent:
            self.extent = self.detect_extent()

        # Get a permanent mask if there is
        # self.mask = self.detect_mask()

        # Clean coordinate variables out of data_vars
        if isinstance(self.ds, xr.Dataset):
            self._set_data_vars_and_coords()

        # Compute bounds if not specified and if possible
        if (not self.lat_bnds or not self.lon_bnds) and compute_bounds:
            if not isinstance(self.ds, xr.Dataset):
                raise InvalidParameterValue(
                    "Bounds can only be attached to xarray.Datasets, not to xarray.DataArrays."
                )
            else:
                self.compute_bounds()

        # todo: possible step to use np.around(in_array, decimals [, out_array])
        # 6 decimals corresponds to precision of ~ 0.1m (deg), 6m (rad)
        self._cap_precision(coord_precision_hor)

        # Create md5 hash of the coordinate variable arrays
        # Takes into account lat/lon + bnds + mask (if defined)
        self.hash = self.compute_hash()

        # Detect collapsing grid cells
        if self.lat_bnds and self.lon_bnds and self.contains_collapsing_cells is None:
            self.detect_collapsed_grid_cells()

        self.label = self._get_label()

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
                "Lat x Lon:        {} x {}\n".format(self.nlat, self.nlon)
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

    def _get_label(self):
        """Generate a label (str) for the Grid."""
        if self.source.startswith("Predefined_"):
            return ".".join(
                ga
                for ga in roocs_grids.grid_annotations[
                    self.source.replace("Predefined_", "")
                ].split(".")
                if "land-sea mask" not in ga
            )
        else:
            if self.type != "irregular":
                return f"{self.extent} {self.type} {self.nlat}x{self.nlon} ({self.ncells} cells) grid."
            else:
                return f"{self.extent} {self.type} {self.ncells} cells grid."

    def _grid_from_id(self, grid_id):
        """Load pre-defined grid from netCDF file (uses roocs_grids)."""
        # Load predefined grid files from disk
        try:
            grid_file = roocs_grids.get_grid_file(grid_id)
            grid = xr.open_dataset(grid_file)
        except KeyError:
            raise KeyError(f"The grid_id '{grid_id}' you specified does not exist.")

        # Set attributes
        self.ds = grid
        self.source = "Predefined_" + grid_id
        self.type = "regular_lat_lon"
        self.format = self.detect_format()

    @require_xesmf
    def _grid_from_instructor(self, grid_instructor: Union[tuple, float, int]):
        """Process instructions to create regional or global grid (uses xESMF utility functions)."""
        # Create tuple of length 1 if input is either float or int
        if isinstance(grid_instructor, (int, float)):
            grid_instructor = (grid_instructor,)

        # Call xesmf.util functions to create the grid
        if len(grid_instructor) not in [1, 2, 3, 6]:
            raise InvalidParameterValue(
                "The grid_instructor has to be a tuple of length 1, 2, 3 or 6."
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

        # Set attributes
        self.ds = grid
        self.source = "xESMF"
        self.type = "regular_lat_lon"
        self.format = "xESMF"

    @require_xesmf
    def _grid_from_ds_adaptive(self, ds: Union[xr.Dataset, xr.DataArray]):
        """Create Grid of similar extent and resolution of input dataset."""
        # todo: dachar/daops to deal with missing values occuring in the coordinate variables
        #       while no _FillValue/missing_value attribute is set
        #  -> FillValues else might get selected as minimum/maximum lat/lon value
        #     since they are not masked

        # Create temporary Grid object out of input dataset
        grid_tmp = Grid(ds=ds)

        # Determine "edges" of the grid
        xfirst = float(grid_tmp.ds[grid_tmp.lon].min())
        xlast = float(grid_tmp.ds[grid_tmp.lon].max())
        yfirst = float(grid_tmp.ds[grid_tmp.lat].min())
        ylast = float(grid_tmp.ds[grid_tmp.lat].max())

        # fix for regional grids that wrap around the Greenwich meridian
        if grid_tmp.extent == "regional" and (xfirst > 180 or xlast > 180):
            grid_tmp.ds.lon.data = grid_tmp.ds.lon.where(
                grid_tmp.ds.lon <= 180, grid_tmp.ds.lon - 360.0
            )
            xfirst = float(grid_tmp.ds[grid_tmp.lon].min())
            xlast = float(grid_tmp.ds[grid_tmp.lon].max())

        # For irregular grids:
        #    Distribute the number of grid cells to nlat and nlon, in proportion
        #    to extent in latitudinal and longitudinal direction
        if grid_tmp.type == "irregular":
            xsize = int(
                sqrt(abs(xlast - xfirst) / abs(ylast - yfirst) * grid_tmp.ncells)
            )
            ysize = int(
                sqrt(abs(ylast - yfirst) / abs(xlast - xfirst) * grid_tmp.ncells)
            )
        # Else, use nlat and nlon of the dataset
        else:
            xsize = grid_tmp.nlon
            ysize = grid_tmp.nlat

        # Compute latitudinal / longitudinal resolution (=increment)
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

        # Create regular lat-lon grid with these specifics
        self._grid_from_instructor((xfirst, xlast, xinc, yfirst, ylast, yinc))

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
        # todo: removes attrs - should attrs be kept?
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
            raise InvalidParameterValue(
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
            if dup_cells.size > 0:
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
                    "Several rows or columns of the grid are partially duplicated and thus cannot be removed."
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

        # Test if latitude and longitude can be found - standard_names would be set later if undef.
        elif (
            "latitude" in self.ds.cf.standard_names
            and "longitude" in self.ds.cf.standard_names
        ) or (
            get_coord_by_type(self.ds, "latitude", ignore_aux_coords=False) is not None
            and get_coord_by_type(self.ds, "longitude", ignore_aux_coords=False)
            is not None
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
        xfirst = float(self.ds[self.lon].min())
        xlast = float(self.ds[self.lon].max())
        yfirst = float(self.ds[self.lat].min())
        ylast = float(self.ds[self.lat].max())
        # Approximate the resolution in x direction
        if self.ds[self.lon].ndim == 2:
            xsize = self.nlon
            ysize = self.nlat
            xinc = (xlast - xfirst) / (xsize - 1)
            yinc = (ylast - yfirst) / (ysize - 1)
            approx_xres = (xinc + yinc) / 2.0
        elif self.ds[self.lon].ndim == 1:
            if self.type == "irregular":
                # Distribute the number of grid cells to nlat and nlon,
                # in proportion to extent in latitudinal and longitudinal direction
                # todo: Alternatively one can use the kdtree method to calculate the approx. resolution
                # once it is implemented here
                xsize = int(
                    sqrt(abs(xlast - xfirst) / abs(ylast - yfirst) * self.ncells)
                )
                ysize = int(
                    sqrt(abs(ylast - yfirst) / abs(xlast - xfirst) * self.ncells)
                )
                xinc = (xlast - xfirst) / (xsize - 1)
                yinc = (ylast - yfirst) / (ysize - 1)
                approx_xres = (xinc + yinc) / 2.0
                # raise Exception("The grid type is not supported.")
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
                "The longitude values have to be within the range (-180, 360)."
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
                "The coordinate variables %s and %s have not the same number of dimensions."
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
                "The coordinate variables %s and %s are not 1- or 2-dimensional."
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
        coord = self.ds.cf[coord_type]
        # coord = get_coord_by_type(self.ds, coord_type, ignore_aux_coords=False)

        try:
            return coord.name
        except AttributeError:
            raise AttributeError(
                "A %s coordinate cannot be identified in the dataset." % coord_type
            )

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
        self.contains_collapsing_cells = bool(np.any(self.coll_mask))

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
        self.hash_dict = OrderedDict()
        for coord, coord_var in OrderedDict(
            [
                ("lat", self.lat),
                ("lon", self.lon),
                ("lat_bnds", self.lat_bnds),
                ("lon_bnds", self.lon_bnds),
                ("mask", self.mask),
            ]
        ).items():
            if coord_var:
                self.hash_dict[coord] = md5(
                    str(self.ds[coord_var].values.tobytes()).encode("utf-8")
                ).hexdigest()
            else:
                self.hash_dict[coord] = md5(b"undefined").hexdigest()
        return md5("".join(self.hash_dict.values()).encode("utf-8")).hexdigest()

    def compare_grid(self, ds_or_Grid, verbose=False):
        if isinstance(ds_or_Grid, xr.Dataset) or isinstance(ds_or_Grid, xr.DataArray):
            grid_tmp = Grid(ds=ds_or_Grid)
        elif isinstance(ds_or_Grid, Grid):
            grid_tmp = ds_or_Grid
        else:
            raise InvalidParameterValue(
                "The provided input has to be of one of the types [xarray.DataArray, xarray.Dataset, clisops.core.Grid]."
            )
        if verbose:
            diff = [
                coord_var
                for coord_var in self.hash_dict
                if self.hash_dict[coord_var] != grid_tmp.hash_dict[coord_var]
            ]
            if len(diff) > 0:
                print(f"The two grids differ in their respective {', '.join(diff)}.")
            else:
                print("The two grids are considered equal.")
        return grid_tmp.hash == self.hash

    def _drop_vars(self, keep_attrs=False):
        "Remove all non necessary (=non-horizontal) coords and data_vars"
        to_keep = [
            var for var in [self.lat, self.lon, self.lat_bnds, self.lon_bnds] if var
        ]
        to_drop = [
            var
            for var in list(self.ds.data_vars) + list(self.ds.coords)
            if var not in to_keep
        ]
        if not keep_attrs:
            self.ds.attrs = {}
        self.ds = self.ds.drop_vars(names=to_drop)

    def _transfer_coords(self, source_grid, keep_attrs=True):
        "Transfer all non-horizontal coordinates to a dataset"
        # Skip all coords with horizontal dimensions or
        #  coords with no dimensions that are not listed
        #  in the coordinates attribute of the data_vars
        dims_to_skip = set(
            source_grid.ds[source_grid.lat].dims + source_grid.ds[source_grid.lon].dims
        )
        coordinates_attr = []
        for var in source_grid.ds.data_vars:
            cattr = ChainMap(
                source_grid.ds[var].attrs, source_grid.ds[var].encoding
            ).get("coordinates", "")
            if cattr:
                coordinates_attr += cattr.split()
        to_skip = [
            var
            for var in list(source_grid.ds.coords)
            if source_grid.ds[var].ndim == 0 and var not in coordinates_attr
        ]
        to_transfer = [
            var
            for var in list(source_grid.ds.coords)
            if all([dim not in source_grid.ds[var].dims for dim in dims_to_skip])
        ]
        coord_dict = {}
        for coord in to_transfer:
            if coord not in to_skip:
                coord_dict.update({coord: source_grid.ds[coord]})

        if not keep_attrs:
            self.ds.attrs = {}
        elif keep_attrs == "target":
            pass  # attrs will be the original target grid values
        elif keep_attrs:
            self.ds.attrs.update(source_grid.ds.attrs)
        else:
            raise InvalidParameterValue("Illegal value for the parameter 'keep_attrs'.")

        self.ds = self.ds.assign_coords(coord_dict)

    def _set_data_vars_and_coords(self):
        "Set all non data vars as coordinates."
        to_coord = []
        to_datavar = []

        # Collect all (dimensionless) coordinates
        #  dimensionless variables that are not registered in the coordinates attribute
        #   may not be assigned as coords or xarray will keep them around forever
        #   though they are likely useless after remapping (eg. rotated_latitude_longitude)
        coordinates_attr = []
        for var in self.ds.data_vars:
            cattr = ChainMap(self.ds[var].attrs, self.ds[var].encoding).get(
                "coordinates", ""
            )
            if cattr:
                coordinates_attr += cattr.split()
        # Also add cell_measure variables
        cell_measures = list()
        for cmtype, cm in self.ds.cf.cell_measures.items():
            cell_measures += cm

        # Set as coord for auxiliary coord. variables not supposed to be remapped
        if self.ds[self.lat].ndim == 2:
            for var in self.ds.data_vars:
                if var in cell_measures:
                    to_coord.append(var)
                elif self.ds[var].ndim < 2 and (
                    self.ds[var].ndim > 0 or var in coordinates_attr
                ):
                    to_coord.append(var)
                elif self.ds[var].ndim == 0:
                    continue
                elif self.ds[var].shape[-2:] != self.ds[self.lat].shape:
                    to_coord.append(var)
        elif self.ds[self.lat].ndim == 1:
            for var in self.ds.data_vars:
                if var in cell_measures:
                    to_coord.append(var)
                    continue
                elif self.ds[var].ndim == 0 and var in coordinates_attr:
                    to_coord.append(var)
                    continue
                elif self.type == "irregular":
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
                if var in cell_measures:
                    continue
                elif self.ds[var].ndim == 0 and var not in coordinates_attr:
                    to_datavar.append(var)
                elif self.type == "irregular":
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
            if (
                np.amin(self.ds[self.lat].values) < -90.0
                or np.amax(self.ds[self.lat].values) > 90.0
            ):
                warnings.warn("At least one latitude value exceeds [-90,90].")
                return
            if self.ncells < 3:
                warnings.warn(
                    "The latitude and longitude axes need at least 3 entries"
                    " to be able to calculate the bounds."
                )
                return
            if self.type == "curvilinear":
                # The bounds creation for curvilinear grids has been adapted from
                # https://github.com/SantanderMetGroup/ATLAS/blob/mai-devel/scripts/ATLAS-data/ \
                #  bash-interpolation-scripts/AtlasCDOremappeR_CORDEX/grid_bounds_calc.py
                # It is no longer available there.

                # Rearrange lat/lons
                lons_row = self.ds[self.lon].data.flatten()
                lats_row = self.ds[self.lat].data.flatten()

                # Allocate lat/lon corners
                lons_cor = np.zeros(lons_row.size * 4)
                lats_cor = np.zeros(lats_row.size * 4)

                lons_crnr = np.empty(
                    (self.ds[self.lon].shape[0] + 1, self.ds[self.lon].shape[1] + 1)
                )
                lons_crnr[:] = np.nan
                lats_crnr = np.empty(
                    (self.ds[self.lat].shape[0] + 1, self.ds[self.lat].shape[1] + 1)
                )
                lats_crnr[:] = np.nan

                # -------- Calculating corners --------- #

                # Loop through all grid points except at the boundaries
                for lat in range(1, self.ds[self.lon].shape[0]):
                    for lon in range(1, self.ds[self.lon].shape[1]):
                        # SW corner for each lat/lon index is calculated
                        lons_crnr[lat, lon] = (
                            self.ds[self.lon][lat - 1, lon - 1]
                            + self.ds[self.lon][lat, lon - 1]
                            + self.ds[self.lon][lat - 1, lon]
                            + self.ds[self.lon][lat, lon]
                        ) / 4.0
                        lats_crnr[lat, lon] = (
                            self.ds[self.lat][lat - 1, lon - 1]
                            + self.ds[self.lat][lat, lon - 1]
                            + self.ds[self.lat][lat - 1, lon]
                            + self.ds[self.lat][lat, lon]
                        ) / 4.0

                # Grid points at boundaries
                lons_crnr[0, :] = lons_crnr[1, :] - (lons_crnr[2, :] - lons_crnr[1, :])
                lons_crnr[-1, :] = lons_crnr[-2, :] + (
                    lons_crnr[-2, :] - lons_crnr[-3, :]
                )
                lons_crnr[:, 0] = lons_crnr[:, 1] + (lons_crnr[:, 1] - lons_crnr[:, 2])
                lons_crnr[:, -1] = lons_crnr[:, -2] + (
                    lons_crnr[:, -2] - lons_crnr[:, -3]
                )

                lats_crnr[0, :] = lats_crnr[1, :] - (lats_crnr[2, :] - lats_crnr[1, :])
                lats_crnr[-1, :] = lats_crnr[-2, :] + (
                    lats_crnr[-2, :] - lats_crnr[-3, :]
                )
                lats_crnr[:, 0] = lats_crnr[:, 1] - (lats_crnr[:, 1] - lats_crnr[:, 2])
                lats_crnr[:, -1] = lats_crnr[:, -2] + (
                    lats_crnr[:, -2] - lats_crnr[:, -3]
                )

                # ------------ DONE ------------- #

                # Fill in counterclockwise and rearrange
                count = 0
                for lat in range(self.ds[self.lon].shape[0]):
                    for lon in range(self.ds[self.lon].shape[1]):

                        lons_cor[count] = lons_crnr[lat, lon]
                        lons_cor[count + 1] = lons_crnr[lat, lon + 1]
                        lons_cor[count + 2] = lons_crnr[lat + 1, lon + 1]
                        lons_cor[count + 3] = lons_crnr[lat + 1, lon]

                        lats_cor[count] = lats_crnr[lat, lon]
                        lats_cor[count + 1] = lats_crnr[lat, lon + 1]
                        lats_cor[count + 2] = lats_crnr[lat + 1, lon + 1]
                        lats_cor[count + 3] = lats_crnr[lat + 1, lon]

                        count += 4

                        lon_bnds = lons_cor.reshape(self.nlat, self.nlon, 4)
                        lat_bnds = lats_cor.reshape(self.nlat, self.nlon, 4)

                # Add to the dataset
                self.ds["lat_bnds"] = (
                    (self.ds[self.lat].dims[0], self.ds[self.lat].dims[1], "vertices"),
                    lat_bnds,
                )
                self.ds["lon_bnds"] = (
                    (self.ds[self.lon].dims[0], self.ds[self.lon].dims[1], "vertices"),
                    lon_bnds,
                )

            elif self.type == "regular_lat_lon":
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
                # Add to the dataset
                self.ds["lat_bnds"] = ((self.ds[self.lat].dims[0], "bnds"), lat_bnds)
                self.ds["lon_bnds"] = ((self.ds[self.lon].dims[0], "bnds"), lon_bnds)
            else:
                warnings.warn(
                    "The bounds cannot be calculated for grid_type '%s' and format '%s'."
                    % (self.type, self.format)
                )
                return

            # Add common set of attributes and set as coordinates
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
                " of the model grid."
            )
        else:
            warnings.warn(
                "The bounds cannot be calculated for grid_type '%s' and format '%s'."
                % (self.type, self.format)
            )
            return

    def to_netcdf(
        self,
        folder: Optional[Union[str, Path]] = "./",
        filename: Optional[str] = "",
        grid_format: Optional[str] = "CF",
    ):
        """
        Store a copy of the horizontal Grid as netCDF file on disk.

        Define output folder, filename and output format (currently only 'CF' is supported).

        Parameters
        ----------
        folder : str or Path, optional
            Output folder. The default is the current working directory "./".
        filename : str, optional
            Output filename, to be defined separately from folder. The default is 'grid_<grid.id>.nc'.
        grid_format : str, optional
            The format the grid information shall be stored as (in terms of variable attributes and dimensions).
            The default is "CF", which is also the only supported output format currently supported.
        """
        # Check inputs
        if filename:
            if "/" in str(filename):
                raise Exception(
                    "Target directory and filename have to be passed separately."
                )
            filename = Path(folder, filename).as_posix()
        else:
            filename = Path(folder, "grid_" + self.hash + ".nc").as_posix()

        # Write to disk (just horizontal coordinate variables + global attrs)
        #  if not written by another process
        if not os.path.isfile(filename):
            LOCK = filename + ".lock"
            lock_obj = FileLock(LOCK)
            try:
                lock_obj.acquire(timeout=10)
                locked = False
            except Exception as exc:
                if str(exc) == f"Could not obtain file lock on {LOCK}":
                    locked = True
                else:
                    locked = False
            if locked:
                warnings.warn(
                    f"Could not write grid '{filename}' to cache because a lockfile of "
                    "another process exists."
                )
            else:
                grid_tmp = Grid(ds=self.ds)
                if grid_tmp.format != grid_format:
                    grid_tmp.reformat(grid_format)
                grid_tmp.ds.attrs.update({"clisops": clversion})
                grid_tmp._drop_vars(keep_attrs=True)
                try:
                    grid_tmp.ds.to_netcdf(filename)
                finally:
                    lock_obj.release()


class Weights:
    """
    Creates remapping weights out of two Grid objects serving as source and target grid.

    Reads weights from cache if possible. Reads weights from disk if specified (not yet implemented).
    In the latter case, the weight file format has to be supported, to reformat it to xESMF format.

    Parameters
    ----------
    grid_in : Grid
        Grid object serving as source grid.
    grid_out : Grid
        Grid object serving as target grid.
    method : str
        Remapping method the weights should be / have been calculated with. One of ["nearest_s2d",
        "bilinear", "conservative", "patch"] if weights have to be calculated. Free text if weights
        are read from disk.
    from_disk : str, optional
        Not yet implemented. Instead of calculating the regridding weights (or reading them from
        the cache), read them from disk. The default is None.
    format: str, optional
        Not yet implemented. When reading weights from disk, the input format may be specified.
        If omitted, there will be an attempt to detect the format. The default is None.
    """

    @require_xesmf
    def __init__(
        self,
        grid_in: Grid,
        grid_out: Grid,
        method: str,
        from_disk: Optional[Union[str, Path]] = None,
        format: Optional[str] = None,
    ):
        """Initialize Weights object, incl. calculating / reading the weights."""
        if not isinstance(grid_in, Grid) or not isinstance(grid_out, Grid):
            raise InvalidParameterValue(
                "Input and output grids have to be instances of clisops.core.Grid."
            )
        self.grid_in = grid_in
        self.grid_out = grid_out
        self.method = method

        # Compare source and target grid
        if grid_in.hash == grid_out.hash:
            raise Exception(
                "The selected source and target grids are the same. "
                "No regridding operation required."
            )

        # Periodic in longitude
        # todo: properly test / check the periodic attribute of the xESMF Regridder.
        #  The grid.extent check done here might not be suitable to set the periodic attribute:
        #  global == is grid periodic in longitude
        self.periodic = False
        try:
            if self.grid_in.extent == "global":
                self.periodic = True
        except AttributeError:
            # forced to False for conservative regridding in xesmf
            #  todo: check if this is proper behaviour of xesmf
            if self.method not in ["conservative", "conservative_normed"]:
                warnings.warn(
                    "The grid extent could not be accessed. "
                    "It will be assumed that the input grid is not periodic in longitude."
                )

        # Activate ignore degenerate cells setting if collapsing cells are found within the grids.
        #  The default setting within ESMF is None, not False!
        self.ignore_degenerate = (
            True
            if (
                self.grid_in.contains_collapsing_cells
                or self.grid_out.contains_collapsing_cells
            )
            else None
        )

        self.id = self._generate_id()
        self.filename = "weights_" + self.id + ".nc"

        if not from_disk:
            # Read weights from cache or compute & save to cache
            self.format = "xESMF"
            self._compute()
        else:
            # Read weights from disk
            self._load_from_disk(self, filename=from_disk, format=format)

        # Reformat and cache the weights if required
        if not self.tool.startswith("xESMF"):
            self.format = self._detect_format()
            self._reformat("xESMF")

    def _compute(self):
        """Generate the weights with xESMF or read them from cache."""
        # Read weights_dir from CONFIG
        weights_dir = CONFIG["clisops:grid_weights"]["local_weights_dir"]

        # Check if bounds are present in case of conservative remapping
        if self.method in ["conservative", "conservative_normed"] and (
            not self.grid_in.lat_bnds
            or not self.grid_in.lon_bnds
            or not self.grid_out.lat_bnds
            or not self.grid_out.lon_bnds
        ):
            raise Exception(
                "For conservative remapping, horizontal grid bounds have to be defined for the source and target grids."
            )

        # Use "Locstream" functionality of xESMF as workaround for unstructured grids.
        #  Yet, the locstream functionality only supports the nearest neighbour remapping method
        locstream_in = False
        locstream_out = False
        if self.grid_in.type == "irregular":
            locstream_in = True
        if self.grid_out.type == "irregular":
            locstream_out = True

        # Read weights from cache (= reuse weights) if they are not currently written
        #  to the cache by another process
        #    Note: xESMF writes weights to disk if filename is specified and reuse_weights==False
        #          (latter is default) else it will create a default filename and weights can
        #          be manually written to disk with Regridder.to_netcdf(filename).
        #          Weights are read from disk by xESMF if filename is specified and reuse_weights==True.
        lock_obj = create_lock(Path(weights_dir, self.filename + ".lock").as_posix())
        if not lock_obj:
            warnings.warn(
                f"Could not reuse cached weights '{self.filename}' because a "
                "lockfile of another process exists that is writing to that file."
            )
            reuse_weights = False
            regridder_filename = None
        else:
            regridder_filename = Path(weights_dir, self.filename).as_posix()
            if os.path.isfile(regridder_filename):
                reuse_weights = True
            else:
                reuse_weights = False

        try:
            # Read the tool & version the weights have been computed with - backup: current version
            self.tool = self._read_info_from_cache("tool")
            if not self.tool:
                self.tool = "xESMF_v" + xe.__version__

            # Call xesmf.Regridder
            self.regridder = xe.Regridder(
                self.grid_in.ds,
                self.grid_out.ds,
                self.method,
                periodic=self.periodic,
                locstream_in=locstream_in,
                locstream_out=locstream_out,
                ignore_degenerate=self.ignore_degenerate,
                unmapped_to_nan=True,
                filename=regridder_filename,
                reuse_weights=reuse_weights,
            )

            # Save Weights to cache
            self._save_to_cache(lock_obj)
        finally:
            # Release file lock
            if lock_obj:
                lock_obj.release()

        # The default filename is important for later use, so it needs to be reset.
        self.regridder.filename = self.regridder._get_default_filename()

    def _generate_id(self):
        """
        Create an unique id for a Weights object.

        The id consists of
        - hashes / checksums of source and target grid (namely lat, lon, lat_bnds, lon_bnds variables)
        - info about periodicity in longitude
        - info about collapsing cells
        - remapping method

        Returns
        -------
        wid : str
            The id as str.
        """
        peri_dict = {True: "peri", False: "unperi"}
        ignore_degenerate_dict = {
            None: "no-degen",
            True: "skip-degen",
            False: "no-skip-degen",
        }
        wid = "_".join(
            filter(
                None,
                [
                    self.grid_in.hash,
                    self.grid_out.hash,
                    peri_dict[self.periodic],
                    ignore_degenerate_dict[self.ignore_degenerate],
                    self.method,
                ],
            )
        )
        return wid

    @check_weights_dir
    def _save_to_cache(self, store_weights: Union[FileLock, None, bool]):
        """Save Weights and source/target grids to cache (netCDF), incl. metadata (JSON)."""
        # Read weights_dir from CONFIG
        weights_dir = CONFIG["clisops:grid_weights"]["local_weights_dir"]

        # Compile metadata
        grid_in_source = self.grid_in.ds.encoding.get("source", "")
        grid_out_source = self.grid_out.ds.encoding.get("source", "")
        grid_in_tracking_id = self.grid_in.ds.attrs.get("tracking_id", "")
        grid_out_tracking_id = self.grid_out.ds.attrs.get("tracking_id", "")
        weights_dic = {
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
            "source_source": grid_in_source,
            "source_tracking_id": grid_in_tracking_id,
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
            "target_source": grid_out_source,
            "target_tracking_id": grid_out_tracking_id,
            "format": self.format,
            "ignore_degenerate": str(self.ignore_degenerate),
            "periodic": str(self.periodic),
            "method": self.method,
            "uid": self.id,
            "filename": self.filename,
            "def_filename": self.regridder._get_default_filename(),
            "tool": self.tool,
        }

        # Save Grid objects to cache
        self.grid_in.to_netcdf(folder=weights_dir)
        self.grid_out.to_netcdf(folder=weights_dir)

        # Save Weights object (netCDF) and metadata (JSON) to cache if desired
        #  (usually, if no lockfile exists)
        if store_weights:
            if not os.path.isfile(Path(weights_dir, self.filename).as_posix()):
                self.regridder.to_netcdf(Path(weights_dir, self.filename).as_posix())
            if not os.path.isfile(
                Path(weights_dir, Path(self.filename).stem + ".json").as_posix()
            ):
                with open(
                    Path(weights_dir, Path(self.filename).stem + ".json").as_posix(),
                    "w",
                ) as weights_dic_path:
                    json.dump(weights_dic, weights_dic_path, sort_keys=True, indent=4)

    @check_weights_dir
    def _read_info_from_cache(self, key: str):
        """
        Read info 'key' from cached metadata of current weightfile.

        Returns the value for the given key, unless the key does not exist in the metadata or the
        file cannot be read. In this case, None is returned.

        Parameters
        ----------
        key : str

        Returns
        -------
        str or None
            Value for the given key, or None.
        """
        # Read weights_dir from CONFIG
        weights_dir = CONFIG["clisops:grid_weights"]["local_weights_dir"]

        # Return requested value if weight and metadata files are present, else return None
        if os.path.isfile(
            Path(weights_dir, self.filename).as_posix()
        ) and os.path.isfile(
            Path(weights_dir, Path(self.filename).stem + ".json").as_posix()
        ):
            with open(
                Path(weights_dir, Path(self.filename).stem + ".json").as_posix()
            ) as f:
                weights_dic = json.load(f)
                try:
                    return weights_dic[key]
                except KeyError:
                    warnings.warn(
                        f"Requested info {key} does not exist in the metadata"
                        " of the cached weights."
                    )
                    return None
        else:
            warnings.warn(
                f"Requested info '{key}' could not be retrieved from remapping cache."
                " A cached file does not exist."
            )
            return None

    def save_to_disk(self, filename=None, wformat="xESMF"):
        """Write weights to disk in a certain format, not yet implemented."""
        # todo: if necessary, reformat weights, then save under specified path.
        raise NotImplementedError

    def _load_from_disk(self, filename=None, format=None):
        """Read and process weights from disk, not yet implemented."""
        # todo: Reformat to other weight-file formats when loading/saving from disk
        # if format != "xESMF":
        #  read file, compare Grid and weight matrix dimensions,
        #  reformat to xESMF sparse matrix and initialize xesmf.Regridder,
        #  generate_id, set ignore_degenerate, periodic, method to unknown if cannot be determined
        raise NotImplementedError

    def reformat(self, format_from: str, format_to: str):
        """Reformat remapping weights. Not yet implemented."""
        raise NotImplementedError

    def _detect_format(self, ds: Union[xr.Dataset, xr.DataArray]):
        """Detect format of remapping weights (read from disk), not yet implemented."""
        raise NotImplementedError


@require_xesmf
def regrid(
    grid_in: Grid,
    grid_out: Grid,
    weights: Weights,
    adaptive_masking_threshold: Optional[float] = 0.5,
    keep_attrs: Optional[bool] = True,
):
    """
    Perform regridding operation incl. dealing with dataset and variable attributes.

    Parameters
    ----------
    grid_in : Grid
        Grid object of the source grid, eg. created out of source xarray.Dataset.
    grid_out : Grid
        Grid object of the target grid.
    weights : Weights
        Weights object, as created by using grid_in and grid_out Grid objects as input.
    adaptive_masking_threshold : float, optional
        DESCRIPTION. The default is 0.5.
    keep_attrs : bool / str, optional
        Sets the global attributes of the resulting dataset, apart from the ones set by this routine:
            True: attributes of grid_in.ds will be in the resulting dataset.
            False: no attributes but the ones newly set by this routine
            "target": attributes of grid_out.ds will be in the resulting dataset.
        The default is True.

    Returns
    -------
    xarray.Dataset
        The regridded data in form of an xarray.Dataset.
    """
    if not isinstance(grid_out.ds, xr.Dataset):
        raise InvalidParameterValue(
            "The target Grid object 'grid_out' has to be built from an xarray.Dataset"
            " and not an xarray.DataArray."
        )

    # Create attrs
    attrs_append = {}
    if "grid" in grid_in.ds.attrs:
        attrs_append["grid_original"] = grid_in.ds.attrs["grid"]
    if "grid_label" in grid_in.ds.attrs:
        attrs_append["grid_label_original"] = grid_in.ds.attrs["grid_label"]
    nom_res_o = grid_in.ds.attrs.pop("nominal_resolution", None)
    if nom_res_o:
        attrs_append["nominal_resolution_original"] = nom_res_o
    # todo: should nominal_resolution of the target grid be calculated if not specified in the attr?
    nom_res_n = grid_out.ds.attrs.pop("nominal_resolution", None)
    if nom_res_n:
        attrs_append["nominal_resolution"] = nom_res_n

    # Remove all unnecessary coords, data_vars (and optionally attrs) from grid_out.ds
    if keep_attrs == "target":
        grid_out._drop_vars(keep_attrs=True)
    else:
        grid_out._drop_vars(keep_attrs=False)
    # Transfer all non-horizontal coords (and optionally attrs) from grid_out.ds to grid_in.ds
    grid_out._transfer_coords(grid_in, keep_attrs=keep_attrs)
    regridded_ds = grid_out.ds

    # Add new attrs
    regridded_ds.attrs.update(attrs_append)
    regridded_ds.attrs.update(
        {
            "grid": grid_out.label,
            "grid_label": "gr",  # regridded data reported on the data provider's preferred target grid
            "regrid_operation": weights.regridder.filename.split(".")[0],
            "regrid_tool": weights.tool,
            "regrid_weights_uid": weights.id,
        }
    )

    # todo: It might in general be sufficient to always act as if the threshold was
    #  set correctly and let xesmf handle it. But then we might not allow it for
    #  the bilinear method, as the results do not look too great and I am still
    #  not sure/convinced adaptive_masking makes sense for this method.

    # Allow Dataset and DataArray as input, but always return a Dataset
    if isinstance(grid_in.ds, xr.Dataset):
        for data_var in grid_in.ds.data_vars:
            if not all(
                [
                    dim in grid_in.ds[data_var].dims
                    for dim in grid_in.ds[grid_in.lat].dims
                    + grid_in.ds[grid_in.lon].dims
                ]
            ):
                continue
            if (
                weights.regridder.method
                in ["conservative", "conservative_normed", "patch"]
                and adaptive_masking_threshold >= 0.0
                and adaptive_masking_threshold <= 1.0
            ):
                regridded_ds[data_var] = weights.regridder(
                    grid_in.ds[data_var],
                    skipna=True,
                    na_thres=adaptive_masking_threshold,
                )
            else:
                regridded_ds[data_var] = weights.regridder(
                    grid_in.ds[data_var], skipna=False
                )
            if keep_attrs:
                regridded_ds[data_var].attrs.update(grid_in.ds[data_var].attrs)
        return regridded_ds
    else:
        if (
            weights.regridder.method in ["conservative", "conservative_normed", "patch"]
            and adaptive_masking_threshold >= 0.0
            and adaptive_masking_threshold <= 1.0
        ):
            regridded_ds[grid_in.ds.name] = weights.regridder(
                grid_in.ds, skipna=True, na_thres=adaptive_masking_threshold
            )
        else:
            regridded_ds[grid_in.ds.name] = weights.regridder(grid_in.ds, skipna=False)
        if keep_attrs:
            regridded_ds[data_var].attrs.update(grid_in.ds[data_var].attrs)
        return regridded_ds  # [grid_in.ds.name] #-> in case of returning a DataArray
