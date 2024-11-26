Version History
===============

v0.15.0 (2024-11-26)
--------------------

New Features
^^^^^^^^^^^^
* The functionality of the `roocs-utils` library has been folded into `clisops`. `roocs-utils` is no longer a dependency of `clisops` (#368, #370).
* The `CONFIG` settings have been updated to reflect the changes in `roocs-utils` and those required in `clisops` (#370).
* Several development tooling libraries have been updated to their latest versions (#368).

Other Changes
^^^^^^^^^^^^^
* The `pytest` testing suite has been updated to reflect the changes in the `roocs-utils` library and the new `clisops` configuration settings (#368, #370):
    * Tests previously found in `roocs-utils` have migrated to `clisops`.
    * ``xclim.testing.utils.open_dataset()`` has been deprecated in favour of ``xclim.testing.utils.nimbus().fetch()``.

v0.14.1 (2024-11-05)
--------------------

New Features
^^^^^^^^^^^^
* Added new methods to `clisops.core.regrid.Grid`
    * Added possibility to apply land or ocean mask if present in the file
    * Adapted method from `ESMF` to detect smashed cells
    * Masking degenerate (i.e. collapsed and smashed) cells
    * Dropping lat/lon bounds if an integrity check fails
    * Added a few attributes to the `Grid` object
* `clisops.ops.regrid`: added option to request a land/sea mask for the output grid
* `clisops.utils.dataset_utils`
    * Added function `determine_lon_lat_range` to determine the min. and max. lat and lon values
    * Added function `fix_unmasked_missing_values_lon_lat` to identify and mask yet unmasked missing values in lat and lon arrays
    * Added `force` parameter to `cf_convert_between_lon_frames`

Bug Fixes
^^^^^^^^^
* `clisops.utils.dataset_utils`
    * Fixed issue in `cf_convert_between_lon_frames` causing the longitude frame to not be adjusted in case of NaNs in the longitude array
    * Addressed issues in `generate_bounds_curvilinear`
        * latitudes are now clipped above 90 or below -90 degrees north
        * longitudes are converted to longitude frame -180, 180
        * longitude bounds are adjusted at the Greenwich meridian or anti meridian to avoid grid cells wrapping once or more times around the globe
        * Bounds are generated significantly faster due to making use of index slicing and `numpy.vectorize`

Breaking Changes
^^^^^^^^^^^^^^^^
* Adapted functions from `roocs_utils.xarray_utils.xarray_utils` into `clisops.utils.dataset_utils`
    * `get_coord_by_type` now returns the name of the coordinate variable and not the coordinate variable
    * `get_coord_by_type` optionally returns a list with further matches for the coordinate variable
    * `get_coord_by_type` does no longer raise an exception when more than one coordinate variable matches the requested type
    * `get_coord_by_type` raises `ValueError` instead of `Exception` when the coordinate type is unknown
    * `detect_coordinate` raises `KeyError` instead of `AttributeError` if no coordinate could be detected
    * `detect_gridtype` raises `ValueError` for unsupported grid types rather than `InvalidParameterValue` and `Exception`
* `clisops.core.regrid`
    * `Grid.detect_coordinate`: raises `KeyError` instead of `AttributeError` if no coordinate could be detected
* `clisops.ops.regrid`
    * `Regrid._calculate`: issues `UserWarning` instead of letting `clisops.core.Weights.__init__` raise an `Exception` when input and output grid are alike

Other Changes
^^^^^^^^^^^^^
* The testing suite has been refactored to make better use of context handlers when opening files with `xarray`, preventing synonymous read errors and improving the overall performance of the tests.
* Several tests that were failing due to significantly long runtimes have been marked as `slow` and are now skipped by default.
* GitHub Workflows now use a timeout of 20 minutes for the build suite to prevent hanging builds.

v0.14.0 (2024-10-03)
--------------------

New Features
^^^^^^^^^^^^
* `clisops` now makes use of `pytest-xdist` for parallel testing. This can be enabled using `--numprocesses={int}`. See the `pytest-xdist documentation <https://pytest-xdist.readthedocs.io/en/latest/>`_ for more information (#345).
* Testing data caching is now handled by `pooch` and testing data registries ('stratus' for `roocs/mini-esgf-data` and 'nimbus' for `Ouranosinc/xclim-testdata`) (#345).
* `clisops` coding conventions now use Python 3.9+ conventions (#345).

Breaking Changes
^^^^^^^^^^^^^^^^
* `clisops` has dropped support for Python 3.8 (#345).
* Several dependencies have been updated to include lower bounds for clearer compatibility and easier maintenance (#345, #XYZ).
    * The affected core dependencies are: `dask >=2023.6.0`, `filelock >=3.15.4`, `geopandas >=0.14.0`, `jinja2 >=2.11`, `numpy >=1.23.0`, `packaging >=23.2`, `pandas >=1.5.0`, `pooch >=1.8.0`, `scipy >=1.9.0`, and `xarray >=2022.6.0`.
    * Extra dependencies are `ipython >=8.5.0`, `matplotlib >=3.6.0`, `nbconvert >=7.14.0`, `nbsphinx >=0.9.5`, `pre-commit >=3.5.0`, and `sphinx >=7.0.0`.
* `clisops` no longer requires `gitpython >=3.1.30` and `requests >=2.0` (#345).
* The development dependencies have been updated to include `deptry >=0.20.0` and `pytest-xdist[psutil] >=3.2` (#345).
* `netCDF4` has been moved from core dependency to development dependency (#345).

Other Changes
^^^^^^^^^^^^^
* `clisops.utils.testing` has replaced `clisops.utils.tutorial`. This submodule contains several functions and variables for allowing user control over testing data fetching (#345).
* The `_common` testing tools have been migrated to `clisops.utils.testing` or rewritten as `pytest` fixtures (#345).
* Testing data fetching now uses worker threads to copy cached data to threadsafe data caches that are separated by worker (#345).

v0.13.1 (2024-08-20)
--------------------

Bug Fixes
^^^^^^^^^
* Changed the order of operations in `clisops.core.subset.subset_shape` to ensure that the CRS of the shapefile is compatible with the dataset CRS before attempting to subset (#340).

Breaking Changes
^^^^^^^^^^^^^^^^
* Anaconda builds now require `cartopy >=0.23` and only support Python 3.9 and above (#340).
* Many dependency version pins now include lower bounds for clearer compatibility and easier maintenance (#343).

Other Changes
^^^^^^^^^^^^^
* Internal warnings now consistently use the `clisops` configured `loguru` logger (#335).
* CI Actions now use the commit hashes for version tracking (#343).

v0.13.0 (2024-02-16)
--------------------

New Features
^^^^^^^^^^^^
* `clisops` now officially supports Python 3.12 (#330).

Bug Fixes
^^^^^^^^^
* Fixed standard file-namer fallback method (#318).
* Fixed `KeyError` for temporal subsetting by components if not all components can be found in the dataset (#316).
* Raising `KeyError` for temporal subsetting by components when no time steps match the selection criteria (#316).
* Coordinate detection for remapping operator via standard_name if detection via `cf-xarray` fails / is ambiguous (#316).
* Remove encoding settings with regards to compression for string variables to avoid netCDF write errors with newer `netcdf-c` library versions (>4.9.0) (#319).
* Fixed a few docstrings, specifies some class methods as static methods (#321).
* Renamed a few internal variables for clarity, rephrased a few sentences for grammar/spelling (#321).
* Fixed a bug related to the creation of the `weights_dir` for regridding that was causing issues for Windows platforms (#313).

Other Changes
^^^^^^^^^^^^^
* The compression level is capped at 1 to reduce write times (#319).
* Updated `pre-commit` hooks, pinned linting tools to their pre-commit equivalents (#321).
* Added a pre-commit hook as well as a configuration for `codespell` (#321).
* Added `dependabot` to maintain package and GitHub Action versions (#322).
* The `require_module` decorator can now accept supported version information (#321).
* Testing data caching now uses platformdirs to determine the OS-appropriate caching location (#321).
* Updated `black` in linting tools to v24.2.0 (#330).
* Changes some print calls into logging calls in the tests (#330).
* A warning is now emitted on `clisops` import if the installed `xesmf` is too old (#330).
* Replaced `styfle/cancel-workflow-action` with GitHub Workflow concurrency settings (#330).

v0.12.2 (2024-01-03)
--------------------

New Features
^^^^^^^^^^^^
* ``clisops.ops.average.average_shape`` added (#312). Exposing average_shape from clisops.core to clisops.ops.

Bug Fixes
^^^^^^^^^
* Now also applying fix for datasets with shifted longitude frames (#218) for the regrid operator (#313).

Other Changes
^^^^^^^^^^^^^
* Warnings are now emitted when the user attempts to regrid a zonal mean dataset (#313).

v0.12.1 (2023-11-30)
--------------------

Bug Fixes
^^^^^^^^^
* Instead of raising an exception, now aligning _FillValue and missing_value if they deviate from one another. (#309).

Other Changes
^^^^^^^^^^^^^
* Warnings are now emitted if the user attempts to run the regridding utilities with a version of `xarray` that is not compatible with `cf-xarray`. (#310).
* Dependency pins now constrain the `xarray` version when installing with `$ pip install ".[extra]"`. (#310).

v0.12.0 (2023-11-23)
--------------------

New Features
^^^^^^^^^^^^
* ``clisops.ops.regrid``, ``clisops.core.regrid``, ``clisops.core.Weights`` and ``clisops.core.Grid`` added (#243). Allowing the remapping of geospatial data on various grids by applying the `xESMF <https://pangeo-xesmf.readthedocs.io/en/latest/>`_ regridder.

Bug Fixes
^^^^^^^^^
* Calling `subset_shape()` with a `locstream case` (#288) returned all coordinates inside `inner_mask` which is equivalent to the bounding box of the polygon, not the area inside the polygon. Fixed by defining the `inner_mask` in `subset_shape()` for the locstream case. (#292).

Other Changes
^^^^^^^^^^^^^
* Extending the removal of redundant _FillValue attributes to all data variables and coordinates (#243).
* Extending the removal of redundant coordinates in the coordinates variable attribute from bounds to all data variables (#243).
* GitHub Workflows for upstream dependencies are now examined a schedule or via `workflow_dispatch` (#243).
* `black` steps are now called `lint` for clarity/inclusiveness of other linting hooks. (#243).
* pre-commit hooks now include checks for TOML files, and for ReadTheDocs and GitHub Actions configuration files. (#243).
* pre-commit hooks now include sorting of TOML file sections and running `black` on docstring Python examples. (#306).
* `clisops` now uses GitHub Actions with environments for handling deployment via Trusted Publishing. (#306).
* Documentation has been updated to reflect the new GitHub Actions CI/CD workflow. (#306).
* `bump2version` has been replaced with `bump-my-version` for handling versioning. (#306).

v0.11.0 (2023-08-22)
--------------------

New Features
^^^^^^^^^^^^
* `clisops` has adopted `PEP 517 <https://peps.python.org/pep-0517/>`_ and `PEP 621 <https://peps.python.org/pep-0621/>`_ and now uses ``pyproject.toml`` files (using the `flit` backend) for package configuration. (#296).
* Metadata has been modified to reflect current development status and scope of CLISOPS. (#296).
* New file (``requirements_upstream.txt``) and Makefile recipe (``"$ make upstream"``) for tracking and easily installing upstream dependencies. (#296).

Bug Fixes
^^^^^^^^^
* The ``tests`` folder has been flattened and namespace files haves been removed in order to prevent `pip` from recognizing the folder as its own package. (#296).
* The contribution guidelines were duplicated in two locations and contained conflicting information. The guidelines have now been consolidated into a single location and updated to reflect package changes. (#296).

Other Changes
^^^^^^^^^^^^^
* GitHub Workflows for pure Python builds now use `tox` (4.0) to run tests. (#296).
* GitHub Workflows for conda builds now test `clisops` using the ``mamba-org/setup-micromamba`` action. (#296).
* The `travis.yml` file has been removed. (#296).

v0.10.1 (2023-08-21)
--------------------

Bug Fixes
^^^^^^^^^
* Fixed an issue with the type hinting for subset functions that were broken due to changes in `xarray` (2023.08). (#295).
* Updated ReadTheDocs configuration to use `Mambaforge` (22.9) as engine for building documentation. (#295).

v0.10.0 (2023-06-28)
--------------------

New Features
^^^^^^^^^^^^
* Added support for Python 3.11 (#287).

Bug Fixes
^^^^^^^^^
* Fixed bug in `core.subset.shape_bbox_indexer` with the union of invalid geometries. Added regression test. (#280)
* Added support in `core.subset.shape_bbox_indexer` for Point and MultiPoint geometries. (#283)
* Fixed `core.subset.subset_bbox` and `core.subset.subset_shape` for datasets with 1D longitude and latitude (ex: Station data). (#288)

Other Changes
^^^^^^^^^^^^^
* Shapely 2.0 is now faster than pygeos for ``create_mask``. Removed pygeos from extra dependencies and pinned shapely above 2.0. (#289)

v0.9.6 (2023-04-05)
-------------------

Bug Fixes
^^^^^^^^^
* Fixed an issue with the `pytest` fixtures that was needlessly calling ``load_esgf_test_data`` multiple times while tests were running (#278).
* Corrected a temporary workaround for updating split geometries that was causing issues with modern `pandas` versions (#278).

Other Changes
^^^^^^^^^^^^^
* Removed some obsolete tests and adjusted pytest to always report in colour (#272).
* Split conda CI builds to explicitly test against xarray/stable and xarray/dev (#272).
* GitHub CI now reports coverage statistics to Coveralls.io (#276).
* Updated `geopandas` (>=0.11), `pyproj` (>=3.3.0), `shapely` (>=1.9), `tox` (>=4.0), `xarray` (>=0.21), and `xesmf` (>=0.6.3) to use more modern versions (#278).

v0.9.5 (2022-12-14)
-------------------

Bug Fixes
^^^^^^^^^
* Fixed `core.subset.check_levels_exist` decorator by rounding (precision 4) level values like 1000.00000001 (#265).

v0.9.4 (2022-12-13)
-------------------

Bug Fixes
^^^^^^^^^
* Fixed `core.subset_bbox` when using `level_values` (#263).
* Fixed `core.subset_level_by_values` using xarray method *nearest* (#262).
* Updated a test expectation to support newer xarray behaviour (#259).

v0.9.3 (2022-10-03)
-------------------

Bug Fixes
^^^^^^^^^
* Fixed a bug associated with the new xarray (2022.6.0+) accessor for native indexers that was introduced in (#241). (#250, #251).

Other Changes
^^^^^^^^^^^^^
* Fixed a handful of static type hints that were sending out warnings, despite proper use. (#251).
* Replaced all skipped doctests with sphinx-compatible python code blocks to prevent errors in downstream projects. (#251).
* Adjusted GitHub Actions builds to ensure that the `conda-xesmf` run uses the latest `xarray` available. (#251).

v0.9.2 (2022-09-06)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Support has been dropped for Python3.7 and extended to Python3.10. Python3.7 is no longer tested in GitHub actions (#234).
* ``packaging`` has been added as a dependency (#241).

Bug Fixes
^^^^^^^^^
* Adapted ``clisops.core.subset_bbox_indexer`` to the newest indexing API changes in xarray, with backwards compatibility (#241).

Other Changes
^^^^^^^^^^^^^
* Docstrings and documentation configuration adjustments have been made to ensure that builds are adequately tested (#232, #235).

v0.9.1 (2022-05-12)
-------------------

Bug fixes
^^^^^^^^^
* Fix inconsistent bounds in metadata after subset operation (#224).

Other Changes
^^^^^^^^^^^^^
* Use ``roocs-utils`` 0.6.2 to avoid test failure (#226).
* Removed unneeded testing dep from environment.yml (#223).
* Merged pre-commit autoupdate (#227).

v0.9.0 (2022-04-13)
-------------------

New Features
^^^^^^^^^^^^
* ``clisops.ops.average.average_time`` and ``clisops.core.average.average_time`` added (#211). Allowing averaging over time frequencies of day, month and year.
* New function ``create_time_bounds`` in  ``clisops.utils.time_utils``, to generate time bounds for temporally averaged datasets.

* ``clisops`` now uses the `loguru <https://loguru.readthedocs.io/en/stable/index.html>`_ library as its primary logging engine (#216).
  The mechanism for enabling log reporting in scripts/notebooks using ``loguru`` is as follows:

.. code-block:: python

    import sys
    from loguru import logger

    logger.activate("clisops")
    LEVEL = "INFO || DEBUG || WARNING || etc."
    logger.add(sys.stdout, level=LEVEL)  # for logging to stdout
    # or
    logger.add("my_log_file.log", level=LEVEL, enqueue=True)  # for logging to a file

Other Changes
^^^^^^^^^^^^^
* Pandas now pinned below version 1.4.0.
* Pre-commit configuration updated with code style conventions (black, pyupgrade) set to Python3.7+ (#219).
* ``loguru`` is now an install dependency, with ``pytest-loguru`` as a development-only dependency.
* Added function to convert the longitude axis between different longitude frames (eg. [-180, 180] and [0, 360]) (#217, #218).

v0.8.0 (2022-01-13)
-------------------

New Features
^^^^^^^^^^^^
* ``clisops.core.average.average_shape`` copies the global and variable attributes from the input data to the results.
* ``clisops.ops.average.average_time`` and ``clisops.core.average.average_time`` added. Allowing averaging over time frequencies of day, month and year.
* New function ``create_time_bounds`` in  ``clisops.utils.time_utils``, to generate time bounds for temporally averaged datasets.

Bug fixes
^^^^^^^^^
* ``average_shape`` and ``create_weight_masks`` were adapted to work with xESMF 0.6.2, while maintaining compatibility with earlier versions.
* Fix added to remove ``_FillValue`` added to coordinate variables and bounds by xarray when outputting to netCDF.

Other Changes
^^^^^^^^^^^^^
* Passing ``DataArray`` objects to ``clisops.core.average.average_shape`` is now deprecated. Averaging requires grid cell boundaries, which are not ``DataArray`` coordinates, but independent ``Dataset`` variables. Please pass ``Dataset`` objects and an optional list of variables to average.
* ``average_shape`` performs an initial subset over the averaging region, before computing the weights, to reduce memory usage.
* Minimum xesmf version set to 0.6.2.
* Minimum pygeos version set to 0.9.
* Replace ``cascaded_union`` by ``unary_union`` to anticipate a `shapely` deprecation.

v0.7.0 (2021-10-26)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* ``time`` input for ``time`` in ``ops.subset.subset`` but now be one of [<class 'roocs_utils.parameter.param_utils.Interval'>, <class 'roocs_utils.parameter.param_utils.Series'>, <class 'NoneType'>, <class 'str'>].
* ``level`` input for ``level`` in ``ops.subset.subset`` but now be one of [<class 'roocs_utils.parameter.param_utils.Interval'>, <class 'roocs_utils.parameter.param_utils.Series'>, <class 'NoneType'>, <class 'str'>].
* ``roocs-utils``>= 0.5.0 required.

New Features
^^^^^^^^^^^^
* ``time_values`` and ``level_values`` arguments added to ``core.subset.subset_bbox`` which allows the user to provide a list of time/level values to select.
* ``subset_time_by_values`` and ``subset_level_by_values`` added to ``core.subset.subset_bbox``. These allow subsetting on sequence of datetimes or levels.
* ``subset_time_by_components`` added to ``core.subset.subset_bbox``. This allows subsetting by time components - year, month,  day etc.
* ``check_levels_exist`` and ``check_datetimes_exist`` function checkers added in ``core.subset`` to check requested levels and datetimes exist. An exception is raised if they do not exist in the dataset.
* ``time_components`` argument added to ``ops.subset`` to allowing subsetting by time components such as year, month, day etc.

Other Changes
^^^^^^^^^^^^^
* Python 3.6 no longer tested in GitHub actions.

v0.6.5 (2021-06-10)
-------------------

New Features
^^^^^^^^^^^^
* New optional dependency ``PyGEOS``, when installed the performance of ``core.subset.create_mask`` and ``cure.subset.subset_shape`` are greatly improved.

v0.6.4 (2021-05-17)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Exception raised in ``core.average.average_over_dims`` when dims is None.
* Exception raised in ``core.average.average_over_shape`` when grid and polygon have no overlapping values.

New Features
^^^^^^^^^^^^
* ``ops.subset.subset`` now ensures all latitude and longitude bounds are in ascending order before passing to ``core.subset.subset_bbox``
* ``core.subset.subset_level`` now checks that the order of the bounds matches the order of the level data.
* ``core.subset._check_desc_coords`` now checks the bounds provided are ascending before flipping them.

Other Changes
^^^^^^^^^^^^^
* clisops logging no longer disables other loggers.
* GitHub CI now leverages ``tox`` for testing as well as tests averaging functions via a conda-based build.
* Added a CI build to run against xarray@master that is allowed to fail.

v0.6.3 (2021-03-30)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Raise an exception in ``core.subset.subset_bbox`` when there are no data points in the result.
* ``roocs-utils``>=0.3.0 required.

Bug Fixes
^^^^^^^^^
* In ``core.subset.check_start_end_dates`` check if start and end date requested exist in the calendar of the dataset. If not, nudge the date forward if start date or backwards if end date.

Other Changes
^^^^^^^^^^^^^
* Error message improved to include longitude bounds of the dataset when the bounds requested in ``ops.subset.subset`` are not within range and rolling could not be completed.

v0.6.2 (2021-03-22)
-------------------

Bug Fixes
^^^^^^^^^
* Better support for disjoint shapes in ``subset_shape``.
* Identify latitude and longitude using ``cf-xarray`` rather than by "lat" and "lon"

New Features
^^^^^^^^^^^^
* Add ``output_staging_dir`` option in `etc/roocs.ini`, to write files to initially before moving them to the requested output_dir.
* Notebook of examples for average over dims operation added.

v0.6.1 (2021-02-23)
-------------------

Bug Fixes
^^^^^^^^^
* Add ``cf-xarray`` as dependency. This is a dependency of ``roocs-utils``>=0.2.1 so is not a breaking change.
* Remove ``python-dateutil``, ``fiona`` and ``geojson`` as dependencies, no longer needed.

v0.6.0 (2021-02-22)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* New dev dependency: ``GitPython``\ ==3.1.12
* ``roocs-utils``>=0.2.1 required.

New Features
^^^^^^^^^^^^
* ``average_over_dims`` added into ``average.core`` and ``average.ops``
* New ``core.average.average_shape`` + ``core.subset.subset_create_weight_masks``. Depends on `xESMF` >= 0.5.2, which is a new optional dependency.

Bug Fixes
^^^^^^^^^
* Fixed issue where the temporal subset was ignored if level subset selected.
* Roll dataset used in subsetting when the requested longitude bounds are not within those of the dataset.
* Fixed issue with subsetting grid lon and lat coordinates that are in descending order for ``core.subset.subset_bbox``.

Other Changes
^^^^^^^^^^^^^
* Changes to allow datasets without a time dimension to be processed without issues.
* Use ``DatasetMapper`` from ``roocs-utils`` to ensure all datasets are mapped to file paths correctly.
* Using file caching to gather ``mini-esgf-data`` test data.
* Added a ``dev`` recipe for pip installations (`pip install clisops[dev]`).
* Updated pre-commit and pre-commit hooks to newest versions.
* Migrated linux-based integration builds to GitHub CI.
* Added functionality to ``core.subset.create_mask`` so it can accept ``GeoDataFrames`` with non-integer indexes.
* ``clisops.utils.file_namers`` adjusted to allow values to be overwritten and extras to be added to the end before the file extension.

v0.5.1 (2021-01-11)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Reverting breaking changes made by the change to ``core.subset.create_mask``. This change introduces a second evaluation for shapes touching grid-points.


Other Changes
^^^^^^^^^^^^^
* Using file caching to gather ``xclim`` test data.
* Change made to ``core.subset.subset_bbox._check_desc_coords`` to cope with subsetting when only one latitude or longitude exists in the input dataset

v0.5.0 (2020-12-17)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Moved ``core.subset.create_mask_vectorize`` to ``core.subset.create_mask``. The old spatial join option was removed.
* ``core.subset.subset_shape`` lost its ``vectorize`` kwarg, as it is now default.
* ``roocs-utils``>0.1.5 used

Other Changes
^^^^^^^^^^^^^
* ``udunits2``>=2.2 removed as a requirement to make clisops completely pip installable.
* ``rtee`` and ``libspatialindex`` removed as requirements, making it easier to install through pip.
* Static types updated to include missing but permitted types.
* Better handling for paths in ``ops.subset`` allowing windows build to be fixed.


v0.4.0 (2020-11-10)
-------------------

Adding new features, updating doc strings and documentation and inclusion of static type support.

Breaking Changes
^^^^^^^^^^^^^^^^
* ``clisops`` now requires ``udunits2``>=2.2.
* ``roocs-utils``>=0.1.4 is now required.
* ``space`` parameter of ``clisops.ops.subset`` renamed to ``area``.
* ``chunk_rules`` parameter of ``clisops.ops.subset`` renamed to ``split_method``.
* ``filenamer`` parameter of ``clisops.ops.subset`` renamed to ``file_namer``.

New Features
^^^^^^^^^^^^
* ``subset_level`` added.
* PR template.
* Config file now exists at ``clisops.etc.roocs.ini``. This can be overwritten by setting the environment variable
  ``ROOCS_CONFIG`` to the file path of a config file.
* Static typing added to subset operation function.
* info and debugging are now logged rather than printed.
* Notebook of examples for subset operation added.
* ``split_method`` implemented to split output files by if they exceed the memory limit provided in
  ``clisops.etc.roocs.ini`` named ``file_size_limit``.
  Currently only the ``time:auto`` exists which splits evenly on time ranges.
* ``file_namer`` implemented in ``clisops.ops.subset``. This has ``simple`` and ``standard`` options.
  ``simple`` numbers output files whereas ``standard`` names them according to the input dataset.
* Memory usage when completing the subsetting operation is now managed using dask chunking. The memory limit for
  memory usage for this process is set in ``clisops.etc.roocs.ini`` under ``chunk_memory_limit``.

Bug Fixes
^^^^^^^^^
* Nudging time values to nearest available in dataset to fix a bug where subsetting failed when the exact date
  did not exist in the dataset.

Other Changes
^^^^^^^^^^^^^
* ``cfunits`` dependency removed - not needed.
* requirements.txt and environment.yml synced.
* Documentation updated to include API.
* Read the docs build now tested in CI pipeline.
* md files changed to rst.
* tests now use ``mini-esgf-data`` by default.

v0.3.1 (2020-08-04)
-------------------

Other Changes
^^^^^^^^^^^^^
* Add missing ``rtree`` dependency to ensure correct spatial indexing.

v0.3.0 (2020-07-23)
-------------------

Other Changes
^^^^^^^^^^^^^
* Update testdata and subset module (#34).

v0.2.1 (2020-07-08)
-------------------

Other Changes
^^^^^^^^^^^^^
* Fixed docs version (#25).

v0.2.0 (2020-06-19)
-------------------

New Features
^^^^^^^^^^^^^
* Integration of xclim subset module in ``clisops.core.subset``.
* Added jupyter notebook with and example for subsetting from xclim.

Other Changes
^^^^^^^^^^^^^
* Fixed RTD doc build.
* Updated travis CI according to xclim requirements.
* Now employing PEP8 + Black compatible autoformatting.
* Pre-commit is now used to launch code formatting inspections for local development.

v0.1.0 (2020-04-22)
-------------------

* First release.
