Version History
===============

v0.7.0 (unreleased)
-------------------

This is a major release to help synchronize the package with the other `roocs` packages in advance of
an eventual merge with `CLISOPS`. This will be the final release of `roocs-utils` as a standalone package.

Breaking Changes
^^^^^^^^^^^^^^^^
* The package has been refactored to use `pyproject.toml` with the `flit-core` backend for packaging.
* The testing framework has been updated to use `pooch` for testing data caching, with on-the-fly data download and caching.
* `pre-commit` hooks have been updated and simplified.
* `pytest-xdist` has been added to the `dev` dependencies (distributed testing is disabled by default).
* `GitPython` has been removed from the dependencies in favour of `pooch` for testing data caching.

v0.6.9 (2024-07-15)
-------------------

Other Changes
^^^^^^^^^^^^^
* Documentation adjustments to address grammatical mistakes and improve clarity (#118).
* Dropped support for Python versions below 3.8, added support for Python3.12 (#118).
* Added a "dev" recipe to setup.py to install dev dependencies (#118).
* Updated pre-commit and adjusted to use Python3.8+ coding conventions (#118).
* Tests now explicitly use pytest fixtures (#118).
* Updated roocs.ini default values for the new CDS domain name (#118).

v0.6.8 (2024-04-17)
-------------------

Other Changes
^^^^^^^^^^^^^
* Fixed logging and updated conda env (#115).

v0.6.7 (2024-02-05)
-------------------

Other Changes
^^^^^^^^^^^^^
* Updated `roocs.ini` default values for atlas datasets for further methods of how to infer the project name (#113).

v0.6.6 (2024-01-26)
-------------------

Other Changes
^^^^^^^^^^^^^
* Updated `roocs.ini` default values for atlas datasets (#111).
* Updated `xarray_utils` module to support reading `kerchunk` files (#106).

v0.6.5 (2023-11-09)
-------------------

Other Changes
^^^^^^^^^^^^^
* Updated ``realization`` dimension in common coords (#108).
* Code linting.
* Added Python 3.11 to tests.
* Updated requirements for cf_xarray.

v0.6.4 (2023-02-01)
-------------------

Other Changes
^^^^^^^^^^^^^
* Added ``realization`` dimension to known coords (#103).
* Update pre-commit.

v0.6.3 (2022-09-26)
-------------------

Other Changes
^^^^^^^^^^^^^
* Added c3s-cmip-decadal project to default roocs.ini (#101).

v0.6.2 (2022-05-03)
-------------------

Bug Fixes
^^^^^^^^^
* Fixed ``get_coords_by_type`` in ``xarray_utils`` to handle non existing coords (#99).

v0.6.1 (2022-04-19)
-------------------

Bug Fixes
^^^^^^^^^
* Added data_node_root in ``roocs.ini`` for C3S-CORDEX and C3S-CMIP5 (#97).

v0.6.0 (2022-04-14)
-------------------

Bug Fixes
^^^^^^^^^
* Updated default ``roocs.ini`` for C3S-CORDEX (#93, #95).
* Fix added for `get_bbox <https://github.com/roocs/catalog-maker/issues/11>`_ on C3S-CORDEX (#94).

v0.5.0 (2021-10-26)
-------------------

Bug Fixes
^^^^^^^^^
* When a project was provided to ``roocs_utils.project_utils.DatasetMapper``, getting the base directory would be skipped, causing an error. This has been resolved.
* ``roocs_utils.project_utils.DatasetMapper`` can now accept `fixed_path_mappings` that include ".gz" (gzip) files. This is allowed because `Xarray` can read gzipped `netCDF` files.

Breaking Changes
^^^^^^^^^^^^^^^^
* Intake catalog maker removed, now in it's own package: `roocs/catalog-maker <https://github.com/roocs/catalog-maker>`_
* Change to input parameter classes:
    * Added: ``roocs_utils.parameter.time_components_parameter.TimeComponentsParameter``
    * Modified input types required for classes:
        * ``roocs_utils.parameter.time_parameter.TimeParameter``
        * ``roocs_utils.parameter.level_parameter.LevelParameter``
    * They both now require their inputs to be one of:
        * ``roocs_utils.parameter.param_utils.Interval`` - to specify a range/interval
        * ``roocs_utils.parameter.param_utils.Series`` - to specify a series of values

New Features
^^^^^^^^^^^^
* ``roocs_utils.xarray_utils.xarray_utils`` now accepts keyword arguments to pass through to xarray's ``open_dataset`` or ``open_mfdataset``. If the argument provided is not an option for ``open_dataset``, then ``open_mfdataset`` will be used, even for one file.
* The `roocs.ini` config file can now accept `fixed_path_modifiers` to work together with the `fixed_path_mappings` section. For example, you can specify parameters in the modifiers that will be expanded into the mappings::

    fixed_path_modifiers =
        variable:cld dtr frs pet pre tmn tmp tmx vap wet
    fixed_path_mappings =
        cru_ts.4.04.{variable}:cru_ts_4.04/data/{variable}/*.nc
        cru_ts.4.05.{variable}:cru_ts_4.05/data/{variable}/cru_ts4.05.1901.2*.{variable}.dat.nc.gz

  In this example, the `variable` parameter will be expanded out to each of the options provided in the list.
* The ``roocs_utils.xarray_utils.xarray_utils.open_xr_dataset()`` function was improved so that the time units of the first data file are preserved in: ``ds.time.encoding["units"]``. A multi-file dataset has now keeps the time "units" of the first file (if present). This is useful for converting to other formats (e.g. CSV).

Other Changes
^^^^^^^^^^^^^
* Python 3.6 no longer tested in GitHub actions.

v0.4.2 (2021-05-18)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Remove abcunit-backend and psycopg2 dependencies from requirements.txt, these must now be manually installed in order to use the catalog maker.

v0.4.0 (2021-05-18)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Inventory maker now removed and replaced by intake catalog maker which writes a csv file with the dataset entries and a yaml description file.
* In ``etc/roocs.ini`` the option ``use_inventory`` has been replaced by ``use_catalog`` and the inventory maker options have been replaced with equivalent catalog options. However, the option to include file paths or not no longer exists.
* The catalog maker now uses a database backend and creates a csv file so there are three new dependencies for the catalog maker: `pandas` and `abcunit-backend` and `psycopg2`.
  This means a database backend must be specified and the paths for the pickle files in ``etc/roocs.ini`` are no longer necessary. For more information see the README.

Other Changes
^^^^^^^^^^^^^
* `oyaml` removed as a dependency

v0.3.0 (2021-03-30)
-------------------

New Features
^^^^^^^^^^^^
* Added ``AnyCalendarDateTime`` and ``str_to_AnyCalendarDateTime`` to ``utils.time_utils`` to aid in handling date strings that may not exist in all calendar types.
* Inventory maker will check latitude and longitude of the dataset it is scanning are within acceptable bounds and raise an exception if they are not.

v0.2.1 (2021-02-19)
-------------------

Bug Fixes
^^^^^^^^^
* Cleaned up imports.
* Removed `pandas` dependency.

v0.2.0 (2021-02-18)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* cf_xarray>=0.3.1 now required due to differing level identification of coordinates between versions.
* oyaml>=0.9 - new dependency for inventory
* Interface to inventory maker changed. Detailed instructions for use added in README.
* Adjusted file name template. Underscore removed before ``__derive__time_range``
* New dev dependency: `GitPython==3.1.12`

New Features
^^^^^^^^^^^^
* Added ``use_inventory`` option to ``roocs.ini`` config and allow data to be used without checking an inventory.
* ``DatasetMapper`` class and wrapper functions added to ``roocs_utils.project_utils`` and ``roocs_utils.xarray_utils.xarray_utils`` to resolve all paths and dataset ids in the same way.
* ``FileMapper`` added in ``roocs_utils.utils.file_utils`` to resolve resolve multiple files with the same directory to their directory path.
* Fixed path mapping support added in ``DatasetMapper``
* Added ``DimensionParameter`` to be used with the average operation.

Other Changes
^^^^^^^^^^^^^
* Removed submodule for test data. Test data is now cloned from git using GitPython and cached
* ``CollectionParamter`` accepts an instance of ``FileMapper`` or a sequence of ``FileMapper`` objects
* Adjusted file name template to include an ``extra`` option before the file extension.
* Swapped from travis CI to GitHub actions

v0.1.5 (2020-11-23)
-------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Replaced use of ``cfunits`` by ``cf_xarray`` and ``cftime`` (new dependency) in ``roocs_utils.xarray_utils``.

v0.1.4 (2020-10-22)
-------------------

Fixing pip install

Bug Fixes
^^^^^^^^^
* Importing and using roocs-utils when pip installing now works

v0.1.3 (2020-10-21)
-------------------

Fixing formatting of doc strings and imports

Breaking Changes
^^^^^^^^^^^^^^^^
* Use of ``roocs_utils.parameter.parameterise.parameterise``: import should now be ``from roocs_utils.parameter import parameterise`` and usage should be, for example ``parameters = parameterise(collection=ds, time=time, area=area, level=level)``

New Features
^^^^^^^^^^^^

* Added a notebook to show examples

Other Changes
^^^^^^^^^^^^^
* Updated formatting of doc strings

v0.1.2 (2020-10-15)
-------------------

Updating the documentation and improving the changelog.

Other Changes
^^^^^^^^^^^^^
* Updated doc strings to improve documentation.
* Updated documentation.

v0.1.1 (2020-10-12)
-------------------

Fixing mostly existing functionality to work more efficiently with the other packages in roocs.

Breaking Changes
^^^^^^^^^^^^^^^^
* ``environment.yml`` has been updated to bring it in line with requirements.txt.
* ``level`` coordinates would previously have been identified as ``None``. They are now identified as ``level``.

New Features
^^^^^^^^^^^^
* ``parameterise`` function added in ``roocs_utils.parameter`` to use in all roocs packages.
* ``ROOCS_CONFIG`` environment variable can be used to override default config in ``etc/roocs.ini``. To use a local config file set ``ROOCS_CONFIG`` as the file path to this file. Several file paths can be provided separated by a ``:``
* Inventory functionality added - this can be used to create an inventory of datasets. See ``README`` for more info.
* ``project_utils`` added with the following functions to get the project name of a dataset and the base directory for
  that project.
* ``utils.common`` and ``utils.time_utils`` added.
* ``is_level`` implemented in ``xarray_utils`` to identify whether a coordinate is a level or not.

Bug Fixes
^^^^^^^^^
* ``xarray_utils.xarray_utils.get_main_variable`` updated to exclude common coordinates from the search for the main variable. This fixes a bug where coordinates such as ``lon_bounds`` would be returned as the main variable.

Other Changes
^^^^^^^^^^^^^
* ``README`` update to explain inventory functionality.
* ``Black`` and ``flake8`` formatting applied.
* Fixed import warning with ``collections.abc``.

v0.1.0 (2020-07-30)
-------------------

* First release.
