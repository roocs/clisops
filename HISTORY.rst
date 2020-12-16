Version History
===============

v0.5.0 (2020-12-17)
------------------

Breaking Changes
^^^^^^^^^^^^^^^^
* Moved ``core.subset.create_mask_vectorize`` to ``core.subset.create_mask``. The old spatial join option was removed.
``core.subset.subset_shape`` lost its ``vectorize`` kwarg, as it is now default.
* ``roocs-utils``>0.1.5 used

Other Changes
^^^^^^^^^^^^^
* udunits2>=2.2 removed as a requirement to make clisops completely pip installable.
* rtee and libspatialindex removed as requirements, making it easier to install through pip.
* Static types updated to include missing but permitted types.
* Better handling for paths in ``ops.subset`` allowing windows build to be fixed.


v0.4.0 (2020-11-10)
-----------------

Adding new features, updating doc strings and documentation and inclusion of static type support.

Breaking Changes
^^^^^^^^^^^^^^^^
* clisops now requires udunits2>=2.2.
* roocs-utils>=0.1.4 is now required.
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
------------------

Other Changes
^^^^^^^^^^^^^
* Update testdata and subset module (#34).


v0.2.1 (2020-07-08)
-------------------

Other Changes
^^^^^^^^^^^^^
* Fixed docs version (#25).


v0.2.0 (2020-06-19)
------------------

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
------------------

* First release.
