=======================================
clisops - climate simulation operations
=======================================

+----------------------------+---------------------------------+
| Versions                   | |pypi| |conda| |versions|       |
+----------------------------+---------------------------------+
| Documentation and Support  | |docs|                          |
+----------------------------+---------------------------------+
| Open Source                | |license|                       |
+----------------------------+---------------------------------+
| Coding Standards           | |black| |isort| |pre-commit|    |
+----------------------------+---------------------------------+
| Development Status         | |status| |build| |coveralls|    |
+----------------------------+---------------------------------+

The ``clisops`` package (pronounced "clie-sops") provides a python library for running
*data-reduction* operations on `Xarray <http://xarray.pydata.org/>`_ data sets or files
that can be interpreted by Xarray. These basic operations (subsetting, averaging and
regridding) are likely to work where data structures are NetCDF-centric, such as those
found in ESGF data sets.

``clisops`` is employed by the `daops <https://github.com/roocs/daops>`_ library to perform
its basic operations once ``daops`` has applied any necessary *fixes* to data in order
to remove irregularities/anomalies. Users are recommended to investigate using ``daops``
directly in order to access these *fixes* which may affect the scientific credibility of
the results.

``clisops`` can be used stand-alone to read individual, or groups of, NetCDF files directly.

* Free software: BSD
* Documentation: https://clisops.readthedocs.io.

Features
--------

The package provides the following operations:

* subset
* average
* regrid

Online Demo
-----------

..
  todo: Links have to be adjusted to the master or respective branch!

You can try clisops online using Binder (just click on the binder link below),
or view the notebooks on NBViewer.

.. image:: https://mybinder.org/badge_logo.svg
        :target: https://mybinder.org/v2/gh/roocs/clisops/master?filepath=notebooks
        :alt: Binder Launcher

.. image:: https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg
        :target: https://nbviewer.jupyter.org/github/roocs/clisops/tree/master/notebooks/
        :alt: NBViewer
        :height: 20

Credits
-------

This package was created with ``Cookiecutter`` and the ``audreyr/cookiecutter-pypackage`` project template.

* Cookiecutter: https://github.com/audreyr/cookiecutter
* cookiecutter-pypackage: https://github.com/audreyr/cookiecutter-pypackage


.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/python/black
        :alt: Black

.. |build| image:: https://github.com/roocs/clisops/workflows/main.yml/badge.svg
        :target: https://github.com/roocs/clisops/actions/workflows/main.yml
        :alt: Build Status

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/clisops.svg
        :target: https://anaconda.org/conda-forge/clisops
        :alt: Conda-forge Build Version

.. |coveralls| image:: https://coveralls.io/repos/github/roocs/clisops/badge.svg?branch=master
        :target: https://coveralls.io/github/roocs/clisops?branch=master
        :alt: Coveralls

.. |docs| image:: https://readthedocs.org/projects/clisops/badge/?version=latest
        :target: https://clisops.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
        :target: https://pycqa.github.io/isort/
        :alt: Isort

.. |license| image:: https://img.shields.io/github/license/roocs/clisops.svg
        :target: https://github.com/roocs/clisops/blob/master/LICENSE
        :alt: License

.. |pre-commit| image:: https://results.pre-commit.ci/badge/github/roocs/clisops/master.svg
        :target: https://results.pre-commit.ci/latest/github/roocs/clisops/master
        :alt: pre-commit.ci status

.. |pypi| image:: https://img.shields.io/pypi/v/clisops.svg
        :target: https://pypi.python.org/pypi/clisops
        :alt: Python Package Index Build

.. |status| image:: https://www.repostatus.org/badges/latest/active.svg
        :target: https://www.repostatus.org/#active
        :alt: Project Status: Active – The project has reached a stable, usable state and is being actively developed.

.. |versions| image:: https://img.shields.io/pypi/pyversions/clisops.svg
        :target: https://pypi.python.org/pypi/clisops
        :alt: Supported Python Versions
