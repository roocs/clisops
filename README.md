# clisops - climate simulation operations

[![Pypi](https://img.shields.io/pypi/v/clisops.svg)](https://pypi.python.org/pypi/clisops)

[![Travis](https://img.shields.io/travis/roocs/clisops.svg)](https://travis-ci.org/roocs/clisops)

[![Documentation](https://readthedocs.org/projects/clisops/badge/?version=latest)](https://clisops.readthedocs.io/en/latest/?badge=latest)

The `clisops` package (pronounced "clie-sops") provides a python library for running
_data-reduction_ operations on [Xarray](http://xarray.pydata.org/) data sets or files
that can be interpreted by Xarray. These basic operations (subsetting, averaging and
regridding) are likely to work where data structures are NetCDF-centric, such as those
found in ESGF data sets.

`clisops` is employed by the [daops](https://github.com/roocs/daops) library to perform
its basic operations once `daops` has applied any necessary _fixes_ to data in order
to remove irregularities/anomalies. Users are recommended to investigate using `daops`
directly in order to access these _fixes_ which may affect the scientific credibility of
the results.

`clisops` can be used stand-alone to read individual, or groups of, NetCDF files directly.

* Free software: BSD
* Documentation: https://clisops.readthedocs.io.


## Features

The package provides the following operations:
 * subset
 * average
 * regrid

# Credits

This package was created with `Cookiecutter` and the `audreyr/cookiecutter-pypackage` project template.

 * Cookiecutter: https://github.com/audreyr/cookiecutter
 * cookiecutter-pypackage: https://github.com/audreyr/cookiecutter-pypackage

[![Python Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
