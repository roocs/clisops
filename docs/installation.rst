.. highlight:: shell

============
Installation
============


Stable release
--------------

To install clisops, run this command in your terminal:

.. code-block:: console

    $ pip install clisops

This is the preferred method to install clisops, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

.. warning::

   Some average operations (`clisops.core.average_shape`) require at least version 0.5.2 of the `xESMF` package.
   Unfortunately, this package is not available on pypi at the time these lines were written and it also depends
   on packages (ESMF, ESMpy) unavailable on windows.  It can still be installed on osx/linux through `conda` or
   directly [from source](https://github.com/pangeo-data/xESMF/).

From sources
------------

The sources for clisops can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/roocs/clisops

Create Conda environment named `clisops`.

.. warning::

    For windows users, you might need to edit the ``environment.yml`` file
    and remove the `xESMF` package.

.. code-block:: console

   $ conda env create -f environment.yml
   $ source activate clisops

Install clisops in development mode:

.. code-block:: console

  $ pip install -r requirements.txt
  $ pip install -r requirements_dev.txt
  $ python setup.py develop

Run tests:

.. code-block:: console

    $ pytest -v tests/

.. _Github repo: https://github.com/roocs/clisops
