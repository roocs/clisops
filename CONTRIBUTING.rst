============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/roocs/clisops/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

clisops could always use more documentation, whether as part of the
official clisops docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/roocs/clisops/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `clisops` for local development.

#.
    Fork the `clisops` repo on GitHub.

#.
    Clone your fork locally:

    .. code-block:: shell

        $ git clone git@github.com:roocs/clisops.git

#.
    Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

    .. code-block:: shell

        # For virtualenv environments:
        $ mkvirtualenv clisops

        # For Anaconda/Miniconda environments:
        $ conda create -n clisops python=(3.9, 3.10, 3.11, etc.)

        $ cd clisops/
        $ pip install -e .

#.
    Create a branch for local development:

    .. code-block:: shell

        $ git checkout -b name-of-your-bugfix-or-feature

    Now you can make your changes locally!

#.
    When you are done making changes, check that you verify your changes with `flake8` and `black` and run the tests, including testing other Python versions with `tox`:

    .. code-block:: shell

        # For virtualenv environments:
        $ pip install flake8 black pytest pytest-loguru tox

        # For Anaconda/Miniconda environments:
        $ conda install -c conda-forge flake8 black pytest pytest-loguru tox

        $ flake8 clisops tests
        $ black clisops tests
        $ pytest
        $ tox

#.
    Before committing your changes, we ask that you install `pre-commit` in your virtualenv. `Pre-commit` runs git hooks that ensure that your code resembles that of the project and catches and corrects any small errors or inconsistencies when you `git commit`:

    .. code-block:: shell

        # For virtualenv environments:
        $ pip install pre-commit

        # For Anaconda/Miniconda environments:
        $ conda install -c conda-forge pre-commit

        $ pre-commit install
        $ pre-commit run --all-files

#.
    Commit your changes and push your branch to GitHub:

    .. code-block:: shell

        $ git add *

        $ git commit -m "Your detailed description of your changes."
        # `pre-commit` will run checks at this point:
        # if no errors are found, changes will be committed.
        # if errors are found, modifications will be made. Simply `git commit` again.

        $ git push origin name-of-your-bugfix-or-feature

#.
    Submit a pull request through the GitHub website.

Logging
-------

``clisops`` uses the `loguru`_ library as its primary logging engine. In order to integrate this kind of logging in processes, we can use their logger:

.. code-block:: python

    from loguru import logger

    logger.warning("This a warning message!")

The mechanism for enabling log reporting in scripts/notebooks using ``loguru`` is as follows:

.. code-block:: python

    import sys
    from loguru import logger

    logger.enable("clisops")
    LEVEL = "INFO || DEBUG || WARNING || etc."
    logger.add(sys.stdout, level=LEVEL)  # for logging to stdout
    # or
    logger.add("my_log_file.log", level=LEVEL, enqueue=True)  # for logging to a file

For convenience, a preset logger configuration can be enabled via `clisops.enable_logging()`.

.. code-block:: python

    from clisops import enable_logging

    enable_logging()


Pull Request Guidelines
-----------------------

Before you submit a pull request, please follow these guidelines:

#.
    Open an *issue* on our `GitHub repository`_ with your issue that you'd like to fix or feature that you'd like to implement.

#.
    Perform the changes, commit and push them either to new a branch within roocs/clisops or to your personal fork of clisops.

    .. warning::
        Try to keep your contributions within the scope of the issue that you are addressing.
        While it might be tempting to fix other aspects of the library as it comes up, it's better to
        simply to flag the problems in case others are already working on it.

        Consider adding a "**# TODO:**" comment if the need arises.

#.
    Pull requests should raise test coverage for the clisops library. Code coverage is an indicator of how extensively tested the library is.
    If you are adding a new set of functions, they **must be tested** and **coverage percentage should not significantly decrease.**

#.
    If the pull request adds functionality, your functions should include docstring explanations.
    So long as the docstrings are syntactically correct, sphinx-autodoc will be able to automatically parse the information.
    Please ensure that the docstrings adhere to one of the following standards:

    * `numpydoc`_
    * `reStructuredText (ReST)`_

    The version history should also be updated.
    Remember to add the feature or bug fixes explanation to the appropriate section in the HISTORY.rst.

#.
    The pull request should work for Python 3.9+ as well as raise test coverage.
    Pull requests are also checked for documentation build status and for `PEP8`_ compliance.

    The build statuses and build errors for pull requests can be found at:
    https://github.com/roocs/clisops/actions/workflows/main.yml

    .. warning::
        `PEP8`_ and `black` formatting is strongly enforced.
        Ensure that your changes pass **Flake8** and **Black** tests prior to pushing your final commits to your branch.
        Code formatting errors are treated as build errors and will block your pull request from being accepted.

Tips
----

To run a subset of tests:

.. code-block:: shell

    $ pytest tests.test_clisops

Versioning
----------

A reminder for the maintainers on how to bump the version.

In order to update and release the library to PyPI, it's good to use a semantic versioning scheme.

The method we use is as follows::

    major.minor.patch

**Major** releases denote major changes resulting in a stable API;

**Minor** is to be used when adding a module, process or set of components;

**Patch** should be used for bug fixes and optimizations;

Packaging/Deploying
-------------------

A reminder for the maintainers on how to deploy. This section is only relevant for maintainers when they are producing a new point release for the package.

#. Create a new branch from `master` (e.g. `prepare-release-v1.2.3`).
#. Update the `HISTORY.rst` file to change the `unreleased` section to the current date.
#. Create a pull request from your branch to `master`.
#. Once the pull request is merged, create a new release on GitHub. On the main branch, run:

    .. code-block:: shell

        $ bump-my-version bump minor # In most cases, we will be releasing a minor version
        $ git push
        $ git push --tags

    This will trigger the CI to build the package and upload it to TestPyPI. In order to upload to PyPI, this can be done by publishing a new version on GitHub. This will then trigger the workflow to build and upload the package to PyPI.

#. Once the release is published, it will go into a `staging` mode on Github Actions. Admins can then approve the release (an e-mail will be sent) and it will be published on PyPI.

The Manual Approach
~~~~~~~~~~~~~~~~~~~

From the command line in your distribution, simply run the following from the clone's main dev branch:

.. code-block:: shell

    # To build the packages (sources and wheel)
    $ python -m flit build

    # To upload to PyPI
    $ python -m flit publish dist/*

The new version based off of the version checked out will now be available via pip ($ pip install clisops).

Releasing on conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~

Initial Release
^^^^^^^^^^^^^^^

In order to prepare an initial release on conda-forge, we *strongly* suggest consulting the following links:

 * https://conda-forge.org/docs/maintainer/adding_pkgs.html
 * https://github.com/conda-forge/staged-recipes

Subsequent releases
^^^^^^^^^^^^^^^^^^^

If the conda-forge feedstock recipe is built from PyPI, then when a new release is published on PyPI, `regro-cf-autotick-bot` will open Pull Requests automatically on the conda-forge feedstock.
It is up to the conda-forge feedstock maintainers to verify that the package is building properly before merging the Pull Request to the main branch.

Before updating the main conda-forge recipe, we *strongly* suggest performing the following checks:
 * Ensure that dependencies and dependency versions correspond with those of the tagged version, with open or pinned versions for the `host` requirements.
 * If possible, configure tests within the conda-forge build CI (e.g. `imports: clisops`, `commands: pytest clisops`)

.. _`GitHub Repository`: https://github.com/roocs/clisops
.. _`PEP8`: https://www.python.org/dev/peps/pep-0008/
.. _`loguru`: https://loguru.readthedocs.io/en/stable/index.html
.. _`numpydoc`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`reStructuredText (ReST)`: https://www.jetbrains.com/help/pycharm/using-docstrings-to-specify-types.html
