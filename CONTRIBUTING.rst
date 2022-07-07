
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
^^^^^^^^^^^

Report bugs at https://github.com/roocs/clisops/issues.

If you are reporting a bug, please include:


* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
^^^^^^^^

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
^^^^^^^^^^^^^^^^^^

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
^^^^^^^^^^^^^^^^^^^

clisops could always use more documentation, whether as part of the
official clisops docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
^^^^^^^^^^^^^^^

The best way to send feedback is to file an issue at https://github.com/roocs/clisops/issues.

If you are proposing a feature:


* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
^^^^^^^^^^^^

Ready to contribute? Here's how to set up ``clisops`` for local development.


#. Fork the ``clisops`` repo on GitHub.
#.
    Clone your fork locally:

    $ git clone git@github.com:your_name_here/clisops.git

#.
    Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

    $ mkvirtualenv clisops
    $ cd clisops/
    $ python setup.py develop

#.
    Create a branch for local development:

    $ git checkout -b name-of-your-bugfix-or-feature

    Now you can make your changes locally.

#.
    When you are done making changes, check that your changes pass flake8 and the
    tests, including testing other Python versions with tox:

    $ flake8 clisops tests
    $ black --target-version py38 clisops tests
    $ python setup.py test  # (or pytest)
    $ tox

    To get flake8, black, and tox, just pip install them into your virtualenv.
    Alternatively, you can use `pre-commit` to perform these checks at the git commit stage:

    $ pip install pre-commit
    $ pre-commit install
    $ pre-commit run --all-files

#.
    Commit your changes and push your branch to GitHub:

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

#.
    Submit a pull request through the GitHub website.


Logging
-------

``clisops`` uses the `loguru <https://loguru.readthedocs.io/en/stable/index.html>`_ library as its primary logging engine. In order to integrate this kind of logging in processes, we can use their logger:

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

Before you submit a pull request, check that it meets these guidelines:


#. The pull request should include tests.
#. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
#. The pull request should work for Python 3.8, 3.9, and 3.10. Check
   https://github.com/roocs/clisops/actions
   and make sure that the tests pass for all supported Python versions.
