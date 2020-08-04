#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
import os

from setuptools import find_packages, setup

__copyright__ = "Copyright 2018 United Kingdom Research and Innovation"
__license__ = "BSD"


# Populate long description setting with content of README
#
# Use markdown format read me file as GitHub will render it automatically
# on package page
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md")) as readme_file:
    _long_description = readme_file.read()

about = {}
with open(os.path.join(here, "clisops", "__version__.py"), "r") as f:
    exec(f.read(), about)


requirements = [line.strip() for line in open("requirements.txt")]

setup_requirements = [
    "pytest-runner",
]

test_requirements = ["pytest", "tox"]

docs_requirements = [
    "sphinx",
    "sphinx-rtd-theme",
    "nbsphinx",
    "pandoc",
    "ipython",
    "ipykernel",
    "jupyter_client",
    "matplotlib",
]

dev_requirements = [line.strip() for line in open("requirements_dev.txt")]

setup(
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    # See:
    # https://www.python.org/dev/peps/pep-0301/#distutils-trove-classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Security",
        "Topic :: Internet",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Systems Administration :: Authentication/Directory",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    description="clisops - climate simulation operations.",
    license=__license__,
    # This qualifier can be used to selectively exclude Python versions -
    # in this case early Python 2 and 3 releases
    python_requires=">=3.6.0",
    install_requires=requirements,
    long_description=_long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="clisops",
    name="clisops",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={"docs": docs_requirements, "dev": dev_requirements},
    url="https://github.com/roocs/clisops",
    # zip_safe=False,
)
