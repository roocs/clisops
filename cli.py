# -*- coding: utf-8 -*-
"""Console script for clisops."""

__author__ = """Elle Smith"""
__contact__ = "eleanor.smith@stfc.ac.uk"
__copyright__ = "Copyright 2018 United Kingdom Research and Innovation"
__license__ = "BSD"
import argparse
import sys


def main():
    """Console script for clisops."""
    parser = argparse.ArgumentParser()
    parser.add_argument("_", nargs="*")
    args = parser.parse_args()

    print(f"Arguments: {args._}")
    print("Replace this message by putting your code into " "clisops.cli.main")
    return


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
