"""Common utility functions for CLISOPS."""

import functools
import os
import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import FunctionType, ModuleType

from dask.utils import byte_sizes
from loguru import logger
from packaging.version import Version

__all__ = [
    "check_dir",
    "enable_logging",
    "expand_wildcards",
    "parse_size",
    "require_module",
]


def parse_size(size: str) -> int:
    """
    Parse size string into number of bytes.

    Parses a size string with a number and a unit (e.g., "10MB", "2GB")
    into an integer representing the number of bytes.

    Parameters
    ----------
    size : str
        The size string to parse, which should consist of a number followed by a unit (e.g., "10MB", "2GB").

    Returns
    -------
    int
        The size in bytes as an integer.

    Raises
    ------
    ValueError
        If the size string does not match the expected format or if the unit is not recognized.

    Examples
    --------
    >>> parse_size("10MB")
    10485760
    """
    n, suffix = re.match(r"^(\d+\.?\d*)([a-zA-Z]+)$", size).groups()

    try:
        multiplier = byte_sizes[suffix.lower()]

        size_in_bytes = multiplier * float(n)
    except KeyError as err:
        raise ValueError(f"Could not interpret '{suffix}' as a byte unit") from err

    return size_in_bytes


def expand_wildcards(paths: str | Path) -> list:
    """
    Expand the wildcards that may be present in Paths.

    Parameters
    ----------
    paths : str or Path
        The path or paths to expand, which may contain wildcards (e.g., `*.nc`).

    Returns
    -------
    list
        A list of Path objects that match the expanded wildcards.
    """
    path = Path(paths).expanduser()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    return [f for f in Path(path.root).glob(str(Path("").joinpath(*parts)))]


def require_module(
    func: FunctionType,
    module: ModuleType,
    module_name: str,
    min_version: str | None = "0.0.0",
    unsupported_version_range: list | tuple[str, str] | None = None,
    max_supported_version: str | None = None,
    max_supported_warning: str | None = None,
) -> Callable:
    """
    Ensure that module is installed before function/method is called, decorator.

    Parameters
    ----------
    func : FunctionType
        The function to be decorated.
    module : ModuleType
        The module to check for availability.
    module_name : str
        The name of the module to check.
    min_version : str, optional
        The minimum version of the module required. Defaults to "0.0.0".
    unsupported_version_range : list of str or tuple of str, optional
        A list with two elements, with the elements marking a range of unsupported versions,
        with the first element being the first unsupported and the second element being
        the first supported version.
        If provided, a warning will be issued if the module version falls within this range:
        version_0 <= module_version < version_1
        Defaults to None, meaning no unsupported version range check is performed.
    max_supported_version : str, optional
        The maximum supported version of the module.
        If provided, a warning will be issued if the module version exceeds this.
        Defaults to None, meaning no maximum version check is performed.
    max_supported_warning : str, optional
        The warning message to display if the module version exceeds the maximum supported version.

    Returns
    -------
    FunctionType
        The decorated function that checks for the module's availability and version before execution.
    """

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):  # numpydoc ignore=GL08
        exception_msg = f"Package {module_name} >= {min_version} is required to use {func}."
        if module is None:
            raise ModuleNotFoundError(exception_msg)
        if Version(module.__version__) < Version(min_version):
            raise ImportError(exception_msg)

        if max_supported_version is not None:
            if Version(module.__version__) > Version(max_supported_version):
                if max_supported_warning is not None:
                    warnings.warn(max_supported_warning)
                else:
                    warnings.warn(
                        f"Package {module_name} version {module.__version__} "
                        f"is greater than the suggested version {max_supported_version}."
                    )

        if unsupported_version_range is not None:
            if not isinstance(unsupported_version_range, list | tuple) or not len(unsupported_version_range) == 2:
                raise ValueError(
                    "The unsupported_version_range argument must be a list or tuple with two elements of type str, "
                    "with the elements being the minimum and maximum versions of an unsupported version range."
                )
            if Version(module.__version__) >= Version(unsupported_version_range[0]) and Version(
                module.__version__
            ) < Version(unsupported_version_range[1]):
                warnings.warn(max_supported_warning)

        return func(*args, **kwargs)

    return wrapper_func


def check_dir(func: FunctionType, dr: str | Path) -> Callable:
    """
    Ensure that directory 'dr' exists before function/method is called, decorator.

    Parameters
    ----------
    func : FunctionType
        The function to be decorated.
    dr : str or Path
        The directory path to check for existence.

    Returns
    -------
    FunctionType
        The decorated function that checks for the directory's existence before execution.
    """
    if not os.path.isdir(dr):
        os.makedirs(dr)

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):  # numpydoc ignore=GL08
        return func(*args, **kwargs)

    return wrapper_func


def _logging_examples() -> None:
    """Testing module."""
    logger.trace("0")
    logger.debug("1")
    logger.info("2")
    logger.success("2.5")
    logger.warning("3")
    logger.error("4")
    logger.critical("5")


def enable_logging() -> list[int]:
    """
    Enable logging for CLISOPS.

    Returns
    -------
    list[int]
        List of enabled log levels, e.g., [10, 20, 30, 40, 50].
    """
    logger.enable("clisops")

    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC}</>"
                " <red>|</> <lvl>{level}</> <red>|</> <cyan>{name}:{function}:{line}</>"
                " <red>|</> <lvl>{message}</>",
                "level": "INFO",
            },
            {
                "sink": sys.stderr,
                "format": "<red>"
                "{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC} | {level} | {name}:{function}:{line} | {message}"
                "</>",
                "level": "WARNING",
            },
        ]
    }
    return logger.configure(**config)


def _list_ten(list1d: list) -> str:
    """
    Convert list to string of 10 list elements equally distributed to beginning and end of the list.

    Parameters
    ----------
    list1d : list
        1D list.

    Returns
    -------
    str
        String containing the comma separated 5 first and last elements of the list, with "..." in between.
        For example "1, 2, 3, 4, 5 ... , 20, 21, 22, 23, 24, 25".

    """
    if len(list1d) < 11:
        return ", ".join(str(i) for i in list1d)
    else:
        return ", ".join(str(i) for i in list1d[0:5]) + " ... " + ", ".join(str(i) for i in list1d[-5:])
