import functools
import os
import re
import sys
import warnings
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Optional, Union

from dask.utils import byte_sizes
from loguru import logger
from packaging.version import Version


def parse_size(size):
    """

    Parse size string into number of bytes.

    :param size: (str) size to parse in any unit
    :return: (int) number of bytes
    """
    n, suffix = re.match(r"^(\d+\.?\d*)([a-zA-Z]+)$", size).groups()

    try:
        multiplier = byte_sizes[suffix.lower()]

        size_in_bytes = multiplier * float(n)
    except KeyError as err:
        raise ValueError(f"Could not interpret '{suffix}' as a byte unit") from err

    return size_in_bytes


def expand_wildcards(paths: Union[str, Path]) -> list:
    """Expand the wildcards that may be present in Paths."""
    path = Path(paths).expanduser()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    return [f for f in Path(path.root).glob(str(Path("").joinpath(*parts)))]


def require_module(
    func: FunctionType,
    module: ModuleType,
    module_name: str,
    min_version: Optional[str] = "0.0.0",
    max_supported_version: Optional[str] = None,
    max_supported_warning: Optional[str] = None,
):
    """Ensure that module is installed before function/method is called, decorator."""

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        exception_msg = (
            f"Package {module_name} >= {min_version} is required to use {func}."
        )
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
        return func(*args, **kwargs)

    return wrapper_func


def check_dir(func: FunctionType, dr: Union[str, Path]):
    """Ensure that directory dr exists before function/method is called, decorator."""
    if not os.path.isdir(dr):
        os.makedirs(dr)

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
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
    logger.enable("clisops")

    config = dict(
        handlers=[
            dict(
                sink=sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC}</>"
                " <red>|</> <lvl>{level}</> <red>|</> <cyan>{name}:{function}:{line}</>"
                " <red>|</> <lvl>{message}</>",
                level="INFO",
            ),
            dict(
                sink=sys.stderr,
                format="<red>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC} | {level} | {name}:{function}:{line} | {message}</>",
                level="WARNING",
            ),
        ]
    )
    return logger.configure(**config)


def _list_ten(list1d):
    """Convert list to string of 10 list elements equally distributed to beginning and end of the list.

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
        return (
            ", ".join(str(i) for i in list1d[0:5])
            + " ... "
            + ", ".join(str(i) for i in list1d[-5:])
        )
