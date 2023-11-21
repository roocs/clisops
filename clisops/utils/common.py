import functools
import os
import sys
from pathlib import Path
from types import FunctionType, ModuleType
from typing import List, Optional, Union

from loguru import logger

# from roocs_utils.parameter import parameterise


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
):
    """Ensure that module is installed before function/method is called, decorator."""

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        if module is None:
            raise Exception(
                f"Package {module_name} >= {min_version} is required to use {func}."
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


def enable_logging() -> List[int]:
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
        String containing the comma separated 5 first and last elements of the list, with "..." inbetween.
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
