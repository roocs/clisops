import functools
import os
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Optional, Union

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
