"""CLISOPS - Climate simulation operations."""

import os
import warnings

from loguru import logger

from clisops.config import get_config
from clisops.utils.common import enable_logging


def showwarning(message, *args, **kwargs):
    """Inject warnings from `warnings.warn` into `loguru`."""
    logger.warning(message)
    showwarning_(message, *args, **kwargs)


showwarning_ = warnings.showwarning
warnings.showwarning = showwarning

# Disable logging for clisops and remove the logger that is instantiated on import
logger.disable("clisops")
logger.remove()


# Workaround to prevent reimporting
class Package:
    __file__ = __file__  # noqa


package = Package()
CONFIG = get_config(package)

try:
    # Set the memory limit for each dask chunk
    chunk_memory_limit = CONFIG["clisops:read"].get("chunk_memory_limit", None)
except KeyError:
    logger.warning(
        "No chunk_memory_limit set in configuration file. Defaulting to None."
    )
    chunk_memory_limit = None

from clisops.parameter import *
from clisops.utils import *
from clisops.xarray_utils import *

# if get_chunk_mem_limit():
#     dask.config.set({"array.chunk-size": get_chunk_mem_limit()})

for key, value in CONFIG["environment"].items():
    os.environ[key.upper()] = value
