"""CLISOPS - Climate simulation operations."""

import os
import warnings

from loguru import logger
from roocs_utils.config import get_config

from .__version__ import __author__, __copyright__, __email__, __license__, __version__
from .utils.common import enable_logging


def showwarning(message, *args, **kwargs):
    """Inject warnings from `warnings.warn` into `loguru`."""
    logger.warning(message)
    showwarning_(message, *args, **kwargs)


showwarning_ = warnings.showwarning
warnings.showwarning = showwarning

# Disable logging for clisops and remove the logger that is instantiated on import
logger.disable("clisops")
logger.remove()


# Workaround for roocs_utils to not re-import clisops
class Package:
    __file__ = __file__  # noqa


package = Package()
CONFIG = get_config(package)

# Set the memory limit for each dask chunk
chunk_memory_limit = CONFIG["clisops:read"].get("chunk_memory_limit", None)

# if get_chunk_mem_limit():
#     dask.config.set({"array.chunk-size": get_chunk_mem_limit()})

for key, value in CONFIG["environment"].items():
    os.environ[key.upper()] = value
