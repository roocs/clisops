"""CLISOPS - Climate simulation operations."""

import warnings

from loguru import logger

from clisops.__version__ import __version__
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

# Load configuration
CONFIG = get_config(__file__)

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

for key, value in CONFIG["environment"].items():
    os.environ[key.upper()] = value
