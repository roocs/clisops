# -*- coding: utf-8 -*-
"""Top-level package for clisops."""
import os

from loguru import logger
from roocs_utils.config import get_config

from .__version__ import __author__, __email__, __version__

logger.disable("clisops")


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
