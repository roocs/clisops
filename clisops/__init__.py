# -*- coding: utf-8 -*-
"""Top-level package for clisops."""

import logging
import logging.config
logging.config.fileConfig('logging.conf')

import dask

from roocs_utils.config import get_config

import clisops

from .__version__ import __author__, __email__, __version__

CONFIG = get_config(clisops)

# Set the memory limit for each dask chunk
chunk_memory_limit = CONFIG['clisops:read'].get('chunk_memory_limit', None)

if chunk_memory_limit:
    dask.config.set({'array.chunk-size': chunk_memory_limit})

