# -*- coding: utf-8 -*-
"""Top-level package for clisops."""
from roocs_utils.config import get_config

import clisops

from .__version__ import __author__, __email__, __version__

CONFIG = get_config(clisops)
