"""CLISOPS - Climate simulation operations."""

######################################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018-2025, United Kingdom Research and Innovation, Elle Smith, Ag Stephens, Carsten Ehbrecht, Trevor James Smith
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
######################################################################################

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
