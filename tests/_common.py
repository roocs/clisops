import os

DEFAULT_CMIP5_ARCHIVE_BASE = "/badc/cmip5/data/"

def cmip5_archive_base():
    if 'CMIP5_ARCHIVE_BASE' in os.environ:
        return os.environ['CMIP5_ARCHIVE_BASE']
    return DEFAULT_CMIP5_ARCHIVE_BASE

CMIP5_ARCHIVE_BASE = cmip5_archive_base()
