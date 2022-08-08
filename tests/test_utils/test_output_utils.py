import os
import time
from pathlib import Path

import pytest
import xarray as xr

from clisops.utils.output_utils import FileLock


def test_filelock_simple(tmp_path):
    LOCK_FILE = Path(tmp_path, "test.lock")
    DATA_FILE = Path(tmp_path, "test.dat")

    lock = FileLock(LOCK_FILE)

    lock.acquire()
    try:
        time.sleep(1)
        assert os.path.isfile(LOCK_FILE)
        assert lock.state == "LOCKED"
        open(DATA_FILE, "a").write("1")
    finally:
        lock.release()

    time.sleep(1)
    assert not os.path.isfile(LOCK_FILE)


def test_filelock_already_locked(tmp_path):
    LOCK_FILE = Path(tmp_path, "test.lock")

    lock1 = FileLock(LOCK_FILE)
    lock2 = FileLock(LOCK_FILE)

    lock1.acquire(timeout=10)

    try:
        lock2.acquire(timeout=5)
    except Exception as exc:
        assert str(exc) == f"Could not obtain file lock on {LOCK_FILE}"
