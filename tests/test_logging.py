import logging
import sys

import pytest
from loguru import logger

from clisops.utils.common import _logging_examples, enable_logging  # noqa


class TestLoggingFuncs:
    @pytest.mark.xfail(
        reason="pytest-loguru does not implement logging levels for caplog yet"
    )
    def test_logging_configuration(self, caplog):
        caplog.set_level(logging.WARNING, logger="clisops")

        _logging_examples()  # noqa

        assert ("clisops.utils.common", 10, "1") not in caplog.record_tuples
        assert ("clisops.utils.common", 40, "4") in caplog.record_tuples

    def test_disabled_enabled_logging(self, capsys):
        logger.disable("clisops")

        # CLISOPS disabled
        id1 = logger.add(sys.stderr, level="WARNING")
        id2 = logger.add(sys.stdout, level="INFO")

        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "WARNING" not in captured.err
        assert "INFO" not in captured.out

        # re-enable CLISOPS logging
        logger.enable("clisops")

        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "INFO" not in captured.err
        assert "WARNING" in captured.err
        assert "INFO" in captured.out

        logger.remove(id1)
        logger.remove(id2)

    def test_logging_enabler(self, capsys):
        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "WARNING" not in captured.err
        assert "INFO" not in captured.out

        ids = enable_logging()

        _logging_examples()  # noqa

        captured = capsys.readouterr()
        assert "INFO" not in captured.err
        assert "WARNING" in captured.err
        assert "INFO" in captured.out

        for i in ids:
            logger.remove(i)  # sets logging back to default
