import sys

from clisops.utils.common import _logging_examples, enable_logging  # noqa
from clisops.utils.testing import ContextLogger


class TestLoggingFuncs:
    def test_logging_configuration(self, caplog):
        with ContextLogger(caplog):
            caplog.set_level("WARNING", logger="clisops")

            _logging_examples()  # noqa

            assert ("clisops.utils.common", 10, "1") not in caplog.record_tuples
            assert ("clisops.utils.common", 40, "4") in caplog.record_tuples

    def test_disabled_enabled_logging(self, capsys):
        with ContextLogger() as _logger:
            _logger.disable("clisops")

            # CLISOPS disabled
            _logger.add(sys.stderr, level="WARNING")
            _logger.add(sys.stdout, level="INFO")

            _logging_examples()  # noqa

            captured = capsys.readouterr()
            assert "WARNING" not in captured.err
            assert "INFO" not in captured.out

            # re-enable CLISOPS logging
            _logger.enable("clisops")

            _logging_examples()  # noqa

            captured = capsys.readouterr()
            assert "INFO" not in captured.err
            assert "WARNING" in captured.err
            assert "INFO" in captured.out

    def test_logging_enabler(self, capsys):
        with ContextLogger():
            _logging_examples()  # noqa

            captured = capsys.readouterr()
            assert "WARNING" not in captured.err
            assert "INFO" not in captured.out

            enable_logging()

            _logging_examples()  # noqa

            captured = capsys.readouterr()
            assert "INFO" not in captured.err
            assert "WARNING" in captured.err
            assert "INFO" in captured.out
