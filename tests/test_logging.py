import sys

from loguru import logger

from clisops.utils.common import _logging_examples, enable_logging  # noqa


def test_logging_configuration(caplog):
    # no need for logger.remove() because of caplog workaround
    logger.enable("clisops")
    caplog.set_level("WARNING", logger="clisops")

    _logging_examples()  # noqa

    assert ("clisops.utils.common", 10, "1") not in caplog.record_tuples
    assert ("clisops.utils.common", 40, "4") in caplog.record_tuples
    logger.disable("clisops")


def test_disabled_enabled_logging(capsys):
    logger.remove()

    # CLISOPS disabled by default
    logger.add(sys.stderr, level="WARNING")
    logger.add(sys.stdout, level="INFO")

    _logging_examples()  # noqa

    captured = capsys.readouterr()
    assert "WARNING" not in captured.err
    assert "INFO" not in captured.out

    # re-enable CLISOPS logging
    logger.enable("clisops")
    logger.add(sys.stderr, level="WARNING")
    logger.add(sys.stdout, level="INFO")

    _logging_examples()  # noqa

    captured = capsys.readouterr()
    assert "INFO" not in captured.err
    assert "WARNING" in captured.err
    assert "INFO" in captured.out
    logger.disable("clisops")


def test_logging_enabler(capsys):
    logger.remove()

    # CLISOPS disabled by default
    logger.add(sys.stderr, level="WARNING")
    logger.add(sys.stdout, level="INFO")

    _logging_examples()  # noqa

    captured = capsys.readouterr()
    assert "WARNING" not in captured.err
    assert "INFO" not in captured.out

    enable_logging()

    _logging_examples()  # noqa

    captured = capsys.readouterr()
    print(captured.out)
    assert "INFO" not in captured.err
    assert "WARNING" in captured.err
    assert "INFO" in captured.out

    logger.disable("clisops")
    logger.remove()  # sets logging back to default
