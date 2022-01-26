import logging


def test_logging_configuration(capsys):  # noqa
    logger = logging.getLogger("clisops")
    logger.debug("1")
    logger.info("2")
    logger.warning("3")
    logger.error("4")
    logger.critical("5")
    captured = capsys.readouterr()

    assert "DEBUG" not in captured.err
