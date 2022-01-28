import sys
from pathlib import Path
from typing import Union

from loguru import logger


def expand_wildcards(paths: Union[str, Path]) -> list:
    """Expand the wildcards that may be present in Paths."""
    path = Path(paths).expanduser()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    return [f for f in Path(path.root).glob(str(Path("").joinpath(*parts)))]


def _logging_examples() -> None:
    """Testing module"""
    logger.trace("0")
    logger.debug("1")
    logger.info("2")
    logger.success("2.5")
    logger.warning("3")
    logger.error("4")
    logger.critical("5")


def enable_logging():
    logger.remove()

    logger.enable("clisops")
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC}</>"
        " <red>|</> <lvl>{level}</> <red>|</> <cyan>{name}:{function}:{line}</>"
        " <red>|</> <lvl>{message}</>",
        level="INFO",
    )

    logger.add(
        sys.stderr,
        format="<red>{time:YYYY-MM-DD HH:mm:ss.SSS Z UTC} | {level} | {name}:{function}:{line} | {message}</>",
        level="WARNING",
    )
