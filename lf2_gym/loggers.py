import logging
import sys

__all__ = [
    "get_logger",
]

formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)-8s - %(name)-12s - %(message)s",
)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger with the given name in `lf2-gym` namespace
    The default handler is TimedRotatingFileHandler and StreamHandler.
    The formatting of message is "%(asctime)s - %(levelname)-8s - %(name)-12s - %(message)s"

    Parameters
    ----------
    name: Optional[str], default= None

    Returns
    -------
    Logger object

    Examples
    --------
    This example shows how to use the `get_logger` in modules.


    """
    logger = logging.getLogger("lf2-gym")
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    if name:
        logger = logger.getChild(name)

    return logger
