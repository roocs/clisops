"""Exceptions used in CLISOPS."""


class InvalidParameterValue(Exception):
    """Raised when a parameter value is invalid or cannot be processed."""


class MissingParameterValue(Exception):
    """Raised when a required parameter is missing or not provided."""


class InvalidProject(Exception):
    """Raised when the project is unknown to roocs.ini."""


class InconsistencyError(Exception):
    """Raised when there is some inconsistency which prevents files from being scanned."""


class InvalidCollection(Exception):
    """
    Raised when a collection is not valid or not available in the roocs.ini file.

    Parameters
    ----------
    message : str
        Custom error message to be displayed.
    """

    def __init__(
        self,
        message="Some or all of the requested collection are not in the list of available data.",
    ):
        self.message = message
        super().__init__(self.message)
