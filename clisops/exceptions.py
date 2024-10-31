class InvalidParameterValue(Exception):
    pass


class MissingParameterValue(Exception):
    pass


class InvalidProject(Exception):
    """Raised when the project is unknown to roocs.ini"""


class InconsistencyError(Exception):
    """Raised when there is some inconsistency which prevents files
    being scanned."""


class InvalidCollection(Exception):
    def __init__(
        self,
        message="Some or all of the requested collection are not in the list of available data.",
    ):
        self.message = message
        super().__init__(self.message)
