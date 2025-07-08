"""Utility functions for handling time and date objects in CLISOPS."""

import re

import numpy as np


def to_isoformat(tm):
    """
    Return an ISO 8601 string from a time object (of different types).

    Parameters
    ----------
    tm : datetime.datetime or datetime.date or numpy.datetime64 or similar
        A time object that can be converted to an ISO 8601 string.

    Returns
    -------
    str
        An ISO 8601 formatted string representing the time.
    """
    if isinstance(tm, np.datetime64):
        return str(tm).split(".")[0]
    else:
        return tm.isoformat()


class AnyCalendarDateTime:
    """
    A class to represent a datetime that could be of any calendar.

    Can add and subtract a day from the input based on MAX_DAY, MIN_DAY, MAX_MONTH and MIN_MONTH

    Parameters
    ----------
    year : int
        The year of the datetime.
    month : int
        The month of the datetime (1-12).
    day : int
        The day of the month (1-31).
    hour : int
        The hour of the day (0-23).
    minute : int
        The minute of the hour (0-59).
    second : int
        The second of the minute (0-59).
    """

    MONTH_RANGE = range(1, 13)
    # 31 is the maximum number of days in any month in any of the calendars supported by cftime
    DAY_RANGE = range(1, 32)
    HOUR_RANGE = range(0, 24)
    MINUTE_RANGE = range(0, 60)
    SECOND_RANGE = range(0, 60)

    def __init__(self, year, month, day, hour, minute, second):
        """
        Initialize the AnyCalendarDateTime object with year, month, day, hour, minute, and second.

        Parameters
        ----------
        year : int
            The year of the datetime.
        month : int
            The month of the datetime (1-12).
        day : int
            The day of the month (1-31).
        hour : int
            The hour of the day (0-23).
        minute : int
            The minute of the hour (0-59).
        second : int
            The second of the minute (0-59).

        Raises
        ------
        ValueError
            If any of the input values are out of their respective ranges.
        """
        self.year = year

        self.month = month
        self.validate_input(self.month, "month", self.MONTH_RANGE)

        self.day = day
        self.validate_input(self.day, "day", self.DAY_RANGE)

        self.hour = hour
        self.validate_input(self.hour, "hour", self.HOUR_RANGE)

        self.minute = minute
        self.validate_input(self.minute, "minute", self.MINUTE_RANGE)

        self.second = second
        self.validate_input(self.second, "second", self.SECOND_RANGE)

    def validate_input(self, input, name, range):
        """
        Validate input against a given range.

        Parameters
        ----------
        input : int
            The input value to validate.
        name : str
            The name of the input for error messages.
        range : range
            The valid range for the input value.

        Raises
        ------
        ValueError
            If the input value is not within the specified range.
        """
        if input not in range:
            raise ValueError(f"Invalid input {input} for {name}. Expected value between {range[0]} and {range[-1]}.")

    def __repr__(self):
        """Print value."""
        return self.value

    @property
    def value(self):
        """
        Show calendar value.

        Returns
        -------
        str
            A string representation of the datetime in ISO 8601 format.
        """
        return f"{self.year}-{self.month:02d}-{self.day:02d}T{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

    def add_day(self):
        """Add a day to the input datetime."""
        self.day += 1

        if self.day > self.DAY_RANGE[-1]:
            self.month += 1
            self.day = 1

        if self.month > self.MONTH_RANGE[-1]:
            self.year += 1
            self.month = self.MONTH_RANGE[0]

    def sub_day(self):
        """Subtract a day to the input datetime."""
        self.day -= 1

        if self.day < self.DAY_RANGE[0]:
            self.month -= 1
            self.day = self.DAY_RANGE[-1]

        if self.month < self.MONTH_RANGE[0]:
            self.year -= 1
            self.month = self.MONTH_RANGE[-1]


def str_to_AnyCalendarDateTime(dt: str, defaults: list[int] | None = None):
    """
    Given a string representing date/time, return a DateTimeAnyTime object.

    String formats should start with Year and go through to Second, but you
    can miss out anything from month onwards.

    Parameters
    ----------
    dt : str
        A string representing a date/time in the format "YYYY-MM-DDTHH:MM:SS" or similar.
    defaults : list, optional
        A list of default values for year, month, day, hour, minute, and second
        if they cannot be parsed from the string.

    Returns
    -------
    AnyCalendarDateTime
        An instance of AnyCalendarDateTime initialized with the parsed or default values.
    """
    if not dt and not defaults:
        raise Exception("Must provide at least the year as argument, or all defaults, to create date time.")

    # Start with the most common pattern
    regex = re.compile(r"^(\d+)-(\d+)-(\d+)[T ](\d+):(\d+):(\d+)$")
    match = regex.match(dt)

    if match:
        items = match.groups()
    else:
        # Try a more complex split and build of the time string
        if not defaults:
            defaults = [-1, 1, 1, 0, 0, 0]
        else:
            if len(defaults) < 6:
                raise Exception("A default value must be provided for year, month, day, hour, minute and second.")
        components = re.split("[- T:]", dt.strip("Z"))

        # Build a list of time components
        items = components + defaults[len(components) :]

    return AnyCalendarDateTime(*[int(float(i)) for i in items])


def create_time_bounds(ds, freq) -> list[list[AnyCalendarDateTime]]:
    """
    Generate time bounds for datasets that have been temporally averaged.

    Averaging frequencies supported are yearly, monthly and daily.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the time variable.
    freq : str
        The frequency of the time bounds to create. Options are "year", "month", or "day".

    Returns
    -------
    list of AnyCalendarDateTime
        A list of lists containing the start and end datetime objects for each time point in the dataset.
    """
    # get datetime class
    dt_cls = ds.time.values[0].__class__

    if freq == "month":
        time_bounds = [
            [
                dt_cls(tm.year, tm.month, tm.day),
                dt_cls(tm.year, tm.month, tm.daysinmonth),
            ]
            for tm in ds.time.values
        ]

    elif freq == "year":
        # get number of days in december for calendar
        dec_days = dt_cls(2000, 12, 1).daysinmonth
        # generate time bounds
        time_bounds = [[dt_cls(tm.year, 1, 1), dt_cls(tm.year, 12, dec_days)] for tm in ds.time.values]

    elif freq == "day":
        time_bounds = [
            [
                dt_cls(tm.year, tm.month, tm.day, 0, 0, 0),
                dt_cls(tm.year, tm.month, tm.day, 23, 59, 59),
            ]
            for tm in ds.time.values
        ]

    else:
        raise Exception("Time frequency not supported for creation of time bounds.")

    return time_bounds
