def create_time_bounds(ds, freq):
    """Generate time bounds for datasets that have been temporally averaged.

    Averaging frequencies supported are yearly, monthly and daily.
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
        time_bounds = [
            [dt_cls(tm.year, 1, 1), dt_cls(tm.year, 12, dec_days)]
            for tm in ds.time.values
        ]

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
