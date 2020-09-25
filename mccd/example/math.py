# -*- coding: utf-8 -*-


def add_int(x, y):
    """Add Integers

    Add two integer values.

    Parameters
    ----------
    x : int
        First value
    y : int
        Second value

    Returns
    -------
    int
        Result of addition

    Raises
    ------
    TypeError
        For invalid input types.

    """

    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError('Inputs must be integers.')

    return x + y
