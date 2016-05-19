"""
This module contains functions to analyze periodic data.
"""


def folded_shape(array, period_samples):
    if period_samples == 0:
        raise ValueError("Cannot fold unmodulated data or with period=0")
    shape = list(array.shape)
    shape[-1] = -1
    shape.append(period_samples)
    return tuple(shape)


def fold(array, period_samples, reduce=None):
    reshaped = array.reshape(folded_shape(array, period_samples))
    if reduce is None:
        return reshaped
    else:
        return reduce(reshaped, axis=reshaped.ndim - 2)
