"""
This module contains functions to analyze periodic data.
"""
from __future__ import division, print_function

import numpy as np

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


def mask_left_right(size, skip):
    left_mask = (skip * size < np.arange(size)) & (np.arange(size) < size // 2)
    right_mask = size // 2 + skip * size < np.arange(size)
    return left_mask, right_mask


def peak_to_peak(folded, skip=0.1):
    left_mask, right_mask = mask_left_right(size=folded.size, skip=skip)
    left = folded[left_mask]
    right = folded[right_mask]
    return np.mean(left) - np.mean(right), np.sqrt(np.var(left) / left.size + np.var(right) / right.size)
