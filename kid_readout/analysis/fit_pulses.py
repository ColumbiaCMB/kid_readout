import numpy as np
import lmfit
from matplotlib import pyplot as plt

from kid_readout.analysis import fitter


def square_wave(num_points, period, low_level, high_level, phase, dtype='float'):
    index = np.arange(num_points)
    high = ((index - phase) % period) >= period / 2.
    out = np.ones((num_points,), dtype=dtype)
    out[high] *= high_level
    out[~high] *= low_level
    return out


def square_wave_model(params, x):
    num_points = x.shape[0]
    dtype = x.dtype
    period = params['period'].value
    low_level = params['low_level'].value
    high_level = params['high_level'].value
    phase = params['phase'].value
    return square_wave(num_points, period, low_level, high_level, phase, dtype=dtype)


def square_wave_guess(x, y, period):
    params = lmfit.Parameters()
    params.add('period', value=period, vary=False)
    median_y = np.median(y)
    min_y = y.min()
    max_y = y.max()
    ptp_y = max_y - min_y
    params.add('low_level', value=min_y + ptp_y / 4, min=y.min(), max=median_y)
    params.add('high_level', value=max_y - ptp_y / 4, min=median_y, max=y.max())
    params.add('phase', value=period / 2., min=0, max=period)
    return params


def fit_complex_square_wave(y, period=125):
    x = np.arange(len(y))

    def fixed_period_guess(x_, y_):
        return square_wave_guess(x_, y_, period)

    real_fitter = fitter.Fitter(x_data=x, y_data=y.real, model=square_wave_model, guess=fixed_period_guess,
                                method='anneal')
    imag_fitter = fitter.Fitter(x_data=x, y_data=y.imag, model=square_wave_model, guess=fixed_period_guess,
                                method='anneal')
    return real_fitter, imag_fitter


def find_rising_edge(x):
    x = np.abs(x)
    y = np.convolve(x,np.ones((10,))/10.0,mode='valid')
    y = y[:x[10:-10].shape[0]]
    x[10:-10] = y
    mid = (x.max() + x.min()) / 2.
    above = (x > mid)
    nonz = np.flatnonzero((above[1:] & ~above[:-1]))
    if len(nonz)==0:
        return np.array([0])
    else:
        return nonz


def wrap_period(val, period):
    wrapped = np.round(val).astype('int') % period
    wrapped[wrapped == period] = 0
    return wrapped



def find_high_low(x, use_fraction=0.25, debug=False):
    period = len(x)
    rising_edges = find_rising_edge(x)
    for rising_edge in rising_edges:
        falling_edge = (rising_edge + period / 2)

        high_start = rising_edge + (period * 0.5) * (1 - use_fraction) / 2.0
        high_end = high_start + period * 0.5 * use_fraction

        low_start = falling_edge + (period * 0.5) * (1 - use_fraction) / 2.0
        low_end = low_start + period * 0.5 * use_fraction

        raw_indexes = np.arange(period * 2)
        high_indexes = wrap_period(raw_indexes[(raw_indexes > high_start) & (raw_indexes < high_end)], period)
        low_indexes = wrap_period(raw_indexes[(raw_indexes > low_start) & (raw_indexes < low_end)], period)
        high = np.median(x[high_indexes].real) + 1j * np.median(x[high_indexes].imag)
        low = np.median(x[low_indexes].real) + 1j * np.median(x[low_indexes].imag)
        if np.abs(high)/np.abs(low) > 1.:
            break
        else:
            if debug:
                print ("at rising edge candidate:",rising_edge,
                       "found high",abs(high),"low",abs(low),"ratio",abs(high)/abs(low))



    if debug:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3)
        t = np.arange(period)
        ax1.plot(t, x.real, 'b')
        ax1.plot(t[high_indexes], x.real[high_indexes], 'r.')
        ax1.plot(t[low_indexes], x.real[low_indexes], 'k.')
        ax1.axhline(np.real(high), linewidth=2, color='y')
        ax1.axhline(np.real(low), linewidth=2, color='y')
        for rising_edge in rising_edges:
            ax1.axvline(rising_edge)

        ax2.plot(t, x.imag, 'b')
        ax2.plot(t[high_indexes], x.imag[high_indexes], 'r.')
        ax2.plot(t[low_indexes], x.imag[low_indexes], 'k.')
        ax2.axhline(np.imag(high), linewidth=2, color='y')
        ax2.axhline(np.imag(low), linewidth=2, color='y')
        for rising_edge in rising_edges:
            ax2.axvline(rising_edge)
        y = np.convolve(x,np.ones((10,))/10.0,mode='valid')
        y = y[:x[10:-10].shape[0]]
        x[10:-10] = y

        ax3.plot(t,np.abs(x))
        for rising_edge in rising_edges:
            ax3.axvline(rising_edge)


    return high, low, rising_edge
    