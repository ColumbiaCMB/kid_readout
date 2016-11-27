"""
This module contains functions for binning spectral densities.
"""
from __future__ import division

import numpy as np


def log_bin_edges(frequency, bins_per_decade=30, ensure_none_empty=True):
    """
    Return an array of frequency bin edges that form bins with widths that increase approximately exponentially.

    The lowest frequency must be positive and the frequencies are assumed to have constant spacing, though the code may
    not fail if the spacing is irregular. The lowest bin edge will be positive and lower than the lowest input, and the
    highest bin edge will be higher than the highest input.

    Parameters
    ----------
    frequency : numpy.ndarray(float)
        The array of positive frequencies to use to create the bin edges.
    bins_per_decade : int
        The number of histogram bins per decade of frequency.
    ensure_none_empty : bool
        If True, then at low frequencies where the bins would be so small that no input frequencies would land in them,
        return bins with the same width as the input to ensure that no bins are empty.

    Returns
    -------
    numpy.ndarray (float)
        The edges of the frequency bins; there will be one more bin edge than frequency.

    """
    if frequency[0] == 0:
        frequency = frequency[1:]
    df = frequency[1] - frequency[0]
    log_min_edge = np.log10(frequency.min() - df / 2)
    log_max_edge = np.log10(frequency.max() + df / 2)
    num_bins = int(bins_per_decade * (log_max_edge - log_min_edge))
    log_bins = np.logspace(log_min_edge, log_max_edge, num_bins)
    if ensure_none_empty:
        usable_log_bins = log_bins[np.sum(np.diff(log_bins) < df):]
        bins = np.concatenate((frequency[frequency < usable_log_bins.min()] - df / 2, usable_log_bins))
    else:
        bins = log_bins
    return bins


# These are the left bin edges: they stop before the highest frequency.
def make_freq_bins(fr):
    """
    Legacy function, now calls log_bin_edges for unity
    Parameters
    ----------
    fr

    Returns
    -------

    """
    return log_bin_edges(fr)[:-1]
    # scale = 2
    # fmax = fr.max()
    # fout = []
    # fdiff = fr[1] - fr[0]
    # fstart = int(fdiff * 100)
    # if fstart > 10:
    #     fstart = 10
    # fstep = 1
    # if fstart < 1:
    #     fstart = 1
    #     fstep = fdiff * 2
    # #print fstart
    # fout.append(fr[fr < fstart])
    # ftop = scale * fstart
    # # fstep = int(10**int(np.round(np.log10(fstart))-1))
    # #    if fstep < 1:
    # #        fstep = 1
    # if fstep < fdiff:
    #     fstep = fdiff
    # while True:
    #     #        print ftop/10,fmax,fstep
    #     if ftop > fmax:
    #         fout.append(np.arange(ftop / scale, fmax, fstep))
    #         break
    #     else:
    #         fout.append(np.arange(ftop / scale, ftop, fstep))
    #     ftop *= scale
    #     fstep *= scale
    # return np.concatenate(fout)


def log_bin(freqs, data):
    # Need to work out
    freq_bins = make_freq_bins(freqs)
    bin_idxs = np.digitize(freqs, freq_bins)
    if type(data) is list:
        binned_data = []
        for dunit in data:
            binned_data.append(np.array([dunit[bin_idxs == k].mean() for k in range(1, len(freq_bins))]))
    else:
        binned_data = np.array([data[bin_idxs == k].mean() for k in
                                range(1, len(freq_bins))])  # skip the zeroth bin since it has nothing in it
    binned_freqs = np.array([freqs[bin_idxs == k].mean() for k in range(1, len(freq_bins))])
    return binned_freqs, binned_data


def log_bin_with_errors(frequency, data, variance, bins_per_decade=30):
    """
    Propagate errors assuming that the errors in each bin can be added in quadrature.

    :param frequency: The frequency array, assumed to be equally spaced starting from a nonzero frequency.
    :param data: The data array.
    :param variance: The variance of each point in data.
    :return:
    """
    bin_edges = log_bin_edges(frequency,bins_per_decade=bins_per_decade)
    bin_indices = np.digitize(frequency, bin_edges)
    binned_frequency = []
    binned_data = []
    bin_counts = []
    binned_variance = []
    for k in range(1, len(bin_edges)):
        binned_frequency.append(frequency[bin_indices == k].mean())
        binned_data.append(data[bin_indices == k].mean())
        bin_counts.append(np.sum(bin_indices == k))
        binned_variance.append(np.sum(variance[bin_indices == k]) / bin_counts[-1]**2)
    return np.array(binned_frequency), np.array(binned_data), np.array(bin_counts), np.array(binned_variance)
