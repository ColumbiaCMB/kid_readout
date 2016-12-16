"""
This module contains functions for binning spectral densities.
"""
from __future__ import division

import numpy as np


def log_bin_edges(frequency, bins_per_decade, ensure_none_empty):
    """
    Return an array of frequency bin edges that form bins with widths that increase approximately exponentially.

    The frequencies should be non-negative, equally-spaced, and increasing. The code may or may not work as expected if
    the spacing is not equal. The code will work whether or not the first element is 0: in either case, the lowest bin
    edge will be half of the first positive frequency, and the highest bin edge will be slightly larger than the highest
    frequency. Thus, all positive input frequencies will land between the extreme bin edges, and zero will land outside
    the left bin edge.

    Parameters
    ----------
    frequency : ndarray(float)
        The array of frequencies to use to create the bin edges.
    bins_per_decade : int
        The number of histogram bins per decade of frequency.
    ensure_none_empty : bool
        If True, then at low frequencies where the bins would be so small that no input frequencies would land in them,
        return bins with the same width as the input to ensure that no bins are empty.

    Returns
    -------
    numpy.ndarray(float)
        The edges of the frequency bins, all positive.
    """
    if frequency[0] == 0:
        frequency = frequency[1:]
    df = frequency[0]
    log_min_edge = np.log10(frequency.min() - df / 2)
    log_max_edge = np.log10(frequency.max() + df / 2)
    num_bins = int(bins_per_decade * (log_max_edge - log_min_edge))
    log_bins = np.logspace(log_min_edge, log_max_edge, num_bins)
    if ensure_none_empty:
        log_at_least_df = log_bins[np.sum(np.diff(log_bins) <= df):]
        linear = frequency[frequency < log_at_least_df[0]] - df / 2
        # Shift all but the last bin edge to ensure all bins are at least df wide and all bins are populated.
        log_at_least_df[:-1] += df - (log_at_least_df[0] - linear[-1])
        return np.concatenate((linear, log_at_least_df))
    else:
        return log_bins


def log_bin(frequency, bins_per_decade, *data):
    """
    Return the results of binning the given data arrays in frequency bins with widths that increase approximately
    exponentially, intended for plotting on a logarithmic axis.

    At lower frequencies, the bins are equal-width. This spacing is used so that every bin contains at least one
    frequency. At higher frequencies, the bins will contain many frequencies. In the crossover region, the counts may
    not increase monotonically.

    If the zero-frequency data is included, the first element of the mean_frequency array will be 0 and the first
    elements of the binned_data arrays will equal the first elements of the corresponding input arrays. If not, the
    first element of the mean_frequency array will be positive.


    Parameters
    ----------
    frequency : ndarray(float)
        The equally-spaced, non-negative frequencies corresponding to the data arrays.
    bins_per_decade : int
        The number of histogram bins per decade of frequency.
    data : ndarrays
        The data arrays.

    Returns
    -------
    ndarray(float)
        The edges of the bins used to create the histogram; see log_bin_edges().
    ndarray(int)
        The number of counts in each bin.
    ndarray(float)
        Each element is the mean of all the frequency points that lie in the corresponding bin.
    list of ndarray
        The binned data arrays, where each array element is the mean of all data in the corresponding bin.

    Examples
    --------
    One data array:
    edges, counts, f_mean, [binned_data] = log_bin(f, 10, data)

    Unpacking multiple data arrays:
    edges, counts, f_mean, [binned_data1, binned_data2] = log_bin(f, 10, data1, data2)
    """
    edges = log_bin_edges(frequency, bins_per_decade=bins_per_decade, ensure_none_empty=True)
    bin_indices = np.digitize(frequency, edges)
    indices_used = np.unique(bin_indices)
    counts = np.array([np.sum([bin_indices == n]) for n in indices_used])
    mean_frequency = np.array([np.mean(frequency[bin_indices == n]) for n in indices_used])
    binned_data = [np.array([np.mean(d[bin_indices == n]) for n in indices_used]) for d in data]
    return edges, counts, mean_frequency, binned_data


def log_bin_with_variance(frequency, bins_per_decade, *data_and_variance):
    """
    Return the results of binning the given data arrays and variances in frequency bins with widths that increase
    approximately exponentially, intended for plotting on a logarithmic axis.

    The frequency behavior is the same as log_bin(). The variances of the means of the binned data are calculated
    assuming that the variance of the N samples in a bin is the mean of the variances divided by N.

    Parameters
    ----------
    frequency : ndarray(float)
        The equally-spaced, non-negative frequencies corresponding to the data arrays.
    bins_per_decade : int
        The number of histogram bins per decade of frequency.
    data_and_variance : (ndarray, ndarray)
        Tuples containing arrays of the data and corresponding variance.

    Returns
    -------
    ndarray(float)
        The edges of the bins used to create the histogram; see log_bin_edges().
    ndarray(int)
        The number of counts in each bin.
    ndarray(float)
        Each element is the mean of all the frequency points that lie in the corresponding bin.
    list of (ndarray, ndarray) tuples
        A list of two-element tuples containing the binned data and variance arrays.

    Examples
    --------
    Unpacking one (data, variance) pair:
    edges, counts, f_mean, [(binned_data, binned_variance)] = log_bin_with_variance(f, 10, (d, v))

    Unpacking multiple pairs:
    edges, counts, f_mean, [(bd1, bv1), (bd2, bv2)] = log_bin_with_variance(f, 10, (d1, v1), (d2, v2))
    """
    edges = log_bin_edges(frequency, bins_per_decade=bins_per_decade, ensure_none_empty=True)
    bin_indices = np.digitize(frequency, edges)
    indices_used = np.unique(bin_indices)
    counts = np.array([np.sum([bin_indices == n]) for n in indices_used])
    mean_frequency = np.array([np.mean(frequency[bin_indices == n]) for n in indices_used])
    binned_dv = []
    for d, v in data_and_variance:
        binned_dv.append((np.array([np.mean(d[bin_indices == n]) for n in indices_used]),
                          np.array([np.mean(v[bin_indices == n]) / c for n, c in zip(indices_used, counts)])))
    return edges, counts, mean_frequency, binned_dv


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
    return log_bin_edges(frequency=fr, bins_per_decade=30, ensure_none_empty=True)[:-1]
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


def log_bin_old(freqs, data):
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
