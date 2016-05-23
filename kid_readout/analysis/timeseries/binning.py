"""
This module contains functions for binning spectral densities.
"""
from __future__ import division

import numpy as np


# TODO: implement using ideas from branch imcompatible
"""
def loglike(frequency, combine_above):
    left_edges = frequency[frequency <= combine_above]
    return left_edges
"""


# These are the left bin edges: they stop before the highest frequency.
def make_freq_bins(fr):
    scale = 2
    fmax = fr.max()
    fout = []
    fdiff = fr[1] - fr[0]
    fstart = int(fdiff * 100)
    if fstart > 10:
        fstart = 10
    fstep = 1
    if fstart < 1:
        fstart = 1
        fstep = fdiff * 2
    #print fstart
    fout.append(fr[fr < fstart])
    ftop = scale * fstart
    # fstep = int(10**int(np.round(np.log10(fstart))-1))
    #    if fstep < 1:
    #        fstep = 1
    if fstep < fdiff:
        fstep = fdiff
    while True:
        #        print ftop/10,fmax,fstep
        if ftop > fmax:
            fout.append(np.arange(ftop / scale, fmax, fstep))
            break
        else:
            fout.append(np.arange(ftop / scale, ftop, fstep))
        ftop *= scale
        fstep *= scale
    return np.concatenate(fout)


# TODO: this currently throws away the data in the last bin, which has index len(freq_bins + 1)
def log_bin(freqs, data):
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


def log_bin_with_errors(f, data, variance):
    """
    Propagate errors assuming that the errors in each bin can be added in quadrature.

    :param f: The frequency array, assumed to be equally spaced and starting at 0 (?)
    :param data: The data array.
    :param variance: The variance of each point in data.
    :return:
    """
    left_bin_edges = make_freq_bins(f)
    bin_indices = np.digitize(f, left_bin_edges)
    # skip the zeroth bin since it has nothing in it
    # since make_freq_bins stops below the maximum frequency, there is data in the rightmost bin
    binned_f = []
    binned_data = []
    bin_counts = []
    binned_variance = []
    for k in range(1, len(left_bin_edges) + 1):
        binned_f.append(f[bin_indices == k].mean())
        binned_data.append(data[bin_indices == k].mean())
        bin_counts.append(np.sum(bin_indices == k))
        binned_variance.append(np.sum(variance[bin_indices == k]) / bin_counts[-1]**2)
    return np.array(binned_f), np.array(binned_data), np.array(bin_counts), np.array(binned_variance)
