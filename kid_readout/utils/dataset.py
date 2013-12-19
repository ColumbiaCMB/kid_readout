from __future__ import division
import numpy as np
import netCDF4
from collections import OrderedDict
from kid_readout.analysis.resonator import Resonator

class Dataset(object):
    
    def __init__(self, filename):
        self.ds = netCDF4.Dataset(filename)

    def sweep(self, sweep_name):
        """
        Return the netCDF4.Group object corresponding to sweep_name.
        """
        match = [group for name, group in self.ds.groups['sweeps'].groups.items()
                 if sweep_name in name]
        if not match:
            raise ValueError("No sweep names contain {0}".format(sweep_name))
        elif len(match) > 1:
            raise ValueError("Multiple sweep names contain {0}".format(sweep_name))
        else:
            return match.pop()

    def sweep_names(self):
        return self.ds.groups['sweeps'].groups.keys()


class CoarseSweep(object):
    """
    This interface is going to change soon -- it will figure out
    everything from the netCDF group, like the FineSweep.
    """
    def __init__(self, f, s21, coarse_resolution, n_subsweeps):
        self.f = f
        self.s21 = s21
        self.coarse_resolution = coarse_resolution
        self.n_subsweeps = n_subsweeps
        self.resolution = self.coarse_resolution / self.n_subsweeps

    def nth_subsweep(self, n):
        f_int = np.round((self.f-self.f[0])/self.resolution)
        mask = (f_int % self.n_subsweeps == n)
        return mask

    def subsweep_derivative_rms(self):
        return np.array([np.std(np.diff(np.abs(self.s21[self.nth_subsweep(n)])) /
                                np.diff(self.f[self.nth_subsweep(n)]))
                         for n in range(self.n_subsweeps)])
        

class FineSweep(OrderedDict):
    """
    A container class for a fine sweep.

    Parameters:
    group -- a netCDF Group that contains fine sweep data.

    Keywords:
    fit_subsweeps -- if True, the class will attempt to fit the data
    correponding to each index using the current resonator defaults.
    scale_frequency -- frequency values from the data are multiplied
    by this number; the default of 1e6 assumes that the frequency is
    stored in MHz and converts it to Hz.
    """
    
    def __init__(self, group, fit_subsweeps=False, scale_frequency=1e6):
        super(FineSweep, self).__init__()
        self.group = group
        self.f = scale_frequency * group.variables['frequency'][:]
        self.s21 = group.variables['s21'][:].view('complex128')
        # This allows the class to load old data without the subsweep
        # index. Use build_index() instead.
        try:
            self.index = group.variables['index'][:]
        except KeyError:
            pass
        if fit_subsweeps:
            self.fit_subsweeps()

    def fit_subsweeps(self, deglitch=True, threshold=2):
        """
        Attempt to fit the data corresponding to each value of the
        index by instantiating a Resonator object with default
        settings. If *deglitch* is true then points identified as
        glitches with be masked.
        """
        for index in np.unique(self.index):
            mask = self.index == index
            f = self.f[mask]
            s21 = self.s21[mask]
            if deglitch:
                good = ~self.glitch(s21, threshold)
            else:
                good = np.ones_like(s21).astype('bool')
            self[index] = Resonator(f, s21, mask=good)

    # Replace this with a method that uses gaps in the frequencies.
    def build_index(self, f_0, span):
        """
        Build the index for old data that doesn't have an index. This
        works well if the subsweeps don't overlap, but fails if they
        do. Data outside all ranges f_0 +/- span will fall into index
        -1, which is a sign that the frequencies or span should be
        adjusted.
        """
        self.index = -np.ones_like(self.f, dtype=np.int)
        for n, f in enumerate(f_0):
            self.index[np.abs(f - self.f) <= span] = n

    def glitch(self, data, threshold):
        """
        Return a boolean array with the same size as the data that is
        True where the absolute value of the second derivative is at
        least *threshold* standard deviations above the mean.
        """
        # This is a poor man's second derivative.
        k = np.array([1, -2, 1])
        x = np.convolve(k, abs(data), mode='same')
        # This supuriously flags the (k.size - )/2 points at each end
        # due to zero padding; it also flags one point on either side
        # of a single-point glitch. 
        glitch = abs(x) > threshold * np.std(x)
        glitch[:(k.size - 1) / 2] = glitch[-(k.size - 1) / 2:] = False
        return glitch
