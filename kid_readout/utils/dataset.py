from __future__ import division

import numpy as np
import netCDF4

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


class Sweep(object):
    pass


class Coarse(Sweep):
    """
    This interface is going to change soon -- it will figure out
    everything from the netCDF group, like the Fine sweep.
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
        

class Fine(Sweep):
    
    def __init__(self, group, fit_subsweeps=False):
        self.group = group
        self.f = 1e6 * np.copy(group.variables['frequency'])
        self.s21 = np.copy(group.variables['s21']).view('complex128')
        self.index = np.copy(group.variables['index'])
        self.resonators = {}
        if fit_subsweeps:
            self.fit_subsweeps()

    def __getitem__(self, key):
        return self.subsweeps[key]

    def fit_subsweeps(self, deglitch=True, threshold=2):
        for index in np.unique(self.index):
            mask = self.index == index
            f = self.f[mask]
            s21 = self.s21[mask]
            if deglitch:
                good = ~self.glitch(s21, threshold)
            else:
                good = np.ones_like(s21).astype('bool')
            self.resonators[index] = Resonator(f, s21, mask=good)

    def glitch(self, data, threshold):
        # This is a poor man's second derivative.
        k = np.array([1, -2, 1])
        x = np.convolve(k, abs(data), mode='same')
        # This supuriously flags the (k.size - )/2 points at each end
        # due to zero padding; it also flags one point on either side
        # of a single-point glitch.
        glitch = abs(x) > threshold * np.std(x)
        glitch[:(k.size - 1) / 2] = glitch[-(k.size - 1) / 2:] = False
        return glitch
