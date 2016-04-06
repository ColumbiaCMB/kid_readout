import numpy as np
from matplotlib import pyplot as plt

__author__ = 'gjones'


def complex_gaussian(shape):
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def plot_ri(data, ax=None, *args, **kwargs):
    if ax is None:
        plt.plot(data.real, data.imag, *args, **kwargs)
    else:
        ax.plot(data.real, data.imag, *args, **kwargs)
