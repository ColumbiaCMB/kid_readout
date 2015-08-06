"""
This module is for plotting functions related to the physical layout of a detector wafer.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from kid_readout.analysis.resources import skip5x4


def scatter(values, mask=None, colormap=plt.cm.copper, size=40, colorbar=True, **kwargs):
    x = skip5x4.coordinate_array[:, 0]
    y = skip5x4.coordinate_array[:, 1]
    if mask is None:
        mask = np.ones_like(values, dtype=np.bool)
    fig, ax = plt.subplots()
    ax.scatter(x[mask], y[mask], c=values, s=size, cmap=colormap, edgecolor='None', **kwargs)
    if colorbar:
        mappable = plt.cm.ScalarMappable(matplotlib.colors.Normalize(min(values), max(values)), cmap=plt.cm.copper)
        mappable.set_array(values)
        cb = fig.colorbar(mappable)
    return fig, ax


def scatter_text(values, x_offset=-0.5, y_offset=0.15, **kwargs):
    fig, ax = scatter(values, **kwargs)
    for n in range(20):
        ax.text(skip5x4.coordinate_array[:, 0][n] + x_offset,
                skip5x4.coordinate_array[:, 1][n] + y_offset,
                '{:.0f}: {:.2f}'.format(n, skip5x4.nominal_frequencies[n]),
                fontdict={'fontsize': 'xx-small'})
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    return fig, ax
