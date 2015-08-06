from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from kid_readout.analysis import dataframe

def plot_sweeps(df, k, select_key='channel', color_key=None, norm=matplotlib.colors.LogNorm,
                colormap=plt.cm.coolwarm, legend=False):
    fig, ax = plt.subplots()
    r = df[df[select_key]==k]
    if color_key is not None:
        mappable = plt.cm.ScalarMappable(norm(min(df[color_key]), max(df[color_key])), cmap=colormap)
        mappable.set_array(df[color_key])
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            if color_key is None:
                color = 'black'
            else:
                color = mappable.to_rgba(row[color_key])
            ax.plot(row.sweep_freqs_MHz, 20*np.log10(abs(row.sweep_normalized_s21)), label=str(atten),
                    color=color)
    ax.set_xlabel('frequency [MHz]')
    ax.set_ylabel('$S_{21}$ [dB]')
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(select_key, k))
    return fig, ax


def plot_sweeps_IQ(df, k, select_key='channel', color_key=None, norm=matplotlib.colors.LogNorm,
                   colormap=plt.cm.coolwarm, legend=False):
    fig, ax = plt.subplots()
    r = df[df[select_key]==k]
    if color_key is not None:
        mappable = plt.cm.ScalarMappable(norm(min(df[color_key]), max(df[color_key])), cmap=colormap)
        mappable.set_array(df[color_key])
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            if color_key is None:
                color = 'black'
            else:
                color = mappable.to_rgba(row[color_key])
            ax.plot(row.sweep_normalized_s21.real, row.sweep_normalized_s21.imag, label=str(atten),
                    color=color)
    ax.set_xlabel('Re $S_{21}$')
    ax.set_ylabel('Im $S_{21}$')
    ax.set_xlim(0, 1.8)
    ax.set_ylim(-0.6, 0.6)
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(select_key, k))
    return fig, ax


def plot_pca_noise(df, k, select_key='channel', color_key=None, norm=matplotlib.colors.LogNorm,
                   colormap=plt.cm.coolwarm, legend=False, plot_row_1=True, plot_row_0=True):
    fig, ax = plt.subplots()
    r = df[(df[select_key]==k)]
    if color_key is not None:
        mappable = plt.cm.ScalarMappable(norm(min(df[color_key]), max(df[color_key])), cmap=colormap)
        mappable.set_array(df[color_key])
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            if color_key is None:
                color = 'black'
            else:
                color = mappable.to_rgba(row[color_key])
            if plot_row_1:
                ax.loglog(row.pca_freq, row.pca_eigvals[1], label=str(atten), color=color)
            if plot_row_0:
                ax.loglog(row.pca_freq, row.pca_eigvals[0], color=color)
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(select_key, k))
    return fig, ax


def plot_response(df, value, nrows=4, ncols=4, figsize=(6, 6), loglog=False, fits=True,
                      scale=1e6, limits=None, label=None,
                      X_scale=1e6, X_limits=None, X_label='$10^6 X$',
                      I_scale=1e6, I_limits=None, I_label='$10^6 Q_i^{-1}$',
                      X_color='blue', I_color='green'):
    if label is None:
        label = value.replace('_', ' ')
    if fits:
        masked = df[~df['{}_XI_fit_redchi'.format(value)].isnull()]
    else:
        masked = df[~df['{}_X'.format(value)].isnull()]
    fig, X_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # This ensures that X_axes is always a numpy array of AxesSubplot objects.
    if nrows == ncols == 1:
        X_axes = np.array([X_axes])
    I_axes = np.array([X_ax.twinx() for X_ax in X_axes.flatten()]).reshape(X_axes.shape)

    if loglog:
        minimum = 10**(np.floor(np.log10(masked[value].min())) - 1)
        maximum = 10**(np.ceil(np.log10(masked[value].max())) + 1)
        # The minimum value of X is always 0 so we can't display it on the plot, but we can show the top of the errorbar.
        X_minimum = 10**np.floor(np.log10(masked['{}_X_err'.format(value)].min()))
        X_maximum = 10**np.ceil(np.log10(masked['{}_X'.format(value)].max()))
        I_minimum = 10**np.floor(np.log10(masked['{}_I'.format(value)].min()))
        I_maximum = 10**np.ceil(np.log10(masked['{}_I'.format(value)].max()))
        if limits is None:
            limits = (scale * minimum, scale * maximum)
        if X_limits is None:
            X_limits = (X_scale * X_minimum, X_scale * X_maximum)
        if I_limits is None:
            I_limits = (I_scale * I_minimum, I_scale * I_maximum)
        fit_values = np.logspace(np.log10(minimum), np.log10(maximum), 1e3)
    else:
        minimum = -1 / scale
        maximum = np.ceil(scale * 1.1 * masked[value].max()) / scale
        if limits is None:
            limits = (scale * minimum, scale * maximum)
        if X_limits is None:
            X_limits = (-10, np.ceil(X_scale * masked['{}_X'.format(value)].max() / 100) * 100)
        if I_limits is None:
            I_limits = (-1, np.ceil(I_scale * masked['{}_I'.format(value)].max() / 10) * 10)
        fit_values = np.linspace(0, maximum, 1e3)

    for plot_index, (X_ax, I_ax, (channel, group)) in enumerate(zip(X_axes.flatten(), I_axes.flatten(),
                                                                    masked.groupby('channel'))):
        X_ax.errorbar(scale * group[value], X_scale * group['{}_X'.format(value)],
                      yerr=X_scale * np.array(group['{}_X_err'.format(value)]),
                      marker='.', color=X_color, linestyle='None')
        I_ax.errorbar(scale * group[value], I_scale * group['{}_I'.format(value)],
                      yerr=I_scale * np.array(group['{}_I_err'.format(value)]),
                      marker='.', color=I_color, linestyle='None')

        if fits:
            P_0 = group['{}_XI_fit_P_0'.format(value)].iloc[0]
            P_star = group['{}_XI_fit_P_star'.format(value)].iloc[0]
            X_0 = group['{}_XI_fit_X_0'.format(value)].iloc[0]
            I_0 = group['{}_XI_fit_I_0'.format(value)].iloc[0]
            I_C = group['{}_XI_fit_I_C'.format(value)].iloc[0]
            X_ax.plot(scale * fit_values, X_scale * dataframe.X(fit_values, P_0, P_star, X_0), color=X_color)
            I_ax.plot(scale * fit_values, I_scale * dataframe.I(fit_values, P_0, P_star, I_0, I_C), color=I_color)

        if loglog:
            X_ax.set_xscale('log')
            X_ax.set_yscale('log', nonposy='clip')
            I_ax.set_xscale('log')
            I_ax.set_yscale('log', nonposy='clip')
            X_ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))

        X_ax.set_xlim(*limits)
        X_ax.set_ylim(*X_limits)
        I_ax.set_ylim(*I_limits)

        X_ax.tick_params(axis='y', labelcolor=X_color)
        # Put tick labels only on the left edge.
        if plot_index % ncols != 0:
            X_ax.set_yticklabels([])
        I_ax.tick_params(axis='y', labelcolor=I_color)
        # Put ticks only on the right edge.
        if plot_index % ncols != ncols - 1:
            I_ax.set_yticklabels([])

        # Put ticks only on the bottom edge.
        if plot_index < (nrows - 1) * ncols:
            X_ax.set_xticklabels([])
        # Add axis labels.
        if plot_index == (nrows - 1) * ncols:
            X_ax.set_xlabel(label)
            X_ax.set_ylabel(X_label, color=X_color)
        if plot_index == (nrows * ncols - 1):
            I_ax.set_ylabel(I_label, color=I_color)

        try:
            X_ax.set_title('{}: {}'.format(group.channel.iloc[0], group.location.iloc[0]))
        except AttributeError:
            X_ax.set_title('{}'.format(group.channel.iloc[0]))

    return fig, X_axes, I_axes