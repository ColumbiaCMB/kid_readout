from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from kid_readout.analysis import dataframe

def plot_sweeps(df, k, select_key='channel', color_key='zbd_power', norm=matplotlib.colors.LogNorm,
                colormap=plt.cm.coolwarm, legend=False):
    fig, ax = plt.subplots()
    r = df[df[select_key]==k]
    mappable = plt.cm.ScalarMappable(norm(min(df[color_key]), max(df[color_key])), cmap=colormap)
    mappable.set_array(df[color_key])
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            ax.plot(row.sweep_freqs_MHz, 20*np.log10(abs(row.sweep_normalized_s21)), label=str(atten),
                    color=mappable.to_rgba(row[color_key]))
    ax.set_xlabel('frequency [MHz]')
    ax.set_ylabel('$S_{21}$ [dB]')
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(select_key, k))
    return fig, ax


def plot_sweeps_IQ(df, k, select_key='channel', color_key='zbd_power', norm=matplotlib.colors.LogNorm,
                   colormap=plt.cm.coolwarm, legend=False):
    fig, ax = plt.subplots()
    r = df[df[select_key]==k]
    mappable = plt.cm.ScalarMappable(norm(min(df[color_key]), max(df[color_key])), cmap=colormap)
    mappable.set_array(df[color_key])
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            ax.plot(row.sweep_normalized_s21.real, row.sweep_normalized_s21.imag, label=str(atten),
                    color=mappable.to_rgba(row[color_key]))
    ax.set_xlabel('Re $S_{21}$')
    ax.set_ylabel('Im $S_{21}$')
    ax.set_xlim(0, 1.8)
    ax.set_ylim(-0.6, 0.6)
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(select_key, k))
    return fig, ax


def plot_pca_noise(df, k, select_key='channel', color_key='zbd_power', norm=matplotlib.colors.LogNorm,
                   colormap=plt.cm.coolwarm, legend=False, plot_row_1=True, plot_row_0=True):
    fig, ax = plt.subplots()
    r = df[(df[select_key]==k)]
    mappable = plt.cm.ScalarMappable(norm(min(df[color_key]), max(df[color_key])), cmap=colormap)
    mappable.set_array(df[color_key])
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            if plot_row_1:
                ax.loglog(row.pca_freq, row.pca_eigvals[1], label=str(atten), color=mappable.to_rgba(row[color_key]))
            if plot_row_0:
                ax.loglog(row.pca_freq, row.pca_eigvals[0], color=mappable.to_rgba(row[color_key]))
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(select_key, k))
    return fig, ax


def plot_responsivity(df, nrows=4, ncols=4, figsize=(6, 6), loglog=False, power='zbd_power', fits=True,
                      P_scale=1e6, P_limits=None, P_label='power [$\mu$W]',
                      X_scale=1e6, X_limits=None, X_label='$10^6 X$',
                      I_scale=1e6, I_limits=None, I_label='$10^6 Q_i^{-1}$',
                      X_color='blue', I_color='green'):
    masked = df[~df['{}_XI_fit_redchi'.format(power)].isnull()]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # This ensures that axes is always a numpy array of AxesSubplot objects.
    if nrows == ncols == 1:
        axes = np.array([axes])

    if loglog:
        P_minimum = 10**(np.floor(np.log10(masked[power].min())) - 1)
        P_maximum = 10**(np.ceil(np.log10(masked[power].max())) + 1)
        # The minimum value of X is always 0 so we can't display it on the plot, but we can show the top of the errorbar.
        X_minimum = 10**np.floor(np.log10(masked['{}_X_err'.format(power)].min()))
        X_maximum = 10**np.ceil(np.log10(masked['{}_X'.format(power)].max()))
        I_minimum = 10**np.floor(np.log10(masked['{}_I'.format(power)].min()))
        I_maximum = 10**np.ceil(np.log10(masked['{}_I'.format(power)].max()))
        if P_limits is None:
            P_limits = (P_scale * P_minimum, P_scale * P_maximum)
        if X_limits is None:
            X_limits = (X_scale * X_minimum, X_scale * X_maximum)
        if I_limits is None:
            I_limits = (I_scale * I_minimum, I_scale * I_maximum)
        P_fit = np.logspace(np.log10(P_minimum), np.log10(P_maximum), 1e3)
    else:
        P_minimum = -1 / P_scale
        P_maximum = np.ceil(P_scale * 1.1 * masked[power].max()) / P_scale
        if P_limits is None:
            P_limits = (P_scale * P_minimum, P_scale * P_maximum)
        if X_limits is None:
            X_limits = (-10, np.ceil(X_scale * masked['{}_X'.format(power)].max() / 100) * 100)
        if I_limits is None:
            I_limits = (-1, np.ceil(I_scale * masked['{}_I'.format(power)].max() / 10) * 10)
        #P_fit = np.linspace(P_minimum, P_maximum, 1e3)
        P_fit = np.linspace(0, P_maximum, 1e3)

    for plot_index, (X_ax, (channel, group)) in enumerate(zip(axes.flatten(), masked.groupby('channel'))):
        X_ax.errorbar(P_scale * group[power], X_scale * group['{}_X'.format(power)],
                      yerr=X_scale * group['{}_X_err'.format(power)], marker='.', color=X_color, linestyle='None')
        I_ax = X_ax.twinx()
        I_ax.errorbar(P_scale * group[power], I_scale * group['{}_I'.format(power)],
                      yerr=I_scale * group['{}_I_err'.format(power)], marker='.', color=I_color, linestyle='None')

        if fits:
            P_0 = group['{}_XI_fit_P_0'.format(power)].iloc[0]
            P_star = group['{}_XI_fit_P_star'.format(power)].iloc[0]
            X_0 = group['{}_XI_fit_X_0'.format(power)].iloc[0]
            I_0 = group['{}_XI_fit_I_0'.format(power)].iloc[0]
            I_C = group['{}_XI_fit_I_C'.format(power)].iloc[0]
            X_ax.plot(P_scale * P_fit, X_scale * dataframe.X(P_fit, P_0, P_star, X_0), color=X_color)
            I_ax.plot(P_scale * P_fit, I_scale * dataframe.I(P_fit, P_0, P_star, I_0, I_C), color=I_color)

        if loglog:
            X_ax.set_xscale('log')
            X_ax.set_yscale('log', nonposy='clip')
            I_ax.set_xscale('log')
            I_ax.set_yscale('log', nonposy='clip')
            X_ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))

        X_ax.set_xlim(*P_limits)
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
            X_ax.set_xlabel(P_label)
            X_ax.set_ylabel(X_label, color=X_color)
        if plot_index == (nrows * ncols - 1):
            I_ax.set_ylabel(I_label, color=I_color)

        try:
            X_ax.set_title('{}: {}'.format(group.channel.iloc[0], group.location.iloc[0]))
        except AttributeError:
            X_ax.set_title('{}'.format(group.channel.iloc[0]))

    return fig, axes