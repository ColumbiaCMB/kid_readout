import numpy as np
import matplotlib.pyplot as plt

def plot_sweeps(df, n, key='channel', legend=False):
    fig, ax = plt.subplots()
    r = df[df[key]==n]
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            ax.plot(row.sweep_freqs_MHz, 20*np.log10(abs(row.sweep_normalized_s21)), label=str(atten))
    ax.set_xlabel('frequency [MHz]')
    ax.set_ylabel('$S_{21}$ [dB]')
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(key, n))
    return fig, ax

def plot_sweeps_IQ(df, n, key='channel', legend=False):
    fig, ax = plt.subplots()
    r = df[df[key]==n]
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            ax.plot(row.sweep_normalized_s21.real, row.sweep_normalized_s21.imag, label=str(atten))
    ax.set_xlabel('Re $S_{21}$')
    ax.set_ylabel('Im $S_{21}$')
    ax.set_xlim(0, 1.8)
    ax.set_ylim(-0.6, 0.6)
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(key, n))
    return fig, ax

def plot_pca_noise(df, n, key='channel', legend=False):
    fig, ax = plt.subplots()
    r = df[(df[key]==n)]
    for atten, atten_group in r.groupby('atten'):
        for index, row in atten_group.iterrows():
            ax.loglog(row.pca_freq, row.pca_eigvals[1], label=str(atten))
            ax.loglog(row.pca_freq, row.pca_eigvals[0])
    ax.set_xlim(1e-3, 1e4)
    if legend:
        ax.legend(loc='best')
    ax.set_title('{}: {}'.format(key, n))
    return fig, ax
