from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import lmfit

# Calculate everything in the resonator units, then scale the
# frequency at the end. Too complicated otherwise.
def extract(r, normalize_s21=False, freq_scale=1, points=1e3):
    freq_model = np.linspace(r.freq_data.min(), r.freq_data.max(), points)
    s21_data = r.s21_data[r.mask]
    s21_masked = r.s21_data[~r.mask]
    s21_model = r.model(x = freq_model)
    s21_model_0 = s21_model[np.argmin(np.abs(freq_model - r.f_0))]
    if normalize_s21:
        s21_data *= r.get_normalization(r.freq_data[r.mask])
        s21_masked *= r.get_normalization(r.freq_data[~r.mask])
        model_normalization = r.get_normalization(freq_model)
        s21_model *= model_normalization
        s21_model_0 *= model_normalization[np.argmin(np.abs(freq_model - r.f_0))]
    return {'f': r.freq_data[r.mask] * freq_scale,
            'f_masked': r.freq_data[~r.mask] * freq_scale,
            'f_0': r.f_0 * freq_scale,
            'data': s21_data,
            'masked': s21_masked,
            'model': s21_model,
            'f_model': freq_model * freq_scale,
            'model_0': s21_model_0}

def _plot_amplitude_on_axis(extracted, axis, plot_masked):
    axis.plot(extracted['f'], 20*np.log10(np.abs(extracted['data'])),
              linestyle='None', marker='.', markersize=2, color='blue', label='data')
    if plot_masked and extracted['masked'].size:
        axis.plot(extracted['f_masked'], 20*np.log10(np.abs(extracted['masked'])),
                  linestyle='None', marker='.', markersize=2, color='gray', label='masked')
    axis.plot(extracted['f_model'], 20*np.log10(np.abs(extracted['model'])),
              linestyle='-', linewidth=0.5, marker='None', color='brown', label='fit')
    axis.plot(extracted['f_0'], 20*np.log10(np.abs(extracted['model_0'])),
              linestyle='None', marker = '.', markersize=3, color='brown', label='$f_0$')

def _plot_phase_on_axis(extracted, axis, plot_masked):
#    axis.plot(extracted['f'], np.unwrap(np.angle(extracted['data'])),
    axis.plot(extracted['f'], np.angle(extracted['data']),
              linestyle='None', marker='.', markersize=2, color='blue', label='data')
    if plot_masked and extracted['masked'].size:
#        axis.plot(extracted['f_masked'], np.unwrap(np.angle(extracted['masked'])),
        axis.plot(extracted['f_masked'], np.angle(extracted['masked']),
                  linestyle='None', marker='.', markersize=2, color='gray', label='masked')
#    axis.plot(extracted['f_model'], np.unwrap(np.angle(extracted['model'])),
    axis.plot(extracted['f_model'], np.angle(extracted['model']),
              linestyle='-', linewidth=0.5, marker='None', color='brown', label='fit')
    axis.plot(extracted['f_0'], np.angle(extracted['model_0']),
              linestyle='None', marker = '.', markersize=3, color='brown', label='$f_0$')

def amplitude(r, title="", xlabel='frequency [MHz]', ylabel='$|S_{21}|$ [dB]',
              plot_masked=True, **kwds):
    """
    Plot the data, fit, and f_0.
    """
    interactive = plt.isinteractive()
    plt.ioff()
    fig = plt.figure()
    extracted = extract(r, **kwds)
    axis = fig.add_subplot(1, 1, 1)
    _plot_on_axis(extracted, axis)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    xticks = [extracted['f_model'].min(), extracted['f_0'], extracted['f_model'].max()]
    axis.set_xticks(xticks)
    axis.set_xticklabels(['{:.3f}'.format(tick) for tick in xticks])
    plt.legend(loc='best')
    fig.suptitle(title)
    if interactive:
        plt.ion()
        plt.show()
    return fig

def amplitude_and_phase(r, title="", xlabel='frequency [MHz]', amp_label='$|S_{21}|$ [dB]', phase_label='phase [rad]',
                        plot_masked=True, **kwds):
    interactive = plt.isinteractive()
    plt.ioff()
    fig, axes = plt.subplots(2, 1, sharex=True)
    extracted = extract(r, **kwds)
    _plot_phase_on_axis(extracted, axes[0], plot_masked)
    _plot_amplitude_on_axis(extracted, axes[1], plot_masked)
    axes[0].set_ylabel(phase_label)
    axes[1].set_ylabel(amp_label)
    axes[1].set_xlabel(xlabel)
    xticks = [extracted['f_model'].min(), extracted['f_0'], extracted['f_model'].max()]
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(['{:.3f}'.format(tick) for tick in xticks])
    #plt.legend(loc='best')
    fig.suptitle(title)
    if interactive:
        plt.ion()
        plt.show()
    return fig

def IQ_circle(r, title="", xlabel=r"Re $S_{21}$", ylabel=r"Im $S_{21}$", plot_masked=True, **kwds):
    interactive = plt.isinteractive()
    plt.ioff()
    fig, ax = plt.subplots()
    extracted = extract(r, **kwds)
    ax.plot(extracted['data'].real, extracted['data'].imag, linestyle='None', marker='.', color='blue', label='data')
    if plot_masked and extracted['masked'].size:
        ax.plot(extracted['masked'].real, extracted['masked'].imag, linestyle='None', marker='.', color='gray', label='masked')
    ax.plot(extracted['model'].real, extracted['model'].imag, linestyle='-', linewidth=0.5, color='brown', label='fit')
    ax.plot(extracted['model_0'].real, extracted['model_0'].imag, linestyle='None', marker='.', color='brown', label='$f_0$')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    if interactive:
        plt.ion()
        plt.show()
    return fig

def five_by_four(resonators, title="", xlabel='frequency [MHz]', ylabel='$|S_{21}|$ [dB]', sort=False, **kwds):
    if sort:
        resonators.sort(key = lambda r: r.f_0)
    interactive = plt.isinteractive()
    plt.ioff()
    fig = plt.figure(figsize=(4, 3))
    for n, r in enumerate(resonators):
        e = extract(r, **kwds)
        axis = fig.add_subplot(4, 5, n+1)
        _plot_on_axis(e, axis)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(right=False, top=False, direction='out', pad=1)
        xticks = [e['f_0']]
        axis.set_xticks(xticks)
        axis.set_xticklabels(['{:.3f}'.format(tick) for tick in xticks])
        yticks = [20*np.log10(np.abs(e['model_0'])), 20*np.log10(np.mean((np.abs(e['data'][0]), np.abs(e['data'][-1]))))]
        axis.set_yticks(yticks)
        axis.set_yticklabels(['{:.0f}'.format(tick) for tick in yticks])

    fig.text(0.3, 0.94, title, fontsize='medium', color='black')
    fig.text(0.02, 0.94, ylabel, fontsize='medium', color='black')
    fig.text(0.05, 0.04, xlabel, fontsize='medium', color='black')
    fig.text(0.4, 0.04, 'data', fontsize='medium', color='blue')
    fig.text(0.55, 0.04, 'masked', fontsize='medium', color='gray')
    fig.text(0.75, 0.04, 'fit and f_0', fontsize='medium', color='brown')
    fig.set_tight_layout({'pad': 0.5, 'h_pad': 0.2, 'w_pad': 0.3, 'rect': (0, 0.08, 1, 0.94)})
    if interactive:
        plt.ion()
        plt.show()
    return fig

def covar(mzr,nsigma=8,nx=15):
    params = mzr.params
    pnames = [x for x in params.keys() if x != 'A_phase']
    nparam = len(pnames)
    f = plt.figure(figsize=(24,24))
    for p1 in range(nparam):
        for p2 in range(p1+1,nparam):
            print pnames[p1],pnames[p2]
            ax = f.add_subplot(nparam,nparam,nparam*p2+p1+1)
            x0 = params[pnames[p1]].value
            xerr = params[pnames[p1]].stderr*nsigma
            y0 = params[pnames[p2]].value
            yerr = params[pnames[p2]].stderr*nsigma
            x,y,im = lmfit.conf_interval2d(mzr,pnames[p1],pnames[p2],limits=((x0-xerr,x0+xerr),(y0-yerr,y0+yerr)),nx=nx,ny=nx)
            ax.contourf((x-x0)/abs(x0),(y-y0)/abs(y0),im,20)
            ax.set_xlabel(pnames[p1])
            ax.set_ylabel(pnames[p2])
