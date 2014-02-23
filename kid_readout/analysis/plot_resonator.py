from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import lmfit

def extract(r, scale):
    f = r.f[r.mask] * scale
    f_masked = r.f[~r.mask] * scale
    f_0 = r.f_0 * scale
    data = r.data[r.mask]
    masked = r.data[~r.mask]
    f_model = np.linspace(r.f.min(), r.f.max(), 1e3)
    model = r.model(f = f_model)
    f_model *= scale
    model_0 = model[np.argmin(np.abs(f_model - f_0))]
    return {'f': f,
            'f_masked': f_masked,
            'f_0': f_0,
            'data': data,
            'masked': masked,
            'model': model,
            'f_model': f_model,
            'model_0': model_0}

def _plot_on_axis(e, axis):
    axis.plot(e['f'], 20*np.log10(np.abs(e['data'])), linestyle='None', marker='.', markersize=2, color='blue', label='data')
    if e['masked'].size:
        axis.plot(e['f_masked'], 20*np.log10(np.abs(e['masked'])), linestyle='None', marker='.', markersize=2, color='gray', label='masked')

    axis.plot(e['f_model'], 20*np.log10(np.abs(e['model'])), linestyle='-', linewidth=0.5, marker='None', color='brown', label='fit')
    axis.plot(e['f_0'], 20*np.log10(np.abs(e['model_0'])), linestyle='None', marker = '.', markersize=3, color='brown', label='$f_0$')

def one(r, title="", xlabel='frequency [MHz]', ylabel='$|S_{21}|$ [dB]', scale=1e-6, normalize=False): # normalization not implemented
    """
    Plot the data, fit, and f_0.
    """
    interactive = plt.isinteractive()
    plt.ioff()
    fig = plt.figure()
    e = extract(r, scale)
    axis = fig.add_subplot(1, 1, 1)
    _plot_on_axis(e, axis)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    xticks = [e['f_model'].min(), e['f_0'], e['f_model'].max()]
    axis.set_xticks(xticks)
    axis.set_xticklabels(['{:.3f}'.format(tick) for tick in xticks])
    plt.legend(loc='best')
    fig.suptitle(title)
    if interactive:
        plt.ion()
        plt.show()
    return fig

def five_by_four(resonators, title="", xlabel='frequency [MHz]', ylabel='$|S_{21}|$ [dB]', scale=1e-6, normalize=False, sort=False): # normalization not implemented
    if sort:
        resonators.sort(key = lambda r: r.f_0)
    interactive = plt.isinteractive()
    plt.ioff()
    fig = plt.figure(figsize=(4, 3))
    for n, r in enumerate(resonators):
        e = extract(r, scale)
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
