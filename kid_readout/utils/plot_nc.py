import numpy as np
import time
from kid_readout.utils.easync import EasyNetCDF4
from matplotlib import pyplot as plt
from kid_readout.analysis.resonator import Resonator

def plot_sweeps(fname,figsize=(10,10)):
    nc = EasyNetCDF4(fname)
    plots = {}
    for name,swp in nc.sweeps.groups.items():
        try:
            epoch = swp.groups['datablocks'].variables['epoch'][0]
            hwidx = np.flatnonzero(nc.hw_state.epoch[:]<epoch)[-1]
            atten = nc.hw_state.dac_atten[hwidx]
        except:
            atten = -1
        try:
            idx = swp.variables['index'][:]
        except:
            idx = np.arange(swp.variables['frequency'].shape[0])
        frs = swp.variables['frequency'][:]
        s21s = swp.variables['s21'][:].view('complex128')
        uniq = list(set(idx))
        uniq.sort()
        if len(uniq) == len(idx):
            coarse = True
            figsize=(10,10)
        else:
            coarse = False
            
        fig = plt.figure(figsize=figsize)
        if atten == -1:
            attenstr = 'unknown'
        else:
            attenstr = '%.1f' % atten
        fig.text(0.5,0.9,('%s %s dB\nplotted: %s' % (name,attenstr,time.ctime())),ha='center',va='top',bbox=dict(alpha=0.9,color='w'))
        
        if coarse:
            ax = fig.add_subplot(111)
            ax.set_title(' ')
            ax.plot(frs,10*np.log10(s21s),lw=1.5)
            ax.set_xlabel('frequency (MHz)')
            ax.set_ylabel('|S21|^2 dB')
        else:
            nax = np.floor(np.sqrt(len(uniq)))
            if nax == 0:
                nax = 1
            nay = np.ceil(len(uniq)/float(nax))
            for k in range(len(uniq)):
                ax = fig.add_subplot(nay,nax,k+1)
                msk = idx == uniq[k]
                fr = frs[msk]
                s21 = s21s[msk]*np.exp(-1j*398*fr)
                rr = Resonator(fr,s21)                
                ax.plot(fr-rr.f_0,20*np.log10(s21),lw=1.5)
                ffine = np.linspace(fr.min(),fr.max(),100)
                ax.plot(ffine-rr.f_0,20*np.log10(rr.model(f=ffine)),'r',alpha=0.7)
                depth = 20*np.log10(np.abs(rr.model(f=rr.f_0)/rr.A_mag))
                msg = "f0: %.6f\nQi: %.1f\ndepth: %.2f dB" %  (rr.f_0,rr.Q_i,depth)
                ax.text(0.5,0.1,msg,transform=ax.transAxes,ha='center',va='bottom',bbox=dict(alpha=0.7,color='w'))
        plots[name] = (fig,coarse)
    return plots