import numpy as np
import time
import bisect
from kid_readout.utils.easync import EasyNetCDF4
from matplotlib import pyplot as plt
from kid_readout.analysis.resonator import Resonator
import kid_readout.utils.parse_srs

def setup_axes(nres,figsize=(15,10)):
    fig = plt.figure(figsize=figsize)
    nax = np.floor(np.sqrt(nres))
    if nax == 0:
        nax = 1
    nay = np.ceil(nres/float(nax))
    axes = [fig.add_subplot(nay,nax,k+1) for k in range(nres)]
    return fig,axes

def get_f0s(swps,nres = None):
    if nres is None:
        nres = len(set([x.index for x in swps]))
    f0s = dict([(x,0) for x in range(nres)])
    for swp in swps:
        idx = swp.index
        if f0s[idx] < swp.f_0:
            f0s[idx] = swp.f_0
    return f0s

def plot_f0(swps,nres=None):
    if nres is None:
        nres = len(set([x.index for x in swps]))
    f0s = get_f0s(swps,nres)
    temps = list(set([np.round(x.init_temp*100)/100. for x in swps]))
    temps.sort()
    powers = list(set([x.power_dbm for x in swps]))
    powers.sort()
    reslist = np.array([f0s[k] for k in range(nres)]).argsort()
    colors = plt.cm.spectral(np.linspace(0.1,0.9,len(powers)))        
    
    data = []
    for swp in swps:
        data.append((swp.index,swp.power_dbm,swp.init_temp,swp.f_0,swp.Q_i,20*np.log10(np.abs(swp.model(f=swp.f_0)/swp.A_mag))))
    data = np.array(data)
    # data is now (nswp,6) array
    f0fig,f0axes = setup_axes(nres)
    qifig,qiaxes = setup_axes(nres)
    for axn,res in enumerate(reslist):
        f0ax = f0axes[axn]
        qiax = qiaxes[axn]
        rdat = data[data[:,0]==res,:]
        for power in powers:
            color=colors[powers.index(power)]
            pdat = rdat[rdat[:,1] == power,:]
            x = pdat[:,2]
            order = x.argsort()
            y = pdat[:,3]
            x = x[order]
            y = y[order]
            y = 1e6*(y-f0s[res])/f0s[res]
            f0ax.plot(x,y,'.-',color=color,markersize=4,label=('%.1f dBm' % power))
            
            x = pdat[:,2]
            order = x.argsort()
            y = pdat[:,4]
            x = x[order]
            y = y[order]
            qiax.plot(x,y,'.-',color=color,markersize=4,label=('%.1f dBm' % power))
        f0ax.text(0.1,0.1,('%.6f MHz' % f0s[res]),ha='left',va='bottom',bbox=dict(color='w',alpha=0.9),transform = f0ax.transAxes)
        #f0ax.set_xscale('log')
        qiax.text(0.1,0.1,('%.6f MHz' % f0s[res]),ha='left',va='bottom',bbox=dict(color='w',alpha=0.9),transform = qiax.transAxes)
        f0ax.set_title(' ')
        f0ax.set_ylim(-200,10)

    f0axes[0].legend(loc='upper left',prop=dict(size='x-small'))
    f0axes[0].set_ylabel('(delta_f0)*1e6')
    f0axes[-1].set_xlabel('Temperature (K)')
    f0fig.text(0.5,0.9,('plotted: %s' % (time.ctime())),ha='center',va='top',bbox=dict(alpha=0.9,color='w'))

#        ax.set_xlim(-0.05,0.01)
#        ax.set_ylim(-5,1)

def plot_mag_s21(swps):
    nres = len(set([x.index for x in swps]))
    print nres
    f0s = get_f0s(swps,nres)
#    fmax = np.array([x.f.max() for x in range(nres)])
    temps = list(set([np.round(x.init_temp*100)/100. for x in swps]))
    temps.sort()
    powers = list(set([x.power_dbm for x in swps]))
    powers.sort()
    reslist = range(nres)
    colors = plt.cm.spectral(np.linspace(0.1,0.9,len(powers)))
    for temp in temps:  
        powfig,axes = setup_axes(nres)
        for swp in swps:
            if abs(swp.init_temp - temp) > 0.010:
                continue
            ax = axes[reslist[swp.index]]
            color=colors[powers.index(swp.power_dbm)]
            ax.plot(swp.f-f0s[swp.index], 20*np.log10(swp.data/swp.A_mag),'.-',color=color,markersize=2,label=('%.1f dBm' % swp.power_dbm))
            ax.plot([swp.f_0-f0s[swp.index]],20*np.log10(swp.model(f=swp.f_0)/swp.A_mag),'x',mew=2,color='k')
        for res in reslist:
            ax = axes[res]
            ax.text(0.1,0.1,('%.6f MHz' % f0s[res]),ha='left',va='bottom',bbox=dict(color='w',alpha=0.9),transform = ax.transAxes)
            ax.set_title(' ')
            ax.set_xlim(-0.01,0.01)
            ax.set_ylim(-5,1)
        axes[0].legend(loc='upper left',prop=dict(size='x-small'))
        powfig.text(0.5,0.9,('%.1f mK plotted: %s' % ((temp*1000),time.ctime())),ha='center',va='top',bbox=dict(alpha=0.9,color='w'))

    colors = plt.cm.spectral(np.linspace(0.1,0.9,len(temps)))
    for power in powers:  
        tempfig,axes = setup_axes(nres)
        for swp in swps:
            if swp.power_dbm != power:
                continue
            ax = axes[reslist[swp.index]]
            tempidx = temps.index(np.round(swp.init_temp*100)/100.)
            color=colors[tempidx]
            ax.plot(swp.f-f0s[swp.index], 20*np.log10(swp.data/swp.A_mag),'.-',color=color,markersize=2,label=('%.1f mK' % (1000*temps[tempidx])))
            ax.plot([swp.f_0-f0s[swp.index]],20*np.log10(swp.model(f=swp.f_0)/swp.A_mag),'x',mew=2,color='k')
        for res in reslist:
            ax = axes[res]
            ax.text(0.1,0.1,('%.6f MHz' % f0s[res]),ha='left',va='bottom',bbox=dict(color='w',alpha=0.9),transform = ax.transAxes)
            ax.set_title(' ')
            ax.set_xlim(-0.15,0.01)
            ax.set_ylim(-5,1)
        axes[0].legend(loc='upper left',prop=dict(size='x-small'))
        tempfig.text(0.5,0.9,('%.1f dBm plotted: %s' % (power,time.ctime())),ha='center',va='top',bbox=dict(alpha=0.9,color='w'))
               
    

def get_all_sweeps(fname):
    times,temps = kid_readout.utils.parse_srs.get_all_temperature_data()
    nc = EasyNetCDF4(fname)
    sweeps = []
    for (sk,(name,swp)) in enumerate(nc.sweeps.groups.items()):
        try:
            epoch = swp.groups['datablocks'].variables['epoch'][0]
            last_epoch = swp.groups['datablocks'].variables['epoch'][-1]
        except:
            epoch = time.mktime(time.strptime(name,'sweep_%Y%m%d%H%M%S'))
            last_epoch = epoch
            
        
        try:
            hwidx = bisect.bisect(nc.hw_state.epoch[:],epoch)
            atten = nc.hw_state.dac_atten[hwidx]
        except:
            print "failed to find attenuator settings for",swp
            atten = -1
            
        tempidx = bisect.bisect(times,epoch)
        init_temp_time = times[tempidx]
        if abs(init_temp_time - epoch) > 11*60:
            print "Warning, may be missing temperature data for sweep", name
            print "closest value at", time.ctime(init_temp_time)
        init_temp = temps[tempidx]
        tempidx = bisect.bisect(times,last_epoch)
        last_temp_time = times[tempidx]
        last_temp = temps[tempidx]
        try:
            idx = swp.variables['index'][:]
        except:
            idx = np.arange(swp.variables['frequency'].shape[0])
        frs = swp.variables['frequency'][:]
        s21s = swp.variables['s21'][:].view('complex128')
        uniq = list(set(idx))
        uniq.sort()
        if len(uniq) == len(idx):
            continue
        
        for rk in range(len(uniq)):
            msk = idx == uniq[rk]
            fr = frs[msk]
            s21 = s21s[msk]*np.exp(-1j*398*fr)
            if fr.max() - fr.min() > 0.3:
                flo = bisect.bisect(fr,fr.max()-0.3)
                fr = fr[flo:]
                s21 = s21[flo:]
            try:
                rr = Resonator(fr,s21)
            except:
                print "failed to create resonator for", name,rk,fr.mean()    
            rr.index = uniq[rk]
            rr.atten = atten
            rr.power_dbm = -2 - atten - 40
            rr.epoch = epoch
            rr.last_epoch = last_epoch
            rr.init_temp_time = init_temp_time
            rr.init_temp = init_temp
            rr.last_temp_time = last_temp_time
            rr.last_temp = last_temp
            sweeps.append(rr)
    nc.group.close()
    del nc
    return sweeps
            

def plot_all_sweeps(fname,figsize=(10,10)):
    nc = EasyNetCDF4(fname)
    fig = plt.figure(figsize=figsize)
    fig2 = plt.figure(figsize=figsize)
    fig3 = plt.figure(figsize=figsize)
    plots = {}
    nswp = 0
    uniq0 = None
    for name,swp in nc.sweeps.groups.items():
        try:
            idx = swp.variables['index'][:]
        except:
            idx = np.arange(swp.variables['frequency'].shape[0])
        uniq = list(set(idx))
        uniq.sort()
        if len(uniq) == len(idx):
            continue
        else:
            if uniq0 is None:
                uniq0 = uniq
            nswp += 1
    
    nax = np.floor(np.sqrt(len(uniq)))
    if nax == 0:
        nax = 1
    nay = np.ceil(len(uniq)/float(nax))
    axes = [fig.add_subplot(nay,nax,k+1) for k in range(len(uniq))]
    axes2 = [fig2.add_subplot(nay,nax,k+1) for k in range(len(uniq))]
    axes3 = [fig3.add_subplot(nay,nax,k+1) for k in range(len(uniq))]
    colors = plt.cm.spectral(np.linspace(0.1,0.9,nswp))
    
    for (sk,(name,swp)) in enumerate(nc.sweeps.groups.items()):
        try:
            epoch = swp.groups['datablocks'].variables['epoch'][0]
            hwidx = np.flatnonzero(nc.hw_state.epoch[:]>=epoch)[0]
            atten = nc.hw_state.dac_atten[hwidx]
        except:
            atten = nc.hw_state.dac_atten[-1]
            print "no atten for", name
        try:
            idx = swp.variables['index'][:]
        except:
            idx = np.arange(swp.variables['frequency'].shape[0])
        frs = swp.variables['frequency'][:]
        s21s = swp.variables['s21'][:].view('complex128')
        uniq = list(set(idx))
        uniq.sort()
        if len(uniq) == len(idx):
            continue
        
    
        for k in range(len(uniq)):
            ax = axes[k]
            ax2 = axes2[k]
            ax3 = axes3[k]
            msk = idx == uniq[k]
            fr = frs[msk]
            s21 = s21s[msk]*np.exp(-1j*398*fr)
            nf = len(fr)
            #fr = fr[nf/2:]
            #s21 = s21[nf/2:]
            rr = Resonator(fr,s21)
            f0 = fr.mean()                
            ax.plot(fr-f0,20*np.log10(s21/rr.A_mag)-.1*sk,color=colors[sk])
            ax2.plot(10**(-atten/10.),rr.f_0,'o',color=colors[sk])
#            ax3.plot(10**(-atten/10.),rr.Q_i,'o',color=colors[sk])
            ax3.plot(atten,10*np.log10(rr.Q_i),'o',color=colors[sk])
            #depth = 20*np.log10(np.abs(rr.model(f=rr.f_0)/rr.A_mag))
            #msg = "f0: %.6f\nQi: %.1f\ndepth: %.2f dB" %  (rr.f_0,rr.Q_i,depth)
            #ax.text(0.5,0.1,msg,transform=ax.transAxes,ha='center',va='bottom',bbox=dict(alpha=0.7,color='w'))    
#    for ax in axes:
#        ax.set_xlim(-0.02,0.02)
        

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
                ax.plot(fr-rr.f_0,20*np.log10(s21),'b.-',lw=1.5)
                ffine = np.linspace(fr.min(),fr.max(),100)
                ax.plot(ffine-rr.f_0,20*np.log10(rr.model(f=ffine)),'r',alpha=0.7)
                depth = 20*np.log10(np.abs(rr.model(f=rr.f_0)/rr.A_mag))
                msg = "f0: %.6f\nQi: %.1f\ndepth: %.2f dB" %  (rr.f_0,rr.Q_i,depth)
                ax.text(0.5,0.1,msg,transform=ax.transAxes,ha='center',va='bottom',bbox=dict(alpha=0.7,color='w'))
        plots[name] = (fig,coarse)
    return plots