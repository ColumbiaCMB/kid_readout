import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 16.0
from matplotlib import pyplot as plt
mlab = plt.mlab
from kid_readout.utils.easync import EasyNetCDF4
from kid_readout.analysis.resonator import Resonator
import scipy.signal

from kid_readout.utils.fftfilt import fftfilt

import kid_readout.utils.parse_srs
import bisect
import time
import os
from matplotlib.backends.backend_pdf import PdfPages
import cPickle

scale = 4.0/3.814e-6 #amplitude scaling between s21 and time series. Need to fix this in data files eventually
phasecorr = 397.93/2.0 #radians per MHz correction

    
def plot_noise_nc(fname,chip,**kwargs):
    nc = EasyNetCDF4(fname)
    hwg = nc.hw_state.group
    fdir,fbase = os.path.split(fname)
    fbase,ext = os.path.splitext(fbase)
    pdf = PdfPages('/home/data/plots/%s_%s.pdf' % (chip,fbase))
    nms = []
    for (k,((sname,swg),(tname,tsg))) in enumerate(zip(nc.sweeps.groups.items(),nc.timestreams.groups.items())):
        #fig = plot_noise(swg,tsg,hwg,chip,**kwargs)
        nm = NoiseMeasurement(swg,tsg,hwg,chip,k,**kwargs)
        nms.append(nm)
        fig = nm.plot()
        fig.suptitle(('%s %s' % (sname,tname)),fontsize='small')
        pdf.savefig(fig,bbox_inches='tight')
    pdf.close()
    nc.group.close()
    fh = open(os.path.join(fdir,'noise_' +fbase+'.pkl'),'w')
    cPickle.dump(nms,fh,-1)
    fh.close()
    
def plot_per_resonator(pkls):
    dlists = []
    
    for pkl in pkls:
        fh = open(pkl,'r')
        dlists.append(cPickle.load(fh))
        fh.close()
    ids = list(set([x.id for x in dlists[0]]))
    ids.sort()
    byid = {}
    for id in ids:
        byid[id] = []
    for dlist in dlists:
        for nm in dlist:
            byid[nm.id].append(nm)
    pdf = PdfPages('/home/data/plots/noise_summary_%s.pdf' % (dlists[0][0].chip))
    for id in ids:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        axb = ax.twinx()
        axb.set_yscale('log')
        axb.set_xscale('log')
        #ax2 = fig.add_subplot(122)
        hcolors = plt.cm.spectral(np.linspace(0.1,0.9,len(byid[id])))
        ccolors = plt.cm.spectral(np.linspace(0.1,0.9,len(byid[id])))
        for k,nm in enumerate(byid[id]):
            ax.loglog(nm.fr_coarse[1:],nm.prr_coarse[1:],label=('Srr %.1f dBm %.1f mK' % (nm.power_dbm,nm.end_temp*1000)),color=hcolors[k],lw=2)
            ax.loglog(nm.fr_coarse[1:],nm.pii_coarse[1:],'--',color=ccolors[k]) #label=('Sii %.1f dBm %.1f mK' % (nm.power_dbm,nm.end_temp*1000)),
        ax.set_title('%s %d %.6f MHz\nplotted %s' % (nm.chip,id,nm.f0,time.ctime()))
        
        ax.grid()
        ax.set_xlim(50,1e5)
        ax.set_ylabel('Hz$^2$/Hz')
        ax.set_xlabel('Hz')
        axb.grid(color='r')
        ylim = ax.get_ylim()
        axb.set_ylim(np.sqrt(ylim[0])/(nm.f0*1e6),np.sqrt(ylim[1])/(nm.f0*1e6))
        axb.set_ylabel('1/Hz$^{1/2}$')
        ax.legend(prop=dict(size='xx-small'),loc='lower left')
        fig.savefig(('/home/data/plots/noise_%s_%04d.png' % (nm.chip,id)),bbox_inches='tight')
        pdf.savefig(fig,bbox_inches='tight')
    pdf.close()
            
    
class NoiseMeasurement(object):
    def __init__(self,swg,tsg,hwg,chip,id,index=0,phasecorr=phasecorr,scale=scale,filtlen=2**16,loss = -42):
        self.id = id
        self.swp_epoch = swg.groups['datablocks'].variables['epoch'][0]
        self.start_temp =  kid_readout.utils.parse_srs.get_temperature_at(self.swp_epoch)
        self.ts_epoch = tsg.variables['epoch'][:][-1]
        self.end_temp = kid_readout.utils.parse_srs.get_temperature_at(self.ts_epoch)
        self.index = index
        self.chip = chip
        self.phasecorr = phasecorr
        self.scale = scale
        self.filtlen = filtlen
        self.loss = loss
        
        try:
            hwidx = bisect.bisect(hwg.variables['epoch'][:],self.swp_epoch)
            self.atten = hwg.variables['dac_atten'][hwidx]
            self.power_dbm = loss - self.atten #this should be right for a single tone. Off by 6 dB for two tones
        except:
            print "failed to find attenuator settings"
            self.atten = -1
            self.power_dbm = np.nan
        
        idx = swg.variables['index'][:]
        self.fr = swg.variables['frequency'][:]
        self.s21 = swg.variables['s21'][:].view('complex128')*np.exp(-1j*self.fr*phasecorr)*scale
        self.fr = self.fr[idx==index][1:]
        self.s21 = self.s21[idx==index][1:]
        rr = Resonator(self.fr,self.s21)
        self.Q_i = rr.Q_i
        self.params = rr.result.params
        
        ts = tsg.variables['data'][:].view('complex128')
        self.ch = tsg.variables['tone'][index]
        self.nsamp = tsg.variables['nsamp'][index]
        self.fs = tsg.variables['fs'][index]
        self.nfft = tsg.variables['nfft'][index]
        self.f0 = self.fs*self.ch/float(self.nsamp)
        self.tss_raw = ts[index,:]*np.exp(-1j*self.f0*phasecorr)
        self.tsl_raw = fftfilt(scipy.signal.firwin(filtlen,1.0/filtlen), self.tss_raw)[filtlen:]
        
        self.s0 = rr.model(f=self.f0)
        self.sres = rr.model(f=rr.f_0)
        self.ds0 = rr.model(f=self.f0+1e-6)-rr.model(f=self.f0)    
        self.s21m = rr.model(f=np.linspace(self.fr.min(),self.fr.max(),1000)) - self.s0
        
        self.tss = (self.tss_raw-self.s0) * np.exp(-1j*np.angle(self.ds0))
        self.tss = self.tss/np.abs(self.ds0)    
        
        self.prr_fine,self.fr_fine = mlab.psd(self.tss.real,NFFT=2**18,window=mlab.window_none,Fs=self.fs*1e6/self.nfft)
        self.pii_fine,fr = mlab.psd(self.tss.imag,NFFT=2**18,window=mlab.window_none,Fs=self.fs*1e6/self.nfft)
        self.prr_coarse,self.fr_coarse = mlab.psd(self.tss.real,NFFT=2**12,window=mlab.window_none,Fs=self.fs*1e6/self.nfft)
        self.pii_coarse,fr = mlab.psd(self.tss.imag,NFFT=2**12,window=mlab.window_none,Fs=self.fs*1e6/self.nfft)
        
        self.tss_raw = self.tss_raw[:2048]
        self.tss = self.tss[:2048]
        self.tsl_raw = self.tsl_raw[::(filtlen/4)]
        
    def plot(self):
        f1 = plt.figure(figsize=(16,8))
        ax1 = f1.add_subplot(121)
        ax2 = f1.add_subplot(222)
        ax2b = ax2.twinx()
        ax2b.set_yscale('log')
        ax3 = f1.add_subplot(224)
        f1.subplots_adjust(hspace=0.25)
        
        ax1.plot((self.s21-self.s0).real,(self.s21-self.s0).imag,'.-',lw=2,label='measured frequency sweep')
        ax1.plot(self.s21m.real,self.s21m.imag,'.-',markersize=2,label='model frequency sweep')
        ax1.plot([self.sres.real-self.s0.real],[self.sres.imag-self.s0.imag],'kx',mew=2,markersize=20,label='model f0')
        ax1.plot(self.tss_raw.real[:1024]-self.s0,self.tss_raw.imag[:1024]-self.s0.imag,'k,',alpha=0.1,label='timeseries samples')
        ax1.plot(self.tsl_raw.real-self.s0.real,self.tsl_raw.imag-self.s0.imag,'r,') #uses proxy for label
        ax1.annotate("",xytext=(0,0),xy=(self.ds0.real*500,self.ds0.imag*500),arrowprops=dict(lw=2,color='orange',arrowstyle='->'),zorder=0)
        #proxies
        l = plt.Line2D([0,0.1],[0,0.1],color='orange',lw=2)
        l2 = plt.Line2D([0,0.1],[0,0.1],color='r',lw=2)
        
        ax1.text((self.s21-self.s0).real[0],(self.s21-self.s0).imag[0],('%.3f kHz' % ((self.fr[0]-self.f0)*1000)))
        ax1.text((self.s21-self.s0).real[-1],(self.s21-self.s0).imag[-1],('%.3f kHz' % ((self.fr[-1]-self.f0)*1000)))
        ax1.grid()
        handles,labels = ax1.get_legend_handles_labels()
        handles.append(l)
        labels.append('dS21/(500Hz)')
        handles.append(l2)
        labels.append('LPF timeseries')
        ax1.legend(handles,labels,prop=dict(size='xx-small'))
                
        ax2.loglog(self.fr_fine[1:],self.prr_fine[1:],'b',label='Srr')
        ax2.loglog(self.fr_fine[1:],self.pii_fine[1:],'g',label='Sii')
        ax2.loglog(self.fr_coarse[1:],self.prr_coarse[1:],'y',lw=2)
        ax2.loglog(self.fr_coarse[1:],self.pii_coarse[1:],'m',lw=2)
        
        n500 = self.prr_coarse[np.abs(self.fr_coarse-500).argmin()]
        ax2.annotate(("%.2g Hz$^2$/Hz @ 500 Hz" % n500),xy=(500,n500),xycoords='data',xytext=(5,20),textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))
        
        ylim = ax2.get_ylim()
        ax2b.set_xscale('log')
    #    ax2b.set_xlim(ax2.get_xlim())
        ax2.grid()
        ax2b.grid(color='r')
        ax2b.set_ylim(np.sqrt(ylim[0])/(self.f0*1e6),np.sqrt(ylim[1])/(self.f0*1e6))
        ax2.set_xlim(1,1e5)
        ax2.set_ylabel('Hz$^2$/Hz')
        ax2.set_xlabel('Hz')
        ax2b.set_ylabel('1/Hz$^{1/2}$')
        ax2.legend(prop=dict(size='small'))
        
        tsl = (self.tsl_raw-self.s0)/self.ds0
        tsl = tsl - tsl.mean()
        dtl = (self.filtlen/4)/(self.fs*1e6/self.nfft)
        t = dtl*np.arange(len(tsl))
        ax3.plot(t,tsl.real,'b',lw=2,label = 'LPF timeseries real')
        ax3.plot(t,tsl.imag,'g',lw=2,label = 'LPF timeseries imag')
        ax3.set_ylabel('Hz')
        ax3.set_xlabel('seconds')
        ax3.legend(prop=dict(size='xx-small'))
        
        params = self.params
        text = (("measured at: %.6f MHz\n" % self.f0)
                + ("temperature: %.1f - %.1f mK\n" %(self.start_temp*1000, self.end_temp*1000))
                + ("power: ~%.1f dBm\n" %(self.power_dbm))
                + ("fit f0: %.6f +/- %.6f MHz\n" % (params['f_0'].value,params['f_0'].stderr))
                + ("Q: %.1f +/- %.1f\n" % (params['Q'].value,params['Q'].stderr))
                + ("Re(Qe): %.1f +/- %.1f\n" % (params['Q_e_real'].value,params['Q_e_real'].stderr))
                + ("|Qe|: %.1f\n" % (abs(params['Q_e_real'].value+1j*params['Q_e_imag'].value)))
                + ("Qi: %.1f" % (self.Q_i))
                )
        
        ax1.text(0.5,0.5,text,ha='center',va='center',bbox=dict(fc='white',alpha=0.6),transform = ax1.transAxes,
                 fontdict=dict(size='x-small'))
        
        title = ("%s\nmeasured %s\nplotted %s" % (self.chip,time.ctime(self.swp_epoch),time.ctime()))
        ax1.set_title(title,size='small')
        return f1
        
def load_noise_pkl(pklname):
    fh = open(pklname,'r')
    pkl = cPickle.load(fh)
    fh.close()
    return pkl
    
def plot_noise(swg,tsg,hwg,chip,index=0,phasecorr=phasecorr,scale=scale,filtlen=2**16):
    
    swp_epoch = swg.groups['datablocks'].variables['epoch'][0]
    start_temp =  kid_readout.utils.parse_srs.get_temperature_at(swp_epoch)
    ts_epoch = tsg.variables['epoch'][:][-1]
    end_temp = kid_readout.utils.parse_srs.get_temperature_at(ts_epoch)
    
    try:
        hwidx = bisect.bisect(hwg.variables['epoch'][:],swp_epoch)
        atten = hwg.variables['dac_atten'][hwidx]
    except:
        print "failed to find attenuator settings"
        atten = -1
    
    if atten >= 0:
        power_dbm = -40 -2 - atten  #this should be right for a single tone. Off by 6 dB for two tones
    else:
        power_dbm = 999
    
    idx = swg.variables['index'][:]
    fr = swg.variables['frequency'][:]
    s21 = swg.variables['s21'][:].view('complex128')*np.exp(-1j*fr*phasecorr)*scale
    fr = fr[idx==index][1:]
    s21 = s21[idx==index][1:]
    rr = Resonator(fr,s21)
    
    ts = tsg.variables['data'][:].view('complex128')
    ch = tsg.variables['tone'][0]
    nsamp = tsg.variables['nsamp'][0]
    fs = tsg.variables['fs'][0]
    nfft = tsg.variables['nfft'][0]
    f0 = fs*ch/float(nsamp)
    tss = ts[index,:]*np.exp(-1j*f0*phasecorr)
    tsl = fftfilt(scipy.signal.firwin(filtlen,1.0/filtlen), tss)[filtlen:]
    
    s0 = rr.model(f=f0)
    ds0 = rr.model(f=f0+1e-6)-rr.model(f=f0)    
    s21m = rr.model(f=np.linspace(fr.min(),fr.max(),1000)) - s0
    
    
    f1 = plt.figure(figsize=(16,8))
    ax1 = f1.add_subplot(121)
    ax2 = f1.add_subplot(222)
    ax2b = ax2.twinx()
    ax2b.set_yscale('log')
    ax3 = f1.add_subplot(224)
    f1.subplots_adjust(hspace=0.25)
    
    ax1.plot((s21-s0).real,(s21-s0).imag,'.-',lw=2,label='measured frequency sweep')
    ax1.plot(s21m.real,s21m.imag,'.-',markersize=2,label='model frequency sweep')
    ax1.plot([rr.model(f=rr.f_0).real-s0.real],[rr.model(f=rr.f_0).imag-s0.imag],'kx',mew=2,markersize=20,label='model f0')
    ax1.plot(tss.real[:1024]-s0,tss.imag[:1024]-s0.imag,'k,',alpha=0.1,label='timeseries samples')
    ax1.plot(tsl.real[::1024]-s0.real,tsl.imag[::1024]-s0.imag,'r,') #uses proxy for label
    ax1.annotate("",xytext=(0,0),xy=(ds0.real*500,ds0.imag*500),arrowprops=dict(lw=2,color='orange',arrowstyle='->'),zorder=0)
    #proxies
    l = plt.Line2D([0,0.1],[0,0.1],color='orange',lw=2)
    l2 = plt.Line2D([0,0.1],[0,0.1],color='r',lw=2)
    
    ax1.text((s21-s0).real[0],(s21-s0).imag[0],('%.3f kHz' % ((fr[0]-f0)*1000)))
    ax1.text((s21-s0).real[-1],(s21-s0).imag[-1],('%.3f kHz' % ((fr[-1]-f0)*1000)))
    ax1.grid()
    handles,labels = ax1.get_legend_handles_labels()
    handles.append(l)
    labels.append('dS21/(500Hz)')
    handles.append(l2)
    labels.append('LPF timeseries')
    ax1.legend(handles,labels,prop=dict(size='xx-small'))
    
    tss = (tss-s0) * np.exp(-1j*np.angle(ds0))
    tss = tss/np.abs(ds0)    
    
    prr,fr = mlab.psd(tss.real,NFFT=2**18,window=mlab.window_none,Fs=fs*1e6/nfft)
    pii,fr = mlab.psd(tss.imag,NFFT=2**18,window=mlab.window_none,Fs=fs*1e6/nfft)
    ax2.loglog(fr[1:],prr[1:],'b',label='Srr')
    ax2.loglog(fr[1:],pii[1:],'g',label='Sii')
    prr,fr = mlab.psd(tss.real,NFFT=2**12,window=mlab.window_none,Fs=fs*1e6/nfft)
    pii,fr = mlab.psd(tss.imag,NFFT=2**12,window=mlab.window_none,Fs=fs*1e6/nfft)
    ax2.loglog(fr[1:],prr[1:],'y',lw=2)
    ax2.loglog(fr[1:],pii[1:],'m',lw=2)
    
    n500 = prr[np.abs(fr-500).argmin()]
    ax2.annotate(("%.2g Hz$^2$/Hz @ 500 Hz" % n500),xy=(500,n500),xycoords='data',xytext=(5,20),textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    
    ylim = ax2.get_ylim()
    ax2b.set_xscale('log')
#    ax2b.set_xlim(ax2.get_xlim())
    ax2.grid()
    ax2b.grid(color='r')
    ax2b.set_ylim(np.sqrt(ylim[0])/(f0*1e6),np.sqrt(ylim[1])/(f0*1e6))
    ax2.set_xlim(1,1e5)
    ax2.set_ylabel('Hz$^2$/Hz')
    ax2.set_xlabel('Hz')
    ax2b.set_ylabel('1/Hz$^{1/2}$')
    ax2.legend(prop=dict(size='small'))
    
    tsl = (tsl-s0)/ds0
    tsl = tsl - tsl.mean()
    dtl = 1/(fs*1e6/nfft)
    t = dtl*np.arange(len(tsl))
    ax3.plot(t[::filtlen/4],tsl[::filtlen/4].real,'b',lw=2,label = 'LPF timeseries real')
    ax3.plot(t[::filtlen/4],tsl[::filtlen/4].imag,'g',lw=2,label = 'LPF timeseries imag')
    ax3.set_ylabel('Hz')
    ax3.set_xlabel('seconds')
    ax3.legend(prop=dict(size='xx-small'))
    
    params = rr.result.params
    text = (("measured at: %.6f MHz\n" % f0)
            + ("temperature: %.1f - %.1f mK\n" %(start_temp*1000, end_temp*1000))
            + ("power: ~%.1f dBm\n" %(power_dbm))
            + ("fit f0: %.6f +/- %.6f MHz\n" % (rr.f_0,params['f_0'].stderr))
            + ("Q: %.1f +/- %.1f\n" % (rr.Q,params['Q'].stderr))
            + ("Re(Qe): %.1f +/- %.1f\n" % (rr.Q_e_real,params['Q_e_real'].stderr))
            + ("|Qe|: %.1f\n" % (abs(rr.Q_e_real+1j*rr.Q_e_imag)))
            + ("Qi: %.1f" % (rr.Q_i))
            )
    
    ax1.text(0.5,0.5,text,ha='center',va='center',bbox=dict(fc='white',alpha=0.6),transform = ax1.transAxes,
             fontdict=dict(size='x-small'))
    
    title = ("%s\nmeasured %s\nplotted %s" % (chip,time.ctime(swp_epoch),time.ctime()))
    ax1.set_title(title,size='small')
    return f1