import numpy as np
import matplotlib
matplotlib.use('agg')
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 16.0
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
mlab = plt.mlab

from kid_readout.utils.easync import EasyNetCDF4
from kid_readout.analysis.resonator import Resonator,fit_best_resonator
from kid_readout.analysis import khalil
from kid_readout.analysis import iqnoise
import scipy.signal

from kid_readout.utils.fftfilt import fftfilt
from kid_readout.roach.tools import ntone_power_correction

from kid_readout.utils.despike import deglitch_window

import socket
if socket.gethostname() == 'detectors':
    from kid_readout.utils.hpd_temps import get_temperature_at
else:
    from kid_readout.utils.parse_srs import get_temperature_at
import bisect
import time
import os
import glob

import cPickle

scale = 4.0/3.814e-6 #amplitude scaling between s21 and time series. Need to fix this in data files eventually
phasecorr = 397.93/2.0 #radians per MHz correction

    
def plot_noise_nc(fglob,chip,**kwargs):
    if type(fglob) is str:
        fnames = glob.glob(fglob)
    else:
        fnames = fglob
    try:
        plotall = kwargs.pop('plot_all')
    except KeyError:
        plotall = False
    fnames.sort()
    errors = {}
    for fname in fnames:
        try:
            nc = EasyNetCDF4(fname)
            hwg = nc.hw_state.group
            fdir,fbase = os.path.split(fname)
            fbase,ext = os.path.splitext(fbase)
            chipfname = chip.replace(' ','_')
            pdf = PdfPages('/home/data/plots/%s_%s.pdf' % (fbase,chipfname))
            nms = []
            for (k,((sname,swg),(tname,tsg))) in enumerate(zip(nc.sweeps.groups.items(),nc.timestreams.groups.items())):
                #fig = plot_noise(swg,tsg,hwg,chip,**kwargs)
                indexes = np.unique(swg.variables['index'][:])
                for index in indexes:
                    kwargs['index'] = index
                    try:
                        nm = NoiseMeasurement(swg,tsg,hwg,chip,k,**kwargs)
                    except IndexError:
                        print "failed to find index",index,"in",sname,tname
                        continue
                    if pdf is not None:
                        if plotall or k == 0:
                            fig = Figure(figsize=(16,8))
                            title = ('%s %s' % (sname,tname))
                            nm.plot(fig=fig,title=title)
                            canvas = FigureCanvasAgg(fig)
                            fig.set_canvas(canvas)
                            pdf.savefig(fig,bbox_inches='tight')
                        else:
                            pdf.close()
                            pdf = None
                    del nm.fr_fine
                    del nm.pii_fine
                    del nm.prr_fine
                    nms.append(nm)
                    
                print fname,nm.start_temp,"K"
            if pdf is not None:
                pdf.close()
            nc.group.close()
            fh = open(os.path.join('/home/data','noise_' +fbase+'.pkl'),'w')
            cPickle.dump(nms,fh,-1)
            fh.close()
        except Exception,e:
            raise
            errors[fname] = e
    return errors
    
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
    def __init__(self,swg,tsg,hwg,chip,id,index=0,phasecorr=phasecorr,scale=scale,filtlen=2**16,loss = -42, 
                 ntones=None, use_bif=False, delay_estimate = -7.11):
        self.id = id
        self.swp_epoch = swg.groups['datablocks'].variables['epoch'][0]
        self.start_temp =  get_temperature_at(self.swp_epoch)
        self.ts_epoch = tsg.variables['epoch'][:][-1]
        self.end_temp = get_temperature_at(self.ts_epoch)
        self.index = index
        self.chip = chip
        self.phasecorr = phasecorr
        if scale is None:
            self.scale = 1/tsg.variables['wavenorm'][index]
        else:
            self.scale = scale
        self.filtlen = filtlen
        self.loss = loss
        
        try:
            hwidx = bisect.bisect(hwg.variables['epoch'][:],self.swp_epoch)
            if hwidx == hwg.variables['epoch'].shape[0]:
                hwidx = -1
            if ntones is None:
                ntones = hwg.variables['ntones'][hwidx]
            self.atten = hwg.variables['dac_atten'][hwidx]
            self.power_dbm = loss - self.atten - ntone_power_correction(ntones)
        except:
            print "failed to find attenuator settings"
            self.atten = np.nan
            self.power_dbm = np.nan
        
        idx = swg.variables['index'][:]
        self.fr = swg.variables['frequency'][:]
        self.s21 = swg.variables['s21'][:].view('complex128')*np.exp(-1j*self.fr*phasecorr)*self.scale
        self.fr = self.fr[idx==index]#[1:]#[4:-4]
        self.s21 = self.s21[idx==index]#[1:]#[4:-4]
        
        tones = tsg.variables['tone'][:]
        nsamp = tsg.variables['nsamp'][:]
        fs = tsg.variables['fs'][:]
        fmeas = fs*tones/nsamp
        tone_index = np.argmin(abs(fmeas-self.fr.mean()))
        ts = tsg.variables['data'][tone_index,:].view('complex128')
        self.fs = fs[tone_index]
        self.nfft = tsg.variables['nfft'][tone_index]
        window = int(2**np.ceil(np.log2(self.fs*1e6/(2*self.nfft))))
        if window > ts.shape[0]:
            window = ts.shape[0]//2
        ts = deglitch_window(ts,window,thresh=6)
        self.fr = np.hstack((self.fr,[fmeas[tone_index]]))
        self.s21 = np.hstack((self.s21,[ts[:2048].mean()]))
        
        blkidx = swg.groups['datablocks'].variables['sweep_index'][:]
        blks = swg.groups['datablocks'].variables['data'][:][blkidx==index,:].view('complex')
        errors = blks.real.std(1) + 1j*blks.imag.std(1)
        self.errors = errors
        self.errors = np.hstack((self.errors,[ts[:2048].real.std()+ts[:2048].imag.std()]))
        order = self.fr.argsort()
        self.fr = self.fr[order]
        self.s21 = self.s21[order]
        self.errors = self.errors[order]
        def delay_guess(*args):
            if use_bif:
                p = khalil.bifurcation_guess(*args)
            else:
                p = khalil.delayed_generic_guess(*args)
            p['delay'].value = delay_estimate
            return p
        
        if use_bif:
            rr = Resonator(self.fr[2:-2],self.s21[2:-2],model=khalil.bifurcation_s21,guess=delay_guess,errors=errors[2:-2])
        else:
            rr = Resonator(self.fr[2:-2],self.s21[2:-2],errors=errors[2:-2],guess=delay_guess)
        self.delay = rr.delay
        self.s21 = self.s21*np.exp(2j*np.pi*rr.delay*self.fr)
#        if use_bif:
#            rr = Resonator(self.fr,self.s21,model=khalil.bifurcation_s21,guess=khalil.bifurcation_guess,errors=errors)
#        else:
#            rr = Resonator(self.fr,self.s21,mask=rr.mask)#errors=errors)
        rr = fit_best_resonator(self.fr,self.s21,errors=errors)#,min_a=0)
        self.Q_i = rr.Q_i
        self.params = rr.result.params
        
        
        self.ch = tones[tone_index]
        self.nsamp = nsamp[tone_index]
        self.f0 = self.fs*self.ch/float(self.nsamp)
        self.tss_raw = ts*np.exp(-1j*self.f0*phasecorr + 2j*np.pi*self.delay*self.f0)
        self.tsl_raw = fftfilt(scipy.signal.firwin(filtlen,1.0/filtlen), self.tss_raw)[filtlen:]
        
        self.s0 = self.tss_raw.mean() #rr.model(f=self.f0)
        self.sres = rr.model(f=rr.f_0)
        self.ds0 = rr.model(f=self.f0+1e-6)-rr.model(f=self.f0)
        #tsl_hz = np.real((self.tsl_raw - self.s0)/self.ds0)
        #self.ds0s = rr.model(f=self.f0+(tsl_hz+1)*1e-6)-rr.model(f=self.f0+tsl_hz*1e-6)
        self.frm = np.linspace(self.fr.min(),self.fr.max(),1000)
        self.s21m = rr.model(f=self.frm) - self.s0
        
        self.tss = (self.tss_raw-self.s0) * np.exp(-1j*np.angle(self.ds0))
        self.tss = self.tss/np.abs(self.ds0)/(self.f0*1e6)
        #self.tss = (self.tss_raw[:len(self.tsl_raw)] - self.tsl_raw)/self.ds0s
        fr,S,evals,evects,angles,piq = iqnoise.pca_noise(self.tss, NFFT=None, Fs=self.fs*1e6/(2*self.nfft))
        
        self.pca_fr = fr
        self.pca_S = S
        self.pca_evals = evals
        self.pca_evects = evects
        self.pca_angles = angles
        self.pca_piq = piq
        
        self.prr_fine,self.fr_fine = mlab.psd(self.tss.real,NFFT=2**18,window=mlab.window_none,Fs=self.fs*1e6/(2*self.nfft))
        self.pii_fine,fr = mlab.psd(self.tss.imag,NFFT=2**18,window=mlab.window_none,Fs=self.fs*1e6/(2*self.nfft))
        self.prr_coarse,self.fr_coarse = mlab.psd(self.tss.real,NFFT=2**12,window=mlab.window_none,Fs=self.fs*1e6/(2*self.nfft))
        self.pii_coarse,fr = mlab.psd(self.tss.imag,NFFT=2**12,window=mlab.window_none,Fs=self.fs*1e6/(2*self.nfft))
        
        self.tss_raw = self.tss_raw[:2048].copy()
        self.tss = self.tss[:2048].copy()
        self.tsl_raw = self.tsl_raw[::(filtlen/4)].copy()
        
    def plot(self,fig=None,title=''):
        if fig is None:
            f1 = plt.figure(figsize=(16,8))
        else:
            f1 = fig
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
        #ax1.plot(self.pca_evects[0,0,:100]*100,self.pca_evects[1,0,:100]*100,'y.')
        #ax1.plot(self.pca_evects[0,1,:100]*100,self.pca_evects[1,1,:100]*100,'k.')
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
        
        ax1b = inset_axes(parent_axes=ax1, width="20%", height="20%", loc=4)
        ax1b.plot(self.fr,20*np.log10(abs(self.s21)),'.-')
        frm = np.linspace(self.fr.min(),self.fr.max(),1000)
        ax1b.plot(frm,20*np.log10(abs(self.s21m+self.s0)))
                
        ax2.loglog(self.fr_fine[1:],self.prr_fine[1:],'b',label='Srr')
        ax2.loglog(self.fr_fine[1:],self.pii_fine[1:],'g',label='Sii')
        ax2.loglog(self.fr_coarse[1:],self.prr_coarse[1:],'y',lw=2)
        ax2.loglog(self.fr_coarse[1:],self.pii_coarse[1:],'m',lw=2)
        ax2.loglog(self.pca_fr[1:],self.pca_evals[:,1:].T,'k',lw=2)
        ax2.set_title(title,fontdict=dict(size='small'))
        
        n500 = self.prr_coarse[np.abs(self.fr_coarse-500).argmin()]
        ax2.annotate(("%.2g Hz$^2$/Hz @ 500 Hz" % n500),xy=(500,n500),xycoords='data',xytext=(5,20),textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'))
        
        ax2b.set_xscale('log')
    #    ax2b.set_xlim(ax2.get_xlim())
        ax2.grid()
        ax2b.grid(color='r')
        ax2.set_xlim(self.pca_fr[1],self.pca_fr[-1])
        ax2.set_ylabel('1/Hz')
        ax2.set_xlabel('Hz')
        ax2.legend(prop=dict(size='small'))
        
        tsl = (self.tsl_raw-self.s0)/self.ds0
        tsl = tsl - tsl.mean()
        dtl = (self.filtlen/4)/(self.fs*1e6/(2*self.nfft))
        t = dtl*np.arange(len(tsl))
        ax3.plot(t,tsl.real,'b',lw=2,label = 'LPF timeseries real')
        ax3.plot(t,tsl.imag,'g',lw=2,label = 'LPF timeseries imag')
        ax3.set_ylabel('Hz')
        ax3.set_xlabel('seconds')
        ax3.legend(prop=dict(size='xx-small'))
        
        params = self.params
        amp_noise_voltsrthz = np.sqrt(4*1.38e-23*4.0*50)
        vread = np.sqrt(50*10**(self.power_dbm/10.0)*1e-3)
        alpha = 1.0
        Qe = abs(params['Q_e_real'].value+1j*params['Q_e_imag'].value)
        f0_dVdf = 4*vread*alpha*params['Q'].value**2/Qe
        expected_amp_noise = (amp_noise_voltsrthz/f0_dVdf)**2 
        text = (("measured at: %.6f MHz\n" % self.f0)
                + ("temperature: %.1f - %.1f mK\n" %(self.start_temp*1000, self.end_temp*1000))
                + ("power: ~%.1f dBm (%.1f dB att)\n" %(self.power_dbm,self.atten))
                + ("fit f0: %.6f +/- %.6f MHz\n" % (params['f_0'].value,params['f_0'].stderr))
                + ("Q: %.1f +/- %.1f\n" % (params['Q'].value,params['Q'].stderr))
                + ("Re(Qe): %.1f +/- %.1f\n" % (params['Q_e_real'].value,params['Q_e_real'].stderr))
                + ("|Qe|: %.1f\n" % (Qe))
                + ("Qi: %.1f\n" % (self.Q_i))
                + ("Eamp: %.2g 1/Hz" % expected_amp_noise)
                )
        if expected_amp_noise > 0:
            ax2.axhline(expected_amp_noise,linewidth=2,color='m')
            ax2.text(10,expected_amp_noise,r"expected amp noise",va='top',ha='left',fontdict=dict(size='small'))
#        ax2.axhline(expected_amp_noise*4,linewidth=2,color='m')
#        ax2.text(10,expected_amp_noise*4,r"$\alpha = 0.5$",va='top',ha='left',fontdict=dict(size='small'))
        if params.has_key('a'):
            text += ("\na: %.3g +/- %.3g" % (params['a'].value,params['a'].stderr))

        ylim = ax2.get_ylim()
        ax2b.set_ylim(ylim[0]*(self.f0*1e6)**2,ylim[1]*(self.f0*1e6)**2)
        ax2b.set_ylabel('$Hz^2/Hz$')

        
        ax1.text(0.02,0.95,text,ha='left',va='top',bbox=dict(fc='white',alpha=0.6),transform = ax1.transAxes,
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
    start_temp =  get_temperature_at(swp_epoch)
    ts_epoch = tsg.variables['epoch'][:][-1]
    end_temp = get_temperature_at(ts_epoch)
    
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