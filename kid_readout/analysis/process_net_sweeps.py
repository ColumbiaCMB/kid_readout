import numpy as np
import time
import glob
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from collections import defaultdict
import kid_readout.analysis.noise_archive
import kid_readout.analysis.noise_summary


diode_index = 2    
def convtime(tstr):
    return time.mktime(time.strptime(tstr,'%Y%m%d-%H%M%S'))
tdata = np.loadtxt('/home/heather/SRS/20140301-131320.txt',delimiter=',',converters={0:convtime},skiprows=1)
load_time = tdata[:,0]
load_temp = tdata[:,diode_index]
tdata2 = np.loadtxt('/home/heather/SRS/20140305-213554.txt',delimiter=',',converters={0:convtime},skiprows=1)
tdata3 = np.loadtxt('/home/heather/SRS/20140321-222308.txt',delimiter=',',converters={0:convtime},skiprows=1)
tdata4 = np.loadtxt('/home/heather/SRS/20140331-085747.txt',delimiter=',',converters={0:convtime},skiprows=1)
tdata5 = np.loadtxt('/home/heather/SRS/20140404-170807.txt',delimiter=',',converters={0:convtime},skiprows=1)
tdata6 = np.loadtxt('/home/heather/SRS/20140411-145108.txt',delimiter=',',converters={0:convtime},skiprows=1)
tdata7 = np.loadtxt('/home/heather/SRS/20140417-092823.txt',delimiter=',',converters={0:convtime},skiprows=1)
load_time = np.concatenate((load_time,tdata2[:,0],tdata3[:,0],tdata4[:,0],tdata5[:,0],tdata6[:,0],tdata7[:,0]))
far_temp = np.concatenate((load_temp,tdata2[:,diode_index],tdata3[:,diode_index],tdata4[:,diode_index],tdata5[:,diode_index],
                            tdata6[:,diode_index],tdata7[:,diode_index]))
load_temp = np.concatenate((tdata[:,1],tdata2[:,1],tdata3[:,1],tdata4[:,1],tdata5[:,1],tdata6[:,1],tdata7[:,1]))

if not globals().has_key('arch'):
    arch = kid_readout.analysis.noise_archive.load_archive('/home/data/archive/StarCryo_5x4_0813f10_LPF_Horns_NET_1.npy')
    arch['round_temp'] = np.round(arch.end_temp*1000/10)*10
    arch['load_temp'] = np.interp(arch.swp_epoch,load_time,load_temp)

arch = arch[arch.Q_err/arch.Q < 0.1]
#arch = arch[(arch.power_dbm > -114) & (arch.power_dbm <= -98)]
#arch = arch[np.abs(arch.end_temp -0.235) > 0.01]
arch = arch[(arch.Q_i > 0) & (arch.Q_i < 4e5)]
idxs = np.unique(arch['ridx'])

def load_net_pkl(pkl):
    nms = kid_readout.analysis.noise_summary.load_noise_pkl(pkl)
    for nm in nms:
        nm.ts_temp = np.interp(nm.ts_epoch,load_time,load_temp)
        nm.ts_far_temp = np.interp(nm.ts_epoch,load_time,far_temp)
    tstart = nms[0].ts_temp
    tend = nms[-1].ts_temp
    print "Start: %.3f K, End: %.3f K, difference %.3f K" % (tstart,tend,tend-tstart)
    return nms

def plot_all_net_2014_03_cu_pkg(pkls= glob.glob('/home/data/noise_2014-03-*.pkl')):
    pkls.sort()
    res = defaultdict(list)
    for pkl in pkls:
        print pkl
        nms = load_net_pkl(pkl)
        if nms[0].atten == 10.0:
            for idx in range(20):
                thisidx = [nm for nm in nms if nm.index == idx]
                if len(thisidx):
                    res[idx].append(thisidx[0])
    print "loaded all data"
    chipfname = res[0][0].chip.replace(' ','_')
    allpdf = PdfPages('/home/data/plots/summary_net_%s.pdf' % (chipfname))
    for ridx in range(20):
        print "processing resonator",ridx
        res0 = res[ridx]
        chipfname = res0[0].chip.replace(' ','_')
        pdf = PdfPages('/home/data/plots/summary_net_%s_%02d.pdf' % (chipfname,ridx))
        npzname = '/home/data/archive/summary_net_%s_%02d.npz' % (chipfname,ridx)
        T = np.array([nm.ts_temp for nm in res0])
        Q = np.array([nm.params['Q'].value for nm in res0])
        Qi = np.array([nm.Q_i for nm in res0])
        Tphys = np.array([nm.end_temp for nm in res0])
        f0s= np.array([nm.params['f_0'].value for nm in res0])
        f0errs= np.array([nm.params['f_0'].stderr for nm in res0])
        
        title = '%s\n%02d - %f' % (res0[0].chip, ridx, np.median(f0s))

        df0=(f0s-f0s[T<5.5].min())*1e6
        msk = (f0errs < 0.00005) & (Tphys < 0.22) & (Tphys > 0.18)
        if msk.sum() < 2:
            continue
        df0 = df0[msk]
        f0s = f0s[msk]
        f0errs = f0errs[msk]
        T = T[msk]
        Tphys = Tphys[msk]
        Q = Q[msk]
        Qi = Qi[msk]
        pp = np.polyfit(T[T<4],df0[T<4],1)
        Hz_per_K = abs(pp[0])
        print Hz_per_K,"Hz/K"

        fig = Figure(figsize=(11,8))
        ax = fig.add_subplot(221)
        fig.suptitle(title,size='small')
        ax.errorbar(T,df0,yerr = f0errs*1e6,linestyle='none',marker='.')
        Tm = np.linspace(3,7.5,100)
        ax.plot(Tm,np.polyval(pp,Tm),label=('%.1f Hz/K'% pp[0]))
        ax.set_xlabel('$T_{load}$ [K]')
        ax.set_ylabel('Frequency shift [Hz]')
        ax.legend(loc='upper right')

        ax3 = fig.add_subplot(223)
        ax3.plot(T,Qi,'o',label='Qi')
        ax3.plot(T,Q,'x',mew=2,label='Qr')
        ax3.set_ylim(0,7e4)
        ax3.set_ylabel('Quality factor')
        ax3.set_xlabel('$T_{load}$ [K]')
        ax3.legend(loc='upper right',prop=dict(size='small'))


        ax = fig.add_subplot(222)
        ax2 = fig.add_subplot(224,sharex=ax)
        Tn = []
        dev_noise = []
        amp_noise = []
        for k,nm in enumerate(res0):
            if msk[k]:
                idx = (nm.pca_fr > 150)&(nm.pca_fr < 240)
                dev = (np.sqrt(nm.pca_evals[1,idx])*nm.f0*1e6).mean()
                amp = (np.sqrt(nm.pca_evals[1,-200:-10])*nm.f0*1e6).mean()
                Tn.append(nm.ts_temp)
                dev_noise.append(dev)
                amp_noise.append(amp)
                ax.plot(nm.ts_temp,dev,'b.')
                ax.plot(nm.ts_temp,amp,'r.')
                ax.plot(nm.ts_temp,dev-amp,'g.')
                uKrtHz = (dev*1e6/Hz_per_K)
                uKrts = uKrtHz/np.sqrt(2)
                ax2.plot(nm.ts_temp,uKrts,'b.')
        ax.set_ylim(0,0.6)
        ax2.set_ylim(0,60)
        ax.plot(0,0,'b.',label='Device @ 150 Hz')
        ax.plot(0,0,'r.',label='Amp')
        ax.plot(0,0,'g.',label='Dev-Amp')
        ax.set_xlim(3,T.max())
        ax.legend(loc='upper right',prop=dict(size='small'))
        ax2.grid()
        ax.grid()
        ax.set_ylabel('$Hz/\sqrt{Hz}$')
        ax2.set_ylabel('NET $\mu$K$\sqrt{s}$')
        ax2.set_xlabel('$T_{load}$ [K]')
        ax.set_title('Noise')
        ax2.set_title('NET')
        canvas = FigureCanvasAgg(fig)
        pdf.savefig(fig,bbox_inches='tight')
        allpdf.savefig(fig,bbox_inches='tight')
        pdf.close()
        np.savez(npzname, T=T,Tphys=Tphys, f0s = f0s,f0errs=f0errs,Hz_per_K=Hz_per_K,
                 Tn = np.array(Tn), dev_noise = np.array(dev_noise), amp_noise = np.array(amp_noise),
                 Q = Q, Qi = Qi)
    allpdf.close()


def plot_all_net_2014_03_al_pkg(pkls= glob.glob('/home/data/noise_2014-03-2*.pkl')):
    pkls.sort()
    res = defaultdict(list)
    for pkl in pkls:
        print pkl
        nms = load_net_pkl(pkl)
        if True: #nms[0].atten == 10.0:
            for idx in range(20):
                thisidx = [nm for nm in nms if nm.index == idx]
                #print "index",idx,"records",len(thisidx)
                if len(thisidx):
                    res[idx].append(thisidx[0])
    print "loaded all data"
    chipfname = res[0][0].chip.replace(' ','_')
    allpdf = PdfPages('/home/data/plots/summary_net_%s.pdf' % (chipfname))
    for ridx in range(20):
        print "processing resonator",ridx
        res0 = res[ridx]
        chipfname = res0[0].chip.replace(' ','_')
        pdf = PdfPages('/home/data/plots/summary_net_%s_%02d.pdf' % (chipfname,ridx))
        npzname = '/home/data/archive/summary_net_%s_%02d.npz' % (chipfname,ridx)
        T = np.array([nm.ts_temp for nm in res0])
        far_temp = np.array([nm.ts_far_temp for nm in res0])
        Q = np.array([nm.params['Q'].value for nm in res0])
        Qi = np.array([nm.Q_i for nm in res0])
        Tphys = np.array([nm.end_temp for nm in res0])
        f0s= np.array([nm.params['f_0'].value for nm in res0])
        f0errs= np.array([nm.params['f_0'].stderr for nm in res0])
        
        title = '%s\n%02d - %f' % (res0[0].chip, ridx, np.median(f0s))

        if ((T<8).sum()) <1:
            continue
        f0 = f0s[T<8.5].min()
        df0=(f0s-f0)*1e6
        msk = (f0errs < 0.00005) & (Tphys < 0.32) & (Tphys > 0.18)
        if msk.sum() < 1:
            continue
        df0 = df0[msk]
        f0s = f0s[msk]
        f0errs = f0errs[msk]
        T = T[msk]
        Tphys = Tphys[msk]
        Q = Q[msk]
        Qi = Qi[msk]
        far_temp = far_temp[msk]
        try:
            pp = np.polyfit(T[T<8],df0[T<8],1)
        except:
            continue
        Hz_per_K = abs(pp[0])
        print Hz_per_K,"Hz/K"
        pp_far = np.polyfit(far_temp[far_temp<8],df0[far_temp<8],1)
        Hz_per_K_far = abs(pp_far[0])

        fig = Figure(figsize=(11,8))
        ax = fig.add_subplot(221)
        fig.suptitle(title,size='small')
        ax.errorbar(T,df0,yerr = f0errs*1e6,linestyle='none',marker='.')
        ax.errorbar(far_temp,df0,yerr = f0errs*1e6,linestyle='none',marker='.')
        Tm = np.linspace(3,12,100)
        ax.plot(Tm,np.polyval(pp,Tm),label=('%.1f Hz/K copper'% pp[0]))
        ax.plot(Tm,np.polyval(pp_far,Tm),label=('%.1f Hz/K absorber side'% pp_far[0]))
        ax.set_xlabel('$T_{load}$ [K]')
        ax.set_ylabel('Frequency shift [Hz]')
        ax.legend(loc='upper right',prop=dict(size='x-small'))

        ax3 = fig.add_subplot(223)
        ax3.plot(T,Qi,'o',label='Qi')
        ax3.plot(T,Q,'x',mew=2,label='Qr')
        ax3.set_ylim(0,3e5)
        ax3.set_ylabel('Quality factor')
        ax3.set_xlabel('$T_{load}$ [K]')
        ax3.legend(loc='upper right',prop=dict(size='small'))


        ax = fig.add_subplot(222)
        ax2 = fig.add_subplot(224,sharex=ax)
        Tn = []
        dev_noise = []
        amp_noise = []
        for k,nm in enumerate(res0):
            if msk[k]:
                idx = (nm.pca_fr > 150)&(nm.pca_fr < 240)
                dev = (np.sqrt(nm.pca_evals[1,idx])*nm.f0*1e6).mean()
                amp = (np.sqrt(nm.pca_evals[1,-10:])*nm.f0*1e6).mean()
                Tn.append(nm.ts_temp)
                dev_noise.append(dev)
                amp_noise.append(amp)
                ax.plot(nm.ts_temp,dev,'b.')
                ax.plot(nm.ts_temp,amp,'r.')
                ax.plot(nm.ts_temp,dev-amp,'g.')
                uKrtHz = (dev*1e6/Hz_per_K)
                uKrts = uKrtHz/np.sqrt(2)
                uKrtHz_far = (dev*1e6/Hz_per_K_far)
                uKrts_far = uKrtHz_far/np.sqrt(2)
                
                ax2.plot(nm.ts_temp,uKrts,'b.')
                ax2.plot(nm.ts_far_temp,uKrts_far,'rx',mew=2)
        ax.set_ylim(0,0.6)
        ax2.plot(T.min(),-1,'b.',label='Copper temp')
        ax2.plot(T.min(),-1,'rx',mew=2,label='Absorb temp')
        ax2.set_ylim(0,100)
        ax2.legend(loc='upper right',prop=dict(size='x-small'))
        ax.plot(T.min(),0,'b.',label='Device @ 150 Hz')
        ax.plot(T.min(),0,'r.',label='Amp')
        ax.plot(T.min(),0,'g.',label='Dev-Amp')
        #ax.set_xlim(3,T.max())
        ax.legend(loc='upper right',prop=dict(size='x-small'))
        ax2.grid()
        ax.grid()
        ax.set_ylabel('$Hz/\sqrt{Hz}$')
        ax2.set_ylabel('NET $\mu$K$\sqrt{s}$')
        ax2.set_xlabel('$T_{load}$ [K]')
        ax.set_title('Noise')
        ax2.set_title('NET')
        canvas = FigureCanvasAgg(fig)
        pdf.savefig(fig,bbox_inches='tight')
        allpdf.savefig(fig,bbox_inches='tight')
        pdf.close()
        np.savez(npzname, T=T,Tphys=Tphys, f0s = f0s,f0errs=f0errs,Hz_per_K=Hz_per_K,
                 Tn = np.array(Tn), dev_noise = np.array(dev_noise), amp_noise = np.array(amp_noise),
                 Q = Q, Qi = Qi)
    allpdf.close()
    return res

    
def plot_net_set(pkl_glob,expname):
    if type(pkl_glob) is str:
        pklnames = glob.glob(pkl_glob)
    else:
        pklnames = pkl_glob
    pklnames.sort()
    nms = load_net_pkl(pklnames[0])
    nres = len(set([nm.index for nm in nms]))
    ntemps = len(pklnames)
    chipfname = nms[0].chip.replace(' ','_')
    exppdf = PdfPages('/home/data/plots/net_%s_%s.pdf' % (expname,chipfname))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.spectral(np.linspace(0.1,0.9,ntemps+1))
    respdfs = {}
    resfigs = {}
    resfigs2 = {}
    resfigs3 = {}
    nets = np.zeros((ntemps,nres))
    temps = np.zeros((ntemps,))
    spectra = defaultdict(list)
    tidx = 0
    for pklname in pklnames:
        nms = load_net_pkl(pklname)
        for idx in range(nres):
            if not respdfs.has_key(idx):
                respdfs[idx] = PdfPages('/home/data/plots/net_%s_%s_%02d.pdf' % (expname,chipfname,idx))
            rset = [nm for nm in nms if nm.index == idx]
            fig = Figure(figsize=(16,8))
            K_per_Hz,dt,df,ukrs = plot_net_sweep(rset,fig=fig)
            fig.suptitle('%s\n%02d - %f MHz' % (pklname,idx,rset[0].f0))
            canvas = FigureCanvasAgg(fig)
            fig.set_canvas(canvas)
            try:
                respdfs[idx].savefig(fig,bbox_inches='tight')
            except:
                raise
                print "Error with index",idx,pklname
            nm0 = rset[0]
            temps[tidx] = nm0.ts_temp
            nets[tidx,idx] = ukrs
            if not resfigs.has_key(idx):
                resfigs[idx] = Figure(figsize=(16,8))
                resfigs[idx].add_subplot(111)
                resfigs2[idx] = Figure(figsize=(16,8))
                resfigs2[idx].add_subplot(111)
                resfigs3[idx] = Figure(figsize=(16,8))
                resfigs3[idx].add_subplot(111)
            ax = resfigs[idx].axes[0]
            if np.isfinite(K_per_Hz) and abs(K_per_Hz) > 0:
                ax.semilogx(nm0.pca_fr,np.sqrt(nm0.pca_evals[1,:])*1e6*nm0.f0*abs(K_per_Hz)*1e6/np.sqrt(2),label=('%.3f K' % nm0.ts_temp),color=colors[tidx])
                ax.set_ylim(0,100)
                ax.grid(True)
                spectra[idx].append((nm0.pca_fr,np.sqrt(nm0.pca_evals[1,:])*1e6*nm0.f0*abs(K_per_Hz)*1e6/np.sqrt(2)))
            ax = resfigs2[idx].axes[0]
            ax.plot(nms[idx].frm,20*np.log10(abs(nms[idx].s21m+nms[idx].s0)),color=plt.cm.spectral(nms[idx].ts_temp/10.0))
            ax.plot(nms[idx].fr,20*np.log10(abs(nms[idx].s21)),'.',color=plt.cm.spectral(nms[idx].ts_temp/10.0))
            ax = resfigs3[idx].axes[0]
            ax.semilogx(nm0.pca_fr,np.sqrt(nm0.pca_evals[1,:])*1e6*nm0.f0,label=('%.3f K' % nm0.ts_temp),color=colors[tidx])
            ax.set_ylim(0,0.5)
            ax.grid(True)
        tidx += 1
    for idx,fig in resfigs.items():
        ax = fig.axes[0]
        ax.legend(loc='lower left', prop=dict(size='x-small'))
        canvas = FigureCanvasAgg(fig)
        fig.set_canvas(canvas)
        respdfs[idx].savefig(fig,bbox_inches='tight')
        canvas = FigureCanvasAgg(resfigs2[idx])
        resfigs2[idx].set_canvas(canvas)
        respdfs[idx].savefig(resfigs2[idx],bbox_inches='tight')
        canvas = FigureCanvasAgg(resfigs3[idx])
        resfigs3[idx].set_canvas(canvas)
        respdfs[idx].savefig(resfigs3[idx],bbox_inches='tight')
        respdfs[idx].close()
    
    return temps,nets,spectra



def plot_net_sweep(nms,fig = None):
    if fig is None:
        fig = plt.figure(figsize=(16,8))
    fig.subplots_adjust(wspace=0.3,hspace=0.25)
    ax1 = fig.add_subplot(121)
    ax1.plot(nms[0].s21.real,nms[0].s21.imag,'.-')
    s21m = nms[0].s21m + nms[0].s0
    ax1.plot(s21m.real,s21m.imag)
    colors = plt.cm.spectral(np.linspace(0.1,0.9,len(nms)))
    for k,nm in enumerate(nms):
        tslc = nm.tsl_raw*np.exp(-2j*np.pi*(nm.delay*nm.f0 - nms[0].delay*nms[0].f0))
        ax1.plot(tslc.real,tslc.imag,'.', color=colors[k], label=("%.3f K" % nm.ts_temp))
    ax1.plot(nms[0].sres.real,nms[0].sres.imag,'kx',markersize=20,mew=2)
    ax1.legend(loc='lower left',prop=dict(size='x-small'),ncol=4)
    ylim = ax1.get_ylim()
    stretch = (ylim[1]-ylim[0])*0.15/2.
    ax1.set_ylim(ylim[0]-stretch,ylim[1] + stretch)
    
    ax2 = fig.add_subplot(222)
    ax2b = ax2.twinx()
    ax3 = fig.add_subplot(224)
    hz0 = None
    offs_hz = []
    rr0 = kid_readout.analysis.resonator.Resonator(f=nms[0].frm,data=nms[0].s21m)
    for k,nm in enumerate(nms):
        ax2.loglog(nm.pca_fr,np.sqrt(nm.pca_evals[1,:]*1e12*nm.f0**2),color=colors[k],label=("%.3f K" % nm.ts_temp))
        ax2.loglog(nm.pca_fr,np.sqrt(nm.pca_evals[0,:]*1e12*nm.f0**2),':',color=colors[k])
        if nm.params['f_0'].stderr*1e6 < 100:
            ax3.errorbar(nm.ts_temp,(nm.params['f_0'].value-nms[0].params['f_0'].value)*1e6,nm.params['f_0'].stderr*1e6,color=colors[k],mew=2)
        tslc = nm.tsl_raw*np.exp(-2j*np.pi*(nm.delay*nm.f0 - nms[0].delay*nms[0].f0))
        hz = nms[0].frm[abs(s21m-tslc.mean()).argmin()]
        hz = rr0.inverse(tslc.mean(), params=nms[0].params,guess = hz)
        if hz0:
            hz = hz -hz0
        else:
            hz0 = hz
            hz = hz - hz0
        offs_hz.append(hz*1e6)
        ax3.plot(nm.ts_temp,-hz*1e6,'x',color=colors[k],mew=2)

    ax3.set_xlabel("Temperature [K]")
    ax3.set_ylabel("Frequency shift [Hz]")
    ax2.set_xlim(nm.pca_fr[2],5e4)
    ax2.set_ylabel(r'$Hz/\sqrt{Hz}$')
    ax2.set_xlabel('Hz')
    ax2b.set_xlim(nm.pca_fr[2],5e4)
#    ax2b.set_xscale('log')
    ax2b.set_yscale('log')
    ylim = ax2.get_ylim()
    dt = nms[-1].ts_temp - nms[0].ts_temp
    K_per_Hz = abs(dt)/abs(offs_hz[-1])
    print K_per_Hz,dt,offs_hz[-1]
    ax2b.set_ylim(K_per_Hz*ylim[0]*1e6/np.sqrt(2),K_per_Hz*ylim[1]*1e6/np.sqrt(2))
    ax2b.grid(color='r')
    ax2b.set_ylabel(r'$\mu K \sqrt{s}$')
    idx250 = (nm.pca_fr > 100) & (nm.pca_fr < 240)
    ukrs = K_per_Hz*np.sqrt(nms[0].pca_evals[1,idx250])*1e6*nms[0].f0*1e6/np.sqrt(2)
    ukrs = ukrs.mean()
    ax2.annotate((r"%.2f $\mu K \sqrt{s}$ @ 250 Hz" % ukrs),xy=(250,np.sqrt(nms[0].pca_evals[1,idx250].mean())*1e6*nms[0].f0),xycoords='data',xytext=(-15,-30),textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))

    return K_per_Hz,dt,offs_hz[-1],ukrs

#initial files here are not in standard experiment setup
['noise_2014-03-01_145313.pkl',
 'noise_2014-03-01_150116.pkl',
 'noise_2014-03-01_150242.pkl',
 'noise_2014-03-01_150411.pkl',
 'noise_2014-03-01_150548.pkl',
 'noise_2014-03-01_150721.pkl',
 'noise_2014-03-01_150903.pkl',
 'noise_2014-03-01_151049.pkl',
 'noise_2014-03-01_151230.pkl',
 'noise_2014-03-01_152549.pkl',
 'noise_2014-03-01_152951.pkl',
 'noise_2014-03-01_153356.pkl',
 'noise_2014-03-01_153804.pkl',
 'noise_2014-03-01_154217.pkl',
 'noise_2014-03-01_161404.pkl',
 'noise_2014-03-01_161809.pkl',
 'noise_2014-03-01_162219.pkl',
 'noise_2014-03-01_162631.pkl',
 'noise_2014-03-01_163046.pkl',
 'noise_2014-03-01_165758.pkl',
 'noise_2014-03-01_170209.pkl',
 'noise_2014-03-01_170623.pkl',
 'noise_2014-03-01_171033.pkl',
 'noise_2014-03-01_171447.pkl',
 'noise_2014-03-01_174536.pkl',
 'noise_2014-03-01_174944.pkl',
 'noise_2014-03-01_175356.pkl',
 'noise_2014-03-01_175808.pkl',
 'noise_2014-03-01_180224.pkl',
 'noise_2014-03-01_192106.pkl',
 'noise_2014-03-01_193103.pkl',
 ]
#log steps from 3.27 to 9.4 K
[
 'noise_2014-03-01_193754.pkl',
 'noise_2014-03-01_195704.pkl',
 'noise_2014-03-01_201504.pkl',
 'noise_2014-03-01_203253.pkl',
 'noise_2014-03-01_205110.pkl',
 'noise_2014-03-01_210944.pkl',
 'noise_2014-03-01_212848.pkl',
 'noise_2014-03-01_214806.pkl',
 'noise_2014-03-01_220748.pkl',
 'noise_2014-03-01_222543.pkl',
 ]
#log steps from 3.27 to 7 K. 
[
 'noise_2014-03-02_013752.pkl',
 'noise_2014-03-02_015704.pkl',
 'noise_2014-03-02_021653.pkl',
 'noise_2014-03-02_023646.pkl',
 'noise_2014-03-02_025703.pkl',
 'noise_2014-03-02_031743.pkl',
 'noise_2014-03-02_033849.pkl',
 #regulation ran out about here, so last data proabbly no good
# 'noise_2014-03-02_040016.pkl',
 ]
#quadratic steps from 3.27 to 4.96 K
[
 'noise_2014-03-02_083802.pkl',
 'noise_2014-03-02_085710.pkl',
 'noise_2014-03-02_091638.pkl',
 'noise_2014-03-02_093622.pkl',
 'noise_2014-03-02_095646.pkl',
 'noise_2014-03-02_101727.pkl',
 'noise_2014-03-02_103644.pkl',
 'noise_2014-03-02_105602.pkl',
 'noise_2014-03-02_111553.pkl',]
#linear steps from 3.3 to 4.1 K
[
 'noise_2014-03-02_144155.pkl',
 'noise_2014-03-02_150116.pkl',
 'noise_2014-03-02_152049.pkl',
 'noise_2014-03-02_154045.pkl',
 'noise_2014-03-02_160101.pkl',
 'noise_2014-03-02_162142.pkl',
 'noise_2014-03-02_164254.pkl',
 'noise_2014-03-02_170221.pkl',
 'noise_2014-03-02_172211.pkl',
 ]
#linear steps from 3.3 to 4.2 K
[
 'noise_2014-03-02_191654.pkl',
 'noise_2014-03-02_193805.pkl',
 'noise_2014-03-02_195930.pkl',
 'noise_2014-03-02_202110.pkl',
 'noise_2014-03-02_204317.pkl',
 'noise_2014-03-02_210550.pkl',
 'noise_2014-03-02_212850.pkl',
 'noise_2014-03-02_215022.pkl',
 'noise_2014-03-02_221212.pkl',
 'noise_2014-03-02_223435.pkl',
 ]
#linear steps from 3.3 to 4.2 K. bifurcation?? higher drive power?
[
 'noise_2014-03-03_001041.pkl',
 'noise_2014-03-03_003205.pkl',
 'noise_2014-03-03_005332.pkl',
 'noise_2014-03-03_011526.pkl',
 'noise_2014-03-03_013753.pkl',
 'noise_2014-03-03_020042.pkl',
 'noise_2014-03-03_022147.pkl',
 'noise_2014-03-03_024316.pkl',
 'noise_2014-03-03_030507.pkl',
 'noise_2014-03-03_032732.pkl',
 ]
#3.27 to 3.65 K, long settle time
[
 'noise_2014-03-05_215259.pkl',
 'noise_2014-03-05_224530.pkl',
 'noise_2014-03-05_230135.pkl',
 'noise_2014-03-05_235433.pkl',
 'noise_2014-03-06_004622.pkl',
#pkg starts heating up after 01:40
 'noise_2014-03-06_013833.pkl',
 'noise_2014-03-06_023122.pkl',
 'noise_2014-03-06_032437.pkl',
 ]
#3.27 to 3.65 K, good settle time, 3 cycles
[
 'noise_2014-03-06_081955.pkl',
 'noise_2014-03-06_090050.pkl',
 'noise_2014-03-06_094002.pkl',
 'noise_2014-03-06_102012.pkl',
 #data is bad in this last one, don't remember what happened
# 'noise_2014-03-06_110017.pkl',
 ]
# this one looks like an orphan
[
 'noise_2014-03-06_115419.pkl',
 ]
#3.27 to 3.65 K. ended early at start of 2nd cycle for some reason
[
 'noise_2014-03-06_144332.pkl',
 'noise_2014-03-06_151458.pkl',
 'noise_2014-03-06_162413.pkl',
 ]
# 3.27 to 3.37 K. Initially regulated at 200 mK or something accidentlally, so first data probably no good
[
 'noise_2014-03-06_190907.pkl',
 'noise_2014-03-06_194038.pkl',
 'noise_2014-03-06_201224.pkl',
 #temperature of package is corrected right about here
 'noise_2014-03-06_204448.pkl',
 'noise_2014-03-06_211519.pkl',
 'noise_2014-03-06_214642.pkl',
 'noise_2014-03-06_221826.pkl',
 'noise_2014-03-06_225029.pkl',
 ]
# 3.27 to 3.37 K 4 cycles, seems pretty good.
[
 'noise_2014-03-07_002848.pkl',
 'noise_2014-03-07_010038.pkl',
 'noise_2014-03-07_013250.pkl',
 'noise_2014-03-07_020528.pkl',
 'noise_2014-03-07_023613.pkl',
 'noise_2014-03-07_030741.pkl',
 'noise_2014-03-07_033929.pkl',
 'noise_2014-03-07_041127.pkl',
 ]
# 3.27 to 3.37 K 4 cycles, seems ok, some diode glitches
[
 'noise_2014-03-07_165730.pkl',
 'noise_2014-03-07_171933.pkl',
 'noise_2014-03-07_174151.pkl',
 'noise_2014-03-07_180437.pkl',
 'noise_2014-03-07_182605.pkl',
 'noise_2014-03-07_184747.pkl',
 'noise_2014-03-07_190947.pkl',
 'noise_2014-03-07_193210.pkl',
 ]
#3.27 then 5.15 to 5.2 K approx. 3 ish cycles
[
 'noise_2014-03-07_211413.pkl',
 'noise_2014-03-07_213930.pkl',
 'noise_2014-03-07_220459.pkl',
 'noise_2014-03-07_223054.pkl',
 'noise_2014-03-07_225505.pkl',
 'noise_2014-03-07_232009.pkl',
 'noise_2014-03-07_234548.pkl',
 'noise_2014-03-08_001128.pkl',
 ]
#3.27 to 3.37 K 3 cycles, seems a bit short on settling time
[
 'noise_2014-03-08_040236.pkl',
 'noise_2014-03-08_042711.pkl',
 'noise_2014-03-08_045159.pkl',
 'noise_2014-03-08_051557.pkl',
 'noise_2014-03-08_054020.pkl',
 'noise_2014-03-08_060511.pkl',
 ]
#3.27 to 3.37 K 3 cycles, seems a bit short on settling time. diode glitch in middle
[
 'noise_2014-03-08_111542.pkl',
 'noise_2014-03-08_114002.pkl',
 'noise_2014-03-08_120450.pkl',
 'noise_2014-03-08_122849.pkl',
 'noise_2014-03-08_125311.pkl',
 'noise_2014-03-08_131822.pkl',
 ]
#3.3 to 3.4 K 3 cycles. diode glitch in middle. good settle
[
 'noise_2014-03-09_164941.pkl',
 'noise_2014-03-09_171357.pkl',
 'noise_2014-03-09_173845.pkl',
 'noise_2014-03-09_180240.pkl',
 'noise_2014-03-09_182710.pkl',
 'noise_2014-03-09_185204.pkl',
 ]
# This one seems like junkish
# 'noise_2014-03-09_193953.pkl',
#3.3 to 3.4K good settle.
[
 'noise_2014-03-09_212446.pkl',
 'noise_2014-03-09_215238.pkl',
 'noise_2014-03-09_222021.pkl',
 'noise_2014-03-09_224714.pkl',
 'noise_2014-03-09_231447.pkl',
 'noise_2014-03-09_234308.pkl']


"""
for idx in [0]:
#    pdf = PdfPages('/home/data/archive/%s_Resonator_%d.pdf' % ('StarCryo_3x3_0813f5_HPD_Dark_1',idx))
    res = arch[arch.ridx==idx]
    f0max = res.f_0.max()
    
    title = '%s %d ~%.6f MHz' % (arch.chip_name[0],idx,f0max)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    frac_f0 = res.f_0/f0max-1
    ax.plot(res.load_temp,frac_f0,'.-')
    ax.grid()
#    ax.set_xlim(0,700)
    ax.legend(prop=dict(size='small'))
    ax.set_title(title,size='small')
    ax.set_xlabel('Load Temperature (K)')
    ax.set_ylabel('Fractional frequency shift')
#    pdf.savefig(fig,bbox_inches='tight')
#    ax.set_ylim(-1e-4,0)
#    pdf.savefig(fig,bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(res.load_temp,res.Q_i,'.-')
    ax.grid()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.spectral(np.linspace(0.1,0.9,res.shape[0]))
    for k in range(res.shape[0]):
        ax.plot(res.fr.iat[k],20*np.log10(abs(res.s21.iat[k])),'.-')
"""