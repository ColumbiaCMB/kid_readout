import numpy as np
from matplotlib import pyplot as plt
mlab = plt.mlab

def make_freq_bins(fr):
    scale = 2
    fmax = fr.max()
    fout = []
    fdiff = fr[1]-fr[0]
    fstart = int(fdiff*100)
    if fstart > 10:
        fstart = 10
    fstep = 1
    if fstart < 1:
        fstart = 1
        fstep = fdiff*2
    print fstart
    fout.append(fr[fr<fstart])
    ftop = scale*fstart
#    fstep = int(10**int(np.round(np.log10(fstart))-1))
#    if fstep < 1:
#        fstep = 1
    if fstep < fdiff:
        fstep = fdiff
    while True:
#        print ftop/10,fmax,fstep
        if ftop > fmax:
            fout.append(np.arange(ftop/scale,fmax,fstep))
            break
        else:
            fout.append(np.arange(ftop/scale,ftop,fstep))
        ftop *=scale
        fstep *=scale
    return np.concatenate(fout)

def log_bin(freqs,data):
    freq_bins = make_freq_bins(freqs)
    bin_idxs = np.digitize(freqs,freq_bins)
    if type(data) is list:
        binned_data = []
        for dunit in data:
            binned_data.append(np.array([dunit[bin_idxs==k].mean() for k in range(1,len(freq_bins))]))
    else:
        binned_data = np.array([data[bin_idxs==k].mean() for k in range(1,len(freq_bins))])  #skip the zeroth bin since it has nothing in it
    binned_freqs = np.array([freqs[bin_idxs==k].mean() for k in range(1,len(freq_bins))])
    return binned_freqs,binned_data

def pca_noise(d,NFFT=None,Fs=256e6/2.**11,window = mlab.window_hanning,detrend=mlab.detrend_mean, use_log_bins=True):
    if NFFT is None:
        NFFT = int(2**(np.floor(np.log2(d.shape[0]))-3))
        print "using NFFT: 2**",np.log2(NFFT)
    pii,fr_orig = mlab.psd(d.real,NFFT=NFFT,Fs=Fs,window=window,detrend=detrend)
    pqq,fr = mlab.psd(d.imag,NFFT=NFFT,Fs=Fs,window=window,detrend=detrend)
    piq,fr = mlab.csd(d.real,d.imag,NFFT=NFFT,Fs=Fs,window=window,detrend=detrend)
    if use_log_bins:
        fr,data = log_bin(fr,[pii,pqq,piq])
        pii,pqq,piq = data
    nf = pii.shape[0]
    evals = np.zeros((2,nf)) # since the matrix is hermetian, eigvals are real
    evects = np.zeros((2,2,nf),dtype='complex')
    for k in range(nf):
        m = np.array([[pii[k], np.real(piq[k])],[np.conj(np.real(piq[k])),pqq[k]]])
        w,v = np.linalg.eigh(m)
        evals[:,k] = w
        evects[:,:,k] = v
    angles = np.zeros((2,nf))
    angles[0,:] = np.mod(np.arctan2(evects[0,0,:].real,evects[1,0,:].real),np.pi)
    angles[1,:] = np.mod(np.arctan2(evects[0,1,:].real,evects[1,1,:].real),np.pi)
    S = np.zeros((2,nf))
    v = evects[:,:,0]
    invv = np.linalg.inv(v)
    for k in range(nf):
        m = np.array([[pii[k], piq[k]],[np.conj(piq[k]),pqq[k]]])
        ss = np.dot(np.dot(invv,m),v)
        S[0,k] = ss[0,0]
        S[1,k] = ss[1,1]
    return fr,S,evals,evects,angles,piq