import numpy as np
from matplotlib import pyplot as plt
mlab = plt.mlab

def pca_noise(d,NFFT=1024,Fs=256e6/2.**11,window = mlab.window_hanning):
    pii,fr = mlab.psd(d.real,NFFT=NFFT,Fs=Fs,window=window)
    pqq,fr = mlab.psd(d.imag,NFFT=NFFT,Fs=Fs,window=window)
    piq,fr = mlab.csd(d.real,d.imag,NFFT=NFFT,Fs=Fs,window=window)
    nf = pii.shape[0]
    evals = np.zeros((2,nf)) # since the matrix is hermetian, eigvals are real
    evects = np.zeros((2,2,nf),dtype='complex')
    for k in range(nf):
        m = np.array([[pii[k], piq[k]],[np.conj(piq[k]),pqq[k]]])
        w,v = np.linalg.eigh(m)
        evals[:,k] = w
        evects[:,:,k] = v
    angles = np.zeros((2,nf))
    angles[0,:] = np.arctan2(evects[0,0,:].real,evects[1,0,:].real)
    angles[1,:] = np.arctan2(evects[0,1,:].real,evects[1,1,:].real)
    S = np.zeros((2,nf))
    v = evects[:,:,0]
    invv = np.linalg.inv(v)
    for k in range(nf):
        m = np.array([[pii[k], piq[k]],[np.conj(piq[k]),pqq[k]]])
        ss = np.dot(np.dot(invv,m),v)
        S[0,k] = ss[0,0]
        S[1,k] = ss[1,1]
    return fr,S,evals,evects,angles