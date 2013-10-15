import numpy as np

from matplotlib import pyplot as plt

def extract_fine_sweeps(fr,s21):
    deltas = np.diff(fr)
    ends = np.concatenate(([0],np.where(deltas>0.01)[0],[fr.shape[0]-1]))
    nres = len(ends)-1
    ptsperres = np.ceil(len(fr)/float(nres))
    frout = np.zeros((nres,ptsperres))
    s21out = np.zeros((nres,ptsperres),dtype='complex128')
    f0s = np.zeros((nres,))
    for k in range(nres):
        thisfr = fr[ends[k]:ends[k+1]]
        thiss21 = s21[ends[k]:ends[k+1]]
        thislen = len(thisfr)
        frout[k,:thislen] = thisfr
        s21out[k,:thislen] = thiss21
        f0s[k] = thisfr[np.abs(thiss21).argmin()]
    return frout,s21out,f0s 
    