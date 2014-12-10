from __future__ import division
import numpy as np
from lmfit import Parameters
from matplotlib import pyplot as plt


def pkl_mask(f,data):
    index =np.concatenate((np.arange(0,10),np.arange(10,50,2),np.arange(50,120,3),np.arange(120,160,5),np.arange(160,len(f),7)))
    f_masked=f[index]
    data_masked=data[index]
    return f_masked,data_masked

def guess(A,B,alpha,beta,fc,i,N_white=1e-3,va=True,vb=True,vi=True):
    p = Parameters()
    p.add('A', value = A, min=0)
    p.add('B',value = B,min=0)
    p.add('N_white',value = N_white,min=0, max=1)#2e-2)
    p.add('alpha',value = alpha,min = -7, max=0,vary=va)
    p.add('beta',value = beta,min=-2, max=0,vary=vb)
    p.add('fc',value = fc, min=1e3, max=1e6)
    p.add('i',value = i, min=0, max=6,vary=vi)       
    return p

def model(f,p):
    A = p['A'].value
    B = p['B'].value
    N_white = p['N_white'].value
    alpha = p['alpha'].value
    beta = p['beta'].value
    fc = p['fc'].value
    i = p['i'].value
    P1 = A*f**alpha
    P2 = B*f**beta
    P3 = (1/abs(1+1j*f/fc)**i)         
    return ((P1 + P2)*P3 + N_white)
    
def crop_data(f,pxx,front_crop=1,end_crop=2e4):
    f = f[front_crop:]    
    pxx = pxx[front_crop:]
    f =np.array([j for j in f if j<=end_crop])
    pxx = pxx[0:len(f)] 
    return f,pxx

def plot_single_pkl_noise(n,pkls_name=None,single_pkl_no=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(n.f,n.p,'g',lw=2)
    ax.loglog(n.f,n.model(n.result.params),'r',lw=2)
    ax.set_xlim([n.f[0],n.f[-1]])
    ax.set_xlabel('Hz')
    ax.set_ylabel('Hz$^2$/Hz')
    ax.grid()
    text = (("A: %.6f\n" % n.A)
            +("alpha: %.6f\n" % n.alpha)
            +("B: %.6f\n" % n.B)
            +("beta: %.6f\n" % n.beta)
            +("fc: %.6f\n" % n.fc)
            +("i: %.6f\n" % n.i)
            +("N_white: %.6f" % n.N_white)
            )
    ax.text(0.05,0.05,text,ha='left',va='bottom',bbox=dict(fc='white',alpha=0.6),transform = ax.transAxes,
            fontdict=dict(size='x-small'))
    if pkls_name is not None and single_pkl_no is not None:
         ax.set_title("%s, pkl no.%.0f" % (pkls_name,single_pkl_no),size='small')