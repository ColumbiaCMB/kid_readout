from __future__ import division
import numpy as np
from lmfit import minimize, Parameters

def pkl_mask(f,data):
    index =np.concatenate((np.arange(0,10),np.arange(10,50,2),np.arange(50,120,3),np.arange(120,160,5),np.arange(160,len(f),7)))
    f_masked=f[index]
    data_masked=data[index]
    return f_masked,data_masked

def guess(A,B,alp,bet,fc,i):
    p = Parameters()
    p.add('A', value = A, min =0)
    p.add('B',value = B,min =0)
    p.add('N_white',value =1e-3,min =0, max=2e-2)
    p.add('alpha',value = alp,min = -7, max =0)
    p.add('beta',value = bet,min=-2, max =0)
    p.add('fc',value = fc, min =1e3, max = 1e6)
    p.add('i',value = i, min =0, max = 6)        # max value is significant
    return p

def model(f,p):
    A = p['A'].value
    B = p['B'].value
    N_white = p['N_white'].value
    alp = p['alpha'].value
    bet = p['beta'].value
    fc = p['fc'].value
    i = p['i'].value
    P1 = A*f**alp
    P2 = B*f**bet
    P3 = (1/abs(1+1j*f/fc)**i)         
    return ((P1 + P2)*P3 + N_white)
    
