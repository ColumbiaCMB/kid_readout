# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 23:03:10 2013

@author: nan

Zin = Zout =50.0

feed frequency as freq (Hz) and s21 as measured_s21(normalized)

import sweep_est
import kfit
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

cd 'Downloads'
data = np.load('sweep_20131016130211.npz')
freqs=data['freq']
s21s=data['s21']
freq = freqs[0,:]*1e6 # in Hz
measured_s21 = s21s[0,:]

sweep_est.scale(freq,measured_s21) # optional

s21_bw =
Q = 
Qer = 
Qei =

measured_s21a,f0est,Q,Qer,Qei,Cc,theta = sweep_est.est_params(freq,measured_s21,s21_bw,Q,Qer,Qei)
kfit.report(freq,measured_s21a,f0est,Q,Qer,Qei,Cc, theta)
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math


def scale(freq,measured_s21):
    
    measured_s21a= measured_s21/measured_s21[-1]
    plt.plot(freq,np.abs(measured_s21a))
    plt.show()

def est_params(freq,measured_s21,s21_bw,Q,Qer,Qei):
    
    measured_s21a= measured_s21/measured_s21[-1]
    
    f0est = freq[np.argmin(np.abs(measured_s21a))]
    rr = freq[np.abs(measured_s21a)<s21_bw]
    df = rr[-1]-rr[0]
    Qest = f0est/df

    s21min = np.abs(measured_s21a).min()
    Rest= 50/(2/s21min-2)
    Lest = Rest/(2*np.pi*f0est*Qest)
    Cest = 1/(Lest*(2*np.pi*f0est)**2)
    Cc = Cest*1e-5
    wo = 2*np.pi*f0est
    w = 2*np.pi*freq
    Zout = 50
    Zin = 50   
    
    Qe=Qer+1j*Qei
    eh = -1 + 2/(1+(1j*w*Cc+(1/Zout))*Zin)
    s21 = (1 + eh)*(1 - Q*(Qe**-1)/(1 + 2*1j*Q*(w - wo)/wo))
    theta=math.asin(np.imag(measured_s21[0])/np.imag(s21[0]))*math.sin(np.imag(s21)[0])-np.imag(s21)[0]
    s21=s21*np.exp(-1j*theta)
    
    plt.figure(1)
    plt.plot(freq,np.abs(s21),'r--',freq,np.abs(measured_s21a),'b')
    plt.figure(2)
    plt.plot(freq,np.imag(measured_s21a),'r--',freq,np.imag(s21),'b')
    
    return measured_s21a,f0est,Q,Qer,Qei,Cc,theta
    
    
