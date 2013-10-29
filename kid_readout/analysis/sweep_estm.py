# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:36:04 2013

@author: nan

Parameters
----------
freq :(Hz)
mlb : left bound for scale factor A
mrb : right bound for scale factor A
         --> call scale(freq,measured_s21) first to determine mlb, mrb
             or can plot manually
measured_s21_sc: scaled measured_s21
             
            

"""

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt

import model

def scale(freq,measured_s21):
    
    measured_s21a= measured_s21/measured_s21[-1]
    plt.plot(freq,np.abs(measured_s21a))
    plt.show()

def est_params(freq,measured_s21,mlb,mrb,s21_bw,Q,Qer,Qei):
    
    m = np.where(np.logical_or(freq>=mrb, freq<=mlb))
    mean=np.mean(np.abs(measured_s21[m]))
    measured_s21_sc= measured_s21 #/mean #measured_s21[-1]
    
    f0est = freq[np.argmin(np.abs(measured_s21_sc))]
    rr = freq[np.abs(measured_s21_sc)<s21_bw]
    df = rr[-1]-rr[0]
    Qest = f0est/df

    s21min = np.abs(measured_s21_sc).min()
    Rest= 50/(2/s21min-2)
    Lest = Rest/(2*np.pi*f0est*Qest)
    Cest = 1/(Lest*(2*np.pi*f0est)**2)
    Cc = Cest*1e-5*1e12
    wo = 2*np.pi*f0est
    w = 2*np.pi*freq
    Zout = 50
    Zin = 50   
    
    #Qe=Qer+1j*Qei
    #eh = -1 + 2/(1+(1j*w*Cc*1e-12+(1/Zout))*Zin)
    #s21 = (1 + eh)*(1 - Q*(Qe**-1)/(1 + 2*1j*Q*(w - wo)/wo))
    s21 = model.s21(w,Cc,Zout,Zin,Q,Qer,Qei,wo)
    
    #t=np.angle(np.abs(s21[-1]-s21[0]))/np.abs(freq[-1]-freq[0])
    t = 0
    A = mean
    theta=math.asin(np.imag(measured_s21[0])/np.imag(s21[0]))*math.sin(np.imag(s21)[0])-np.imag(s21)[0]
    #s21m = A*s21*np.exp((-1j)*(theta+t*w))
    s21m = model.s21m(w,Cc,Zout,Zin,Q,Qer,Qei,wo,A,theta,t)
    
    plt.figure(1)
    plt.plot(freq,np.abs(s21m),'r--',freq,np.abs(measured_s21_sc),'b')
    plt.figure(2)
    plt.plot(freq,np.imag(s21m),'r--',freq,np.imag(measured_s21_sc),'b')
    
    print 'f0est', f0est
    print 'Cest', Cest
    print 'Lest', Lest
    print 'Rest', Rest
    print 'theta',theta
    print 'Cc', Cc
    print 't', t
    print 'mean', mean
    
    return measured_s21_sc,f0est,Cc,theta,t,mean