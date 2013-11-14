
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt

import model

def scale(freq,measured_s21):
    
    measured_s21a= measured_s21/measured_s21[-1]
    plt.plot(freq,np.abs(measured_s21a))
    plt.show()

def est_params(freq,measured_s21,Q,Qer,Qei):
    
    A=np.abs(measured_s21[-1])
    f0est = freq[np.argmin(np.abs(measured_s21))]
    wo = 2*np.pi*f0est
    w = 2*np.pi*freq
    Zout = 50
    Zin = 50     
    Cc=0
    s21 = model.s21(w,Cc,Zout,Zin,Q,Qer,Qei,wo)  
    t = 0
    theta=math.asin(np.imag(measured_s21[0])/np.imag(s21[0])*math.sin(np.imag(s21)[0]))-np.imag(s21)[0] 
    print 'f0est', f0est
    print 'theta',theta
    print 'Cc', Cc
    print 't', t
    return f0est,Cc,theta,t,A
    