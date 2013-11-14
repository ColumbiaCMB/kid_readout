"""
A = measured_s21[-1]
Cc = 0
"""

import sweep_estm2 as estm
import kfitm
import numpy as np
import matplotlib.pyplot as plt

def dofit(freq,measured_s21,Q,Qer,Qei,va=True,vc=False):
    
    f0est,Cc,theta,t,A = estm.est_params(freq,measured_s21,Q,Qer,Qei) 
    params = kfitm.report(freq,measured_s21,f0est,Q,Qer,Qei,Cc,theta,t,A,va=va,vc=vc)#,vc=False) 
    
    return params
    
def plotfit(freq,measured_s21,Q,Qer,Qei,va=True,vc=False):
    params = dofit(freq,measured_s21,Q,Qer,Qei,va=va,vc=vc)
    s21m = kfitm.model_s21(freq,params)    
    plt.figure(1)
    plt.plot(freq,np.abs(s21m),'r--',freq,np.abs(measured_s21),'b')
    plt.figure(2)
    plt.plot(freq,np.imag(s21m),'r--',freq,np.imag(measured_s21),'b')
    