# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:13:45 2013

@author: nan
"""
import numpy as np

def s21(w,Cc,Zout,Zin,Q,Qer,Qei,wo):
    
    Qe=Qer+1j*Qei
    eh = -1 + 2/(1+(1j*w*Cc*1e-12+(1/Zout))*Zin)
    s21 = (1 + eh)*(1 - Q*(Qe**-1)/(1 + 2*1j*Q*(w - wo)/wo))
    return s21

def s21m(w,Cc,Zout,Zin,Q,Qer,Qei,wo,A,theta,t):
    
    Qe=Qer+1j*Qei
    eh = -1 + 2/(1+(1j*w*Cc*1e-12+(1/Zout))*Zin)
    s21m = A*np.exp((-1j)*(theta+t*w))*(1 + eh)*(1 - Q*(Qe**-1)/(1 + 2*1j*Q*(w - wo)/wo))
    return s21m