# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:05:26 2013

@author: nan
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit

__all__ = ['model_s21', 'residual_func', 'fitdata', 'fit_sweep','report']

def model_s21(f,params):
    f0 = params['f0'].value
    Q = params['Q'].value
    Qer = params['Qer'].value
    Qei = params['Qei'].value
    Cc = params['Cc'].value
    theta = params['theta'].value
    
    Qe = Qer + 1j*Qei    
    Zin = 50.0
    Zout = 50.0
    wo = f0*2*np.pi
    w=f*2*np.pi
    eh = -1 + 2/(1+(1j*w*Cc+(1/Zout))*Zin)
    s21 = np.exp(-1j*theta)*(1 + eh)*(1 - Q*(Qe**-1)/(1 + 2*1j*Q*(w - wo)/wo))
    
    return s21


def residual_func(params,f,measured_s21):
    modelled_s21 = model_s21(f,params)
    residual = np.abs(measured_s21-modelled_s21)
    return residual

def fitdata(residual_func,f,params,measured_s21):
    minimize(residual_func,params,args=(f,measured_s21))
    return params

def fit_sweep(f,measured_s21,f0,Q,Qer,Qei,Cc,theta): #,phi,t):
    params = Parameters()
    params.add('f0', value=f0,min=0.99*f0,max=1.01*f0)
    params.add('Q', value=Q,min=0,max=1e7)
    params.add('Qer', value=Qer,min=-1e7,max=1e7)
    params.add('Qei', value=Qei,min=-1e7,max=1e7)
    params.add('Cc', value=Cc,min=0,max = 1e-8)
    params.add('theta', value=theta,min=-np.pi, max=np.pi)
    #params.add('t', value=t)
   
    params2=fitdata(residual_func, f, params, measured_s21)
    return params2

def report(f, measured_s21,f0,Q,Qer,Qei,Cc,theta): #, phi, t):
    sweep = fit_sweep(f,measured_s21,f0,Q,Qer,Qei,Cc,theta) #,phi,t)
    report = report_fit(sweep)

    s21 = model_s21(f,sweep)

    plt.figure(1)
    #plt.subplot(211)
    plt.plot(f,np.abs(s21),'r--')
    plt.plot(f,np.abs(measured_s21),'b')

    plt.figure(2)
    #plt.subplot(212)
    plt.plot(f,np.imag(s21),'r--')
    plt.plot(f,np.imag(measured_s21),'b')
    plt.show()
    
    print sweep
    return report