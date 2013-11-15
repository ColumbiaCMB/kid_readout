# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:09:33 2013

@author: nan
Parameters:
    -------
    measured_s21 : scaled measured_s21
    Cc: (pF)
    A : mean +- 20%
"""
from __future__ import division
import numpy as np
from lmfit import minimize, Parameters
import model

__all__ = ['model_s21', 'residual_func', 'fitdata', 'fit_sweep','report']

def model_s21(f,params):
    f0 = params['f0'].value
    Q = params['Q'].value
    Qer = params['Qer'].value
    Qei = params['Qei'].value
    Cc = params['Cc'].value
    theta = params['theta'].value
    t = params['t'].value
    A = params['A'].value
      
    Zin = 50.0
    Zout = 50.0
    wo = f0*2*np.pi
    w=f*2*np.pi
    s21 = model.s21m(w,Cc,Zout,Zin,Q,Qer,Qei,wo,A,theta,t)
    return s21

def residual_func(params,f,measured_s21):
    modelled_s21 = model_s21(f,params)
    residual = np.abs(measured_s21-modelled_s21)
    return residual    

def fitdata(residual_func,f,params,measured_s21):
    minimize(residual_func,params,args=(f,measured_s21))
    return params

def fit_sweep(f,measured_s21,f0,Q,Qer,Qei,Cc,theta,t,mean,va = True,vc=True): #,phi,t):
    params = Parameters()
    params.add('f0', value=f0,min=0.99*f0,max=1.01*f0)
    params.add('Q', value=Q,min=0,max=1e7)
    params.add('Qer', value=Qer,min=-1e7,max=1e7)
    params.add('Qei', value=Qei,min=-1e7,max=1e7)
    params.add('Cc', value=Cc,min=0,max = 1e4,vary =vc)
    params.add('theta', value=theta,min=-np.pi, max=np.pi)
    params.add('t', value=t,min = -1e-6,max=1e-6)
    params.add('A',value =mean,min = mean*0.8,max=mean*1.2,vary = va)
   
    params2=fitdata(residual_func, f, params, measured_s21)
    return params2

def report(f, measured_s21,f0,Q,Qer,Qei,Cc,theta,t,mean,va=True,vc=True):
    sweep = fit_sweep(f,measured_s21,f0,Q,Qer,Qei,Cc,theta,t,mean,va=va,vc=vc)
    return sweep

