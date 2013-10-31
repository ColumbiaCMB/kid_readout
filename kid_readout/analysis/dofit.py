# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:59:20 2013

@author: nan

Parameters:
----------

    freq(Hz)
    measured_s21: normalized
    s21_bw: taken from the plot(freq,measured_s21), from 0 to 1. I normally 
            use 0.9
"""
import sweep_est
import kfit

def dofit(freq,measured_s21,Q,Qer,Qei,s21_bw):
    
    measured_s21a,f0est,Cc,theta = sweep_est.est_params(freq,measured_s21,s21_bw,Q,Qer,Qei)

    fit = kfit.report(freq,measured_s21a,f0est,Q,Qer,Qei,Cc, theta)
    
    return fit
    