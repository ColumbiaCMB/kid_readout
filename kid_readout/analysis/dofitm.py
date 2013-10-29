# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:29:50 2013

@author: nan

Parameters:
----------
Cc: (pF)
t : phase added
A : scale model towards data instead of scaling data to unity
"""

import sweep_estm as estm
import kfitm

def dofit(freq,measured_s21,mlb,mrb,Q,Qer,Qei,s21_bw,va=True):
    
    measured_s21a,f0est,Cc,theta,t,mean = estm.est_params(freq,measured_s21,mlb,mrb,s21_bw,Q,Qer,Qei)

    fit = kfitm.report(freq,measured_s21a,f0est,Q,Qer,Qei,Cc,theta,t,mean,va=True)
    
    return fit