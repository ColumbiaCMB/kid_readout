import numpy as np
import scipy.special

import lmfit

h = 6.626e-34 # J/s
hbar = 1.054571e-34 #J/s
kB = 1.38065e-23 #J/K
qC = 1.602e-19 # C
kBeV = kB/qC
Al_N0 = 1.72e10/qC

def sigma1(nqp,T,Delta,N0,f):
    xi = h*f/(2*kB*T)
    num = 2*Delta*nqp*np.sinh(xi)*scipy.special.k0(xi)
    den = h*f * N0 * np.sqrt(2*np.pi*kB*T*Delta)
    return num/den

def sigma2(nqp,T,Delta,N0,f):
    xi = h*f/(2*kB*T)
    term1 = 1 + np.sqrt(2*Delta/(np.pi*kB*T))*np.exp(-xi)*scipy.special.i0(xi)
    term2 = 1 - nqp*term1/(2*N0*Delta)
    return np.pi*Delta*term2/(h*f)

def nqp_thermal(T,Delta,N0):
    return 2*N0*Delta*scipy.special.k1(Delta/(kB*T))

def kinetic_inductance(nqp,T,Delta,N0,f,sigma_n,nsquares,thickness):
    return nsquares/(2*np.pi*f*sigma2(nqp+nqp_thermal(T,Delta,N0),
                                      T,Delta,N0,f)*sigma_n*thickness)

def f0(nqp,T,Delta,N0,f_nominal,sigma_n,nsquares,thickness,Lg,Cg,chi_L=0):
    Lk = kinetic_inductance(nqp,T,Delta,N0,f_nominal,sigma_n,nsquares,thickness)
    #print Lk
    Ltotal = Lk + Lg*(1+chi_L)
    return 1/(2*np.pi*np.sqrt(Cg*Ltotal))

def qi(nqp,T,Delta,N0,f_nominal,sigma_n,nsquares,thickness,Lg,Cg,chi_L=0):
    Lk = kinetic_inductance(nqp,T,Delta,N0,f_nominal,sigma_n,nsquares,thickness)
    #print Lk
    Ltotal = Lk + Lg*(1+chi_L)
    s1 = sigma1(nqp+nqp_thermal(T,Delta,N0),T,Delta,N0,f_nominal)*sigma_n
    s2 = sigma2(nqp+nqp_thermal(T,Delta,N0),T,Delta,N0,f_nominal)*sigma_n
    num = 2*np.pi*Ltotal*f_nominal
    den = (nsquares*s1/(s2**2*thickness))
    return num/den

def f0_qi(nqp,T,Delta,N0,f_nominal,sigma_n,nsquares,thickness,Lg,Cg,chi_L=0):
    Lk = kinetic_inductance(nqp,T,Delta,N0,f_nominal,sigma_n,nsquares,thickness)
    #print Lk
    Ltotal = Lk + Lg*(1+chi_L)
    s1 = sigma1(nqp+nqp_thermal(T,Delta,N0),T,Delta,N0,f_nominal)*sigma_n
    s2 = sigma2(nqp+nqp_thermal(T,Delta,N0),T,Delta,N0,f_nominal)*sigma_n
    num = 2*np.pi*Ltotal*f_nominal
    den = (nsquares*s1/(s2**2*thickness))
    qi = num/den
    f0 = 1/(2*np.pi*np.sqrt(Cg*Ltotal))
    return f0, qi

def guess_chi_L(segment_ids, params=None):
    if params is None:
        params = lmfit.Parameters()
    for id in segment_ids:
        params.add(('chi_L_%d' % id), value = 0, min = 0, max=1e-1)
    return params

def guess_nqp0(segment_ids, params=None):
    if params is None:
        params = lmfit.Parameters()
    for id in segment_ids:
        params.add(('nqp0_%d' % id), value = 2, min = 1, max=1e4)
    return params

def get_segment_params(segment_ids, params, unique_ids=None):
    if unique_ids is None:
        unique_ids = np.unique(segment_ids)
    chi_L = np.empty(segment_ids.shape)
    nqp0 = np.empty(segment_ids.shape)
    for id in unique_ids:
        chi_L[segment_ids==id] = params['chi_L_%d' % id].value
        nqp0[segment_ids==id] = params['nqp0_%d' % id].value
    return chi_L, nqp0

def dark_multisegment_params(f_nominal, sigma_n, nsquares, thickness, Lg, Cg, N0 = Al_N0):
    params = lmfit.Parameters()
    params.add('f_nominal',value=f_nominal,vary=False)
    params.add('sigma_n',value=sigma_n,vary=False)
    params.add('nsquares',value = nsquares, vary=False)
    params.add('thickness', value= thickness, vary=False)
    params.add('Lg',value = Lg, vary= False)
    params.add('Cg',value=Cg,vary=False)
    params.add('N0',value=N0,vary=False)
    return params

class DarkMultisegment(object):
    def __init__(self, data, Tc=1.46, sigma_n=1.2e7, nsquares=21000, thickness=20e-9, Cg=1.2e-11, Lg=53.3e-9,
                 qi_err_scale=1.0):
        self.data = data
        self.f_nominal = data.f_0.max()*1e6
        self.params = dark_multisegment_params(f_nominal=self.f_nominal,sigma_n=sigma_n, nsquares=nsquares,
                                               thickness=thickness, Cg=Cg, Lg=Lg)
        self.params.add('delta_meV',value=1.76*1.46*kB*1e3/qC, min = 1.76*1.0*kB*1e3/qC, max=1.76*2.0*kB*1e3/qC)
        self.unique_segment_ids = np.unique(data.segment)
        self.params = guess_chi_L(self.unique_segment_ids,params=self.params)
        self.params = guess_nqp0(self.unique_segment_ids,params=self.params)
        self.qi_err_scale = qi_err_scale

    def f0_qi(self, T, segment, nqp=0):
        chi_L, nqp0 = get_segment_params(segment, self.params, self.unique_segment_ids)
        f0,qi = f0_qi(nqp=nqp0+nqp,
                      T=T,Delta=self.params['delta_meV'].value*qC/1e3,
                      N0=self.params['N0'].value,
                      f_nominal=self.params['f_nominal'].value,
                      sigma_n=self.params['sigma_n'].value,
                      nsquares=self.params['nsquares'].value,
                      thickness=self.params['thickness'].value,
                      Lg=self.params['Lg'].value, Cg=self.params['Cg'].value,
                      chi_L=chi_L)
        return f0,qi

    def f0_qi_resid(self,params):
        self.params=params
#        print [(k,v.value) for (k,v) in params.items() if v.vary]
        model_f0,model_qi = self.f0_qi(self.data.sweep_primary_package_temperature,
                                       self.data.segment.values)
        data_f0 = self.data.f_0*1e6
        data_qi = self.data.Q_i
        f0_resid = (model_f0 - data_f0) / (self.data.f_0_err*1e6)
        qi_resid = (model_qi - data_qi) / (self.data.Q_i_err*self.qi_err_scale)
        return np.concatenate((f0_resid,qi_resid))

    def f0_logqi(self, T, segment, nqp=0):
        f0,qi = self.f0_qi(T,segment,nqp=nqp)
        return f0, np.log(qi)

    def f0_logqi_resid(self,params):
        self.params=params
#        print [(k,v.value) for (k,v) in params.items() if v.vary]
        model_f0,model_logqi = self.f0_logqi(self.data.sweep_primary_package_temperature,
                                       self.data.segment.values)
        data_f0 = self.data.f_0*1e6
        data_logqi = np.log(self.data.Q_i)
        f0_resid = (model_f0 - data_f0) / (self.data.f_0_err*1e6)
        logqi_resid = np.exp(model_logqi - data_logqi) / np.exp(np.log(self.data.Q_i_err*self.qi_err_scale)-data_logqi)
        return np.concatenate((f0_resid,logqi_resid))






def linear_power_law(x,break_point,scale,exponent):
    return scale * ((x / break_point + 1) ** exponent - 1)

def power_law(x,scale,exponent):
    return scale * x**exponent