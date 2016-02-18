import numpy as np
import scipy.special

import lmfit

h = 6.626e-34 # J/s
hbar = 1.054571e-34 #J/s
kB = 1.38065e-23 #J/K
qC = 1.602e-19 # C
kBeV = kB/qC


def sigmas(fres,Tphys,Tc):
    wres = fres*2*np.pi
    xi = hbar*wres/(2*kB*Tphys)
    Delta = 3.52*kB*Tc/2.0
    
    sigma1 = (((4*Delta) / (hbar*wres)) *
              np.exp(-Delta/(kB*Tphys)) *
              np.sinh(xi) *
              scipy.special.k0(xi))
    sigma2 = (((np.pi*Delta) / (hbar*wres)) *
              (1 - np.sqrt((2*np.pi*kB*Tphys) / Delta)*np.exp(-Delta/(kB*Tphys))
               - 2 * np.exp(-Delta/(kB*Tphys)) * np.exp(-xi) * scipy.special.i0(xi)))
    return sigma1,sigma2 
              
def s1s2(Tphys,Tc,fread):
    xi = h*fread/(2*kB*Tphys)
    Delta = 3.52*kB*Tc/2.0
    S1 = (2/np.pi)*np.sqrt(2*Delta/(np.pi*kB*Tphys))*np.sinh(xi)*scipy.special.k0(xi)
    S2 = 1 + np.sqrt(2*Delta/(np.pi*kB*Tphys))*np.exp(-xi)*scipy.special.i0(xi)
    return S1,S2

def T_nqp_equals_nqp0(Delta,N0,nqp0):
    """
    Find the temperature at which the thermally generated QP density equals the given nqp0
    
    Delta : eV
    Formula found by asking Wolfram Alpha to solve nqp(T) == nqp0 for T
    """
    
    return np.real((2*Delta) /
            (kBeV * scipy.special.lambertw(16 * Delta**2 * N0**2 * np.pi / nqp0)))

class KIDModel(object):
    def __init__(self,Tc=1.46, nqp0=5000, Tbase=0.25, f0_nom = 150e6,
                 sigman=5e5, ind_l_um=46750.0, ind_w_um=2.0, ind_h_nm=20, Lg=7e-8,
                 Qc=1e5, P_read=-76, T_noise=4.0, delta_0 = 1e-3, F_TLS=1.0,
                 N0 = 1.72e10, cap=11.7e-12, T_star=0.35, tau_star=40e-6,
                 delta_loss=1e-5):
        self.params = lmfit.Parameters()
        self.params.add('nqp0',value = nqp0, min = 0,max=1e5)
        self.params.add('delta_0',value = delta_0, min = 0,max=1)
        self.params.add('F_TLS',value=F_TLS,min=0, max=1)
        self.params.add('sigman',value=sigman,min=1e4,max=1e6)
        self.params.add('T_star',value=T_star,min=.1,max=1)
        self.params.add('Tc',value=Tc,min=1,max=2)
        self.params.add('delta_loss',value=delta_loss,min=0,max=1e-1)
        self.params.add('Lg',value = Lg,min=0,max=100)
        self.params.add('cap',value = cap,min=0,max=100)
        self.params.add('foffset',value=0,min=-1e-4,max=1e-4)
        self.nqp0 = nqp0
#       self.Tbase = Tbase
        self.f0_nom = f0_nom
        self.ind_l_um = ind_l_um
        self.ind_w_um = ind_w_um
        self.ind_h_nm = ind_h_nm
        self.N0 = N0
        self.Qc = Qc
        self.tau_star = tau_star
        self.P_read = P_read
        self.T_noise = T_noise

        self.fit_for = None
        
    @property
    def foffset(self):
        return self.params['foffset'].value
    @property
    def delta_0(self):
        return self.params['delta_0'].value
    @property
    def delta_loss(self):
        return self.params['delta_loss'].value
    @property
    def Tc(self):
        return self.params['Tc'].value
    @property
    def Lg(self):
        # can upgrade this later to some sort of estimate
        return self.params['Lg'].value*1e-9
    @property
    def cap(self):
        return self.params['cap'].value*1e-12
    
    @property
    def ind_nsq(self):
        return self.ind_l_um/self.ind_w_um
    
    @property
    def Delta(self):
        return 1.74 * kBeV * self.Tc
    
    @property
    def ind_vol(self):
        return self.ind_w_um * self.ind_l_um * self.ind_h_nm * 1e-3
    
    @property
    def v_read(self):
        return np.sqrt(50 * 10**(self.P_read/10) * 1e-3)
    
    def nqp(self,T):
        nqp0 = self.params['nqp0'].value
        return 2 * self.N0 * np.sqrt(2*np.pi*kBeV*T*self.Delta) * np.exp(-self.Delta/(kBeV*T)) + nqp0
    
    def tau_qp(self,T):
        T_star = self.params['T_star']
        return self.tau_star*self.nqp(T_star)/self.nqp(T)
    
    def mu_star(self,T):
        return kBeV * T * np.log(self.nqp(T)/(2*self.N0*np.sqrt(2*np.pi*kBeV*T*self.Delta))) + self.Delta
    
    def xi(self,T):
        return (h*self.f0_nom)/(2*kB*T)
    
    def sigma1(self,T):
        xi = self.xi(T)
        sigman = self.params['sigman'].value
        return ((sigman*(4*self.Delta*qC)/(h*self.f0_nom)) *
                np.exp(-(self.Delta-self.mu_star(T))/(kBeV*T)) *
                np.sinh(xi) * scipy.special.k0(xi))
    
    def sigma2(self,T):
        xi = self.xi(T)
        sigman = self.params['sigman'].value
        return ((sigman*(np.pi*self.Delta*qC)/(h*self.f0_nom)) *
                (1 - (self.nqp(T)/(2*self.N0*self.Delta))*(1+np.sqrt((2*self.Delta)/(np.pi*kBeV*T))*
                                                           np.exp(-xi)*scipy.special.i0(xi))))
        
    def beta(self,T):
        return self.sigma2(T)/self.sigma1(T)
    
    def Lk(self,T):
        return self.ind_nsq/(2*np.pi*self.f0_nom*self.sigma2(T)*self.ind_h_nm*1e-7)
    
    def alpha(self,T):
        lk = self.Lk(T)
        return lk/(lk+self.Lg)
    
    def Qi(self,T):
        react = 2*np.pi*self.f0_nom*(self.Lg+self.Lk(T))
        diss = self.ind_nsq*self.sigma1(T)/(self.sigma2(T)**2*self.ind_h_nm*1e-7)
        return react/diss
    
    def Qr(self,T):
        return 1/(1/self.Qi(T)+1/self.Qc)
        
    def depth_db(self,T):
        return 20*np.log10(1-self.Qr(T)/self.Qc)
    
    def res_width(self,T):
        return self.f0_nom/self.Qr(T)
    
    def total_qp(self,T):
        return self.nqp(T)*self.ind_vol
    
    def f_res(self,T):
        return 1/(2*np.pi*np.sqrt((self.Lk(T)+self.Lg)*self.cap))
    
    def dfdNqp(self,T):
        y0 = self.f_res(T)
        y1 = self.f_res(T+1e-3)
        x0 = self.total_qp(T)
        x1 = self.total_qp(T+1e-3)
        return -(y1-y0)/(x1-x0)
    
    def gr_noise(self,T):
        return 2*np.sqrt(self.total_qp(T)*self.tau_qp(T))*self.dfdNqp(T)
    
    def dVdf(self,T):
        return 4 * self.v_read * self.alpha(T) * self.Qr(T)**2 / (self.Qc*self.f_res(T))
    
    def amp_noise(self,T):
        vn = np.sqrt(4*kB*self.T_noise*50)
        return vn/self.dVdf(T)
    
    def noise_spectrum(self,freq,Tphys):
        fqp = 1/self.tau_qp(Tphys)
        gr = (self.gr_noise(Tphys)*np.abs(1/(1+1j*freq/self.res_width(Tphys))))**2
        return gr + self.amp_noise(Tphys)**2
    
    def tls_shift(self,T):
        F_TLS = self.params['F_TLS'].value
        delta_0 = self.delta_0 #self.params['delta_0'].value
        xi = self.xi(T)
        return ((F_TLS*delta_0/np.pi) *
                (np.real(scipy.special.psi(0.5+(xi/(1j*np.pi))))-
                 np.log(xi*2)))
        
    def delta_tls(self,T):
        delta_0 = self.delta_0 #self.params['delta_0'].value
        xi = self.xi(T)
        return delta_0 * np.tanh(xi) + self.params['delta_loss'].value
    
    def total_Qi(self,T):
        return 1/(1/self.Qi(T) + self.params['F_TLS'].value*self.delta_tls(T))
    
    def total_fres(self,T):
        #return (1+self.f_res(T))*(1+self.tls_shift(T))-1
        return self.f_res(T)+self.tls_shift(T)+self.foffset
    
    def fit_f0_resid(self,params,T,f0,f0_err=None):
        if f0_err is None:
            return (f0 - self.total_fres(T))#/self.f0_nom
        else:
            return (f0 - self.total_fres(T))/f0_err
    def fit_f0(self,T,f0,f0_err=None):
        self.fit_for = 'f0'
        self.T_data = T
        self.f0_data = f0
        self.f0_err = f0_err
        self.params['F_TLS'].value = 1.0
        self.params['F_TLS'].vary = False
        self.result = lmfit.minimize(self.fit_f0_resid,self.params,args=(T,f0,f0_err))
        
    def fit_qi_resid(self,params,T,Qi,Qi_err=None):
        if Qi_err is None:
            return (Qi - self.total_Qi(T))#/1e7
        else:
            return abs(Qi - self.total_Qi(T))/Qi_err
    
    def fit_qi(self,T,Qi,Qi_err=None):
        self.fit_for = 'qi'
        self.T_data = T
        self.Qi_data = Qi
        self.Qi_err = Qi_err
        self.result = lmfit.minimize(self.fit_qi_resid,self.params,args=(T,Qi,Qi_err))
        
    def fit_f0_qi_resid(self,params,T,f0,Qi,f0_err=None,Qi_err=None):
        return np.concatenate((self.fit_f0_resid(params, T, f0,f0_err),self.fit_qi_resid(params, T, Qi,Qi_err)))
    def fit_f0_qi(self,T,f0,Qi,f0_err = None, Qi_err = None,**kwargs):
        self.fit_for = 'f0qi'
        self.T_data = T
        self.f0_data = f0
        self.f0_err = f0_err
        self.Qi_data = Qi
        self.Qi_err = Qi_err
        self.result = lmfit.minimize(self.fit_f0_qi_resid,self.params,args=(T,f0,Qi,f0_err,Qi_err),**kwargs)

    def residual(self):
        if self.fit_for == 'f0':
            return self.fit_f0_resid(None,self.T_data,self.f0_data),self.f0_err
        if self.fit_for == 'qi':
            return self.fit_qi_resid(None,self.T_data,self.Qi_data),self.Qi_err
        if self.fit_for == 'f0qi':
            if self.f0_err is None or self.Qi_err is None:
                errs = None
            else:
                errs = np.concatenate((self.f0_err,self.Qi_err))
            return self.fit_f0_qi_resid(None,self.T_data,self.f0_data,self.Qi_data),errs
        raise Exception("Got unexpected fit for argument: %s" % str(self.fit_for))
        
class DarkKIDModel(KIDModel):
    def sigma1(self,T):
        xi = self.xi(T)
        sigman = self.params['sigman'].value
        result = (sigman*(2*self.Delta*qC/(h*self.f0_nom)) *
                  (self.nqp(T)/(self.N0*np.sqrt(2*np.pi*kBeV*T*self.Delta))) *
                  np.sinh(xi) * scipy.special.k0(xi))
        return result
    
    def sigma2(self,T):
        xi = self.xi(T)
        sigman = self.params['sigman'].value
        return ((sigman*(np.pi*self.Delta*qC)/(h*self.f0_nom)) *
                (1 - (self.nqp(T)/(2*self.N0*self.Delta))*(1+np.sqrt((2*self.Delta)/(np.pi*kBeV*T))*
                                                           np.exp(-xi)*scipy.special.i0(xi))))

class DarkKIDModelFractional(DarkKIDModel):
    def __init__(self,Tc=1.46, nqp0=0, f0_nom=100e6,
                 sigman=5e5, alpha=.66,
                 Qc=1e5, P_read=-106, T_noise=4.0, delta_0 = 0, F_TLS=1.0,
                 N0 = 1.72e10,
                 delta_loss=0.0):
        self.params = lmfit.Parameters()
        self.params.add('nqp0',value = nqp0, min = 0,max=1e5,vary=False)
        self.params.add('delta_0',value = delta_0, min = 0,max=.1)
        self.params.add('F_TLS',value=F_TLS,vary=False)
#        self.params.add('sigman',value=sigman,min=1e4,max=1e6)
        self.params.add('Tc',value=Tc,min=1,max=2,vary=False)
        self.params.add('delta_loss',value=delta_loss,min=0,max=1e-1)
        self.params.add('alpha',value = alpha,min=0.1,max=1,vary=True)
        self.params.add('foffset',value=0,min=-1e-4,max=1e-4)
#       self.Tbase = Tbase
        self.N0 = N0
        self.Qc = Qc
        self.P_read = P_read
        self.T_noise = T_noise
        self.f0_nom = f0_nom
        
    @property
    def alpha(self):
        return self.params['alpha'].value
    
    @property
    def nqp0(self):
        return self.params['nqp0'].value
    
    def f_res(self, T):
        s1,s2 = s1s2(T, self.Tc, self.f0_nom)
        #T0 = T_nqp_equals_nqp0(self.Delta, self.N0, self.nqp0)
        delta_f = (self.alpha * s2 * self.nqp(T)) / (4 * self.N0 * self.Delta)
        delta_f0 = 0#(self.alpha * s2 * (2*self.nqp0)) / (4 * self.N0 * self.Delta)
        return -(delta_f-delta_f0)
    
    def Qi(self,T):
        s1,s2 = s1s2(T, self.Tc, self.f0_nom)
        return (2*self.N0*self.Delta)/(self.alpha*s1*self.nqp(T))
    
    def invQi(self,T):
        return 1/self.Qi(T)
    
    def total_delta_invQi(self,T):
        delta_TLS_loss = self.delta_tls(T)
        invQi = self.invQi(T)
        return invQi + delta_TLS_loss + self.delta_loss
        
    
    

class DarkKIDModel2(KIDModel):
    def sigma1(self,T):
        xi = self.xi(T)
        sigman = self.params['sigman'].value
        result = (sigman*(2*self.Delta*qC/(h*self.f0_nom)) *
                  (self.nqp(T)/(self.N0*np.sqrt(2*np.pi*kBeV*T*self.Delta))) *
                  np.sinh(xi) * scipy.special.k0(xi))
        return result
    
    def sigma2(self,T):
        xi = self.xi(T)
        sigman = self.params['sigman'].value
        return ((sigman*(np.pi*self.Delta*qC)/(h*self.f0_nom)) *
                (1 - (self.nqp(T)/(2*self.N0*self.Delta))*(1+np.sqrt((2*self.Delta)/(np.pi*kBeV*T))*
                                                           np.exp(-xi)*scipy.special.i0(xi))))
        