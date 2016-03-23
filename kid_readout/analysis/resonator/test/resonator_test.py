from __future__ import division
import unittest 
import kid_readout.analysis.resonator.legacy_resonator as resonator
import kid_readout.analysis.resonator.khalil as khalil
import numpy as np
from scipy.special import cbrt

class bifurcation_test(unittest.TestCase):
    
    def setUp(self):
        '''Verify environment is setup properly
        data taken from the fitted params of the sweep 2013-12-03_174822.nc
        swps[0], swp.atten = 21.0, swp.power_dbm=-51 '''
        
        f = np.linspace(82.85,82.90,10000)
        f_0 = 82.881569344696032
        A_mag = 0.045359757228080611
        A_phase = 0
        A = A_mag*np.exp(1j * A_phase)
        Q = 32253.035141948105
        Q_e_real = 120982.33962292993
        Q_e_imag = 51324.601136160083
        Q_e = (Q_e_real +1j*Q_e_imag)
        delay = 0.05255962136531532
        phi = -1.3313606721103854 
        f_min = f[0]
        a = 0.56495480690974931
        self.Q_i = (Q**-1 - np.real(Q_e**-1))**-1
        
        y_0 = ((f - f_0)/f_0)*Q
        y = (y_0/3. +(y_0**2/9 - 1/12)/cbrt(a/8 + y_0/12 + np.sqrt((y_0**3/27 + y_0/12 + a/8)**2 - (y_0**2/9 - 1/12)**3) + y_0**3/27) + cbrt(a/8 + y_0/12 + np.sqrt((y_0**3/27 + y_0/12 + a/8)**2 - (y_0**2/9 - 1/12)**3) + y_0**3/27))
        x = y/Q
        s21 = A*(1 - (Q/Q_e)/(1+2j*Q*x))
        msk = np.isfinite(s21)
        if not np.all(msk):
            s21_interp_real = np.interp(f,f[msk],s21[msk].real)
            s21_interp_imag = np.interp(f,f[msk],s21[msk].imag)
            s21new = s21_interp_real+1j*s21_interp_imag
        else:
            s21new = s21
            
        cable_delay = np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))   
        bifurcation = s21new*cable_delay
        
        self.f_0 = f_0
        self.f = f
        self.Q = Q
        self.bifurcation = bifurcation
        self.res = resonator.Resonator(self.f,self.bifurcation,model=khalil.bifurcation_s21,guess=khalil.bifurcation_guess)
        
                                  
    def test_model(self):
        """
        Require the fitted params from Resonator are less than 1% derivation
        from the values provided in setUp function. 
        """
        mean_theory = abs(np.mean(self.bifurcation))
        mean_model = abs(np.mean(self.res.model(self.res.result.params)))
        self.assertAlmostEqual(mean_theory,mean_model,4)
    
    
    def test_fitted_f0(self):
        """
        f_0 is in range (1,100), thus the decimal precision (places) is 1 to
        get 1% derivation
        """
        self.assertAlmostEqual(self.res.f_0,self.f_0,places =1)
    
    def test_fitted_Q(self):
        """
        Q ~ 1e5, therefore, they are divided by 1e3 to get the form XX.XXX
        places =1 to obtain 1% deviation
        """
        self.assertAlmostEqual(self.res.Q/1e3,self.Q/1e3,places = 1)
   
    def test_fitted_Qi(self):
        """
        """
        self.assertAlmostEqual(self.res.Q_i/1e3,self.Q_i/1e3,places = 1)
        
    def tearDown(self):
        pass
        
class low_power_test(unittest.TestCase):
    
    def setUp(self):
        """2013-12-03_174822.nc, swp.power_dbm = -63"""
        f = np.linspace(82.85,82.90,10000)
        f_0 = 82.881025609577861
        A_mag = 0.011317218107098626
        A_phase = 0
        A = A_mag*np.exp(1j * A_phase)
        Q = 30798.516362470775
        Q_e_real = 124006.88516111636
        Q_e_imag = 55797.084359985311
        Q_e = (Q_e_real +1j*Q_e_imag)
        delay = 0.037137822000660708
        phi = -1.3310218395107833
        f_min = f[0]
        self.Q_i = (Q**-1 - np.real(Q_e**-1))**-1
        
        cable_delay = np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))
        generic_s21 = A * (1 - (Q * Q_e**-1 /(1 + 2j * Q * (f - f_0) / f_0)))
        data = cable_delay* generic_s21
        
        self.f_0 = f_0
        self.f = f
        self.Q = Q
        self.data = data
        self.res = resonator.Resonator(self.f,self.data)
        
    def test_model(self):
        mean_theory = abs(np.mean(self.data))
        mean_model = abs(np.mean(self.res.model(self.res.result.params)))
        self.assertAlmostEqual(mean_theory,mean_model,4)
        
    def test_fitted_f0(self):
        self.assertAlmostEqual(self.res.f_0,self.f_0,places =1) # f_0 ~ 89
        
    def test_fitted_Q(self):
        self.assertAlmostEqual(self.res.Q/1e3,self.Q/1e3,places = 1)
    
    def test_fitted_Qi(self):
       self.assertAlmostEqual(self.res.Q_i/1e3,self.Q_i/1e3,places =1)  # Qi ~ 41e4
    
    def tearDown(self):
        pass
        
if __name__=='__main__':
   unittest.main(verbosity=2)