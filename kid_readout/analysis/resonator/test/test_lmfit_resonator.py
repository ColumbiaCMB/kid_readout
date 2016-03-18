import numpy as np
from kid_readout.analysis.resonator import lmfit_models, lmfit_resonator

def test_resonator():
    lrm = lmfit_models.LinearResonatorModel()
    f = np.linspace(99.95,100.05,100)
    s21_true = lrm.eval(f=f,Q=1e4,f_0=100.,Q_e_real=9e3,Q_e_imag=9e3)
    np.random.seed(123)
    s21_meas = s21_true + 0.02*(np.random.randn(*f.shape) + 1j*np.random.randn(*f.shape))
    lr = lmfit_resonator.Resonator(frequency=f, s21=s21_meas,errors=None,model=lmfit_models.LinearResonatorModel)
    lr.fit()
    print lr.current_result.params
    assert(np.allclose(s21_meas,lr.s21))
    assert(np.abs(lr.Q-1e4) < 3*lr.Q_error)
    assert(np.abs(lr.f_0 - 100) < 3*lr.f_0_error)
    assert(lr.f_0_error < 1e-4)