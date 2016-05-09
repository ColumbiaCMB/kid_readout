import numpy as np
#import nose.tools
import lmfit
import warnings

import kid_readout.analysis.fitter


def complex_dummy_guess(x, y):
    offset_guess = np.real(y[abs(x).argmin()])
    slope_guess = np.real(y.ptp()/x.ptp())
    params = lmfit.Parameters()
    params.add('offset', value=offset_guess)
    params.add('slope', value=slope_guess)
    return params


def complex_dummy_model(params, x):
    slope = params['slope'].value
    offset = params['offset'].value
    return (slope*x + offset).astype('complex')


def test_dtype_agreement():
    dtypes = [np.complex64, np.complex128, np.float32, np.float64]
    for dtype1 in dtypes:
        for dtype2 in dtypes:
            print dtype1,dtype2
            y_data = np.random.randn(10)+1j*np.random.randn(10)
            errors = np.random.randn(10)+1j*np.random.randn(10)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.ComplexWarning)
                y_data = y_data.astype(dtype1)
                errors = errors.astype(dtype2)
            x_data = np.linspace(100,110,10)
            if np.iscomplexobj(y_data) and np.iscomplexobj(errors):
                kid_readout.analysis.fitter.Fitter(x_data=x_data,y_data=y_data,errors=errors,
                                                   model=complex_dummy_model,
                                                   guess=complex_dummy_guess)
            elif (not np.iscomplexobj(y_data)) and (not np.iscomplexobj(errors)):
                kid_readout.analysis.fitter.Fitter(x_data=x_data,y_data=y_data,errors=errors,
                                                   model=kid_readout.analysis.fitter.line_model,
                                                   guess=kid_readout.analysis.fitter.line_guess)
            else:
                try:
                    kid_readout.analysis.fitter.Fitter(x_data=x_data,y_data=y_data,errors=errors,
                                                   model=complex_dummy_model,
                                                   guess=complex_dummy_guess)
                except TypeError:
                    pass

if __name__ == "__main__":
    test_dtype_agreement()

