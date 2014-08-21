import kid_readout.analysis.resonator

import numpy as np
#import nose.tools

def test_dtype_agreement():
    dtypes = [np.complex64, np.complex128, np.float32, np.float64]
    for dtype1 in dtypes:
        for dtype2 in dtypes:
            print dtype1,dtype2
            y_data = np.random.randn(10)+1j*np.random.randn(10)
            errors = np.random.randn(10)+1j*np.random.randn(10)
            y_data = y_data.astype(dtype1)
            errors = errors.astype(dtype2)
            x_data = np.linspace(100,110,10)
            if np.iscomplexobj(y_data) and np.iscomplexobj(errors):
                kid_readout.analysis.resonator.fit_best_resonator(freq=x_data,s21=y_data,errors=errors)
                print "didn't fail"
            elif (not np.iscomplexobj(y_data)) and (not np.iscomplexobj(errors)):
                try:
                    kid_readout.analysis.resonator.fit_best_resonator(freq=x_data,s21=y_data,errors=errors)
                except TypeError:
                    print "failed as expected"
                    pass
            elif np.iscomplexobj(y_data) and not np.iscomplexobj(errors):
                kid_readout.analysis.resonator.fit_best_resonator(freq=x_data,s21=y_data,errors=errors)
                print "didn't fail"
            else:
                try:
                    kid_readout.analysis.resonator.fit_best_resonator(freq=x_data,s21=y_data,errors=errors)
                except TypeError:
                    print "failed as expected"
                    pass

if __name__ == "__main__":
    test_dtype_agreement()

