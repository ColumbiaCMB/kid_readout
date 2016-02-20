import kid_readout.analysis.resonator.resonator

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
            if np.iscomplexobj(y_data):
                # if the data is complex, the errors will automatically be coerced to match, so this shouldn't fail
                kid_readout.analysis.resonator.resonator.fit_best_resonator(freq=x_data,s21=y_data,errors=errors)
                print "didn't fail"
            else:
                # data is real, so it should fail in any case
                try:
                    kid_readout.analysis.resonator.resonator.fit_best_resonator(freq=x_data,s21=y_data,errors=errors)
                except TypeError:
                    print "failed as expected"
                    pass

if __name__ == "__main__":
    test_dtype_agreement()

