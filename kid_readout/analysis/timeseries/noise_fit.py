import lmfit
from kid_readout.analysis import fitter
import numpy as np
from matplotlib import pyplot as plt

def single_pole(f, fc):
    return 1 / (1 + 1j * (f / fc))


def single_pole_noise_model(params, f):
    A = params['A'].value
    fc = params['fc'].value
    nw = params['nw'].value
    return A * np.abs(single_pole(f, fc)) ** 2 + nw

def single_pole_log_parameters_noise_model(params, f):
    A = np.exp(params['log_A'].value)
    fc = np.exp(params['log_fc'].value)
    nw = np.exp(params['log_nw'].value)
    return A * np.abs(single_pole(f, fc)) ** 2 + nw


def single_pole_noise_guess(f, S):
    params = lmfit.Parameters()
    params.add('A', (S.max() - S.min()), min=0, max=S.max())
    params.add('nw', (S.max() + S.min()) / 2.0, min=S.min() / 2, max=S.max())
    params.add('fc', 500.0, min=10, max=1e4)
    return params

def single_pole_log_parameters_noise_guess(f, S):
    params = lmfit.Parameters()
    params.add('log_A', np.log((S.max() - S.min())), max=np.log(S.max()))
    params.add('log_nw', np.log((S.max() + S.min()) / 2.0), min=np.log(S.min() / 2), max=np.log(S.max()))
    params.add('log_fc', np.log(500.0), min=np.log(10), max=np.log(1e4))
    return params

class SinglePoleNoiseModel(fitter.Fitter):
    def __init__(self, f, data, functions={},
                 mask=None, errors=None):
        model=single_pole_noise_model
        guess=single_pole_noise_guess
        super(SinglePoleNoiseModel, self).__init__(f, data, model=model, guess=guess, functions=functions, mask=mask,
                                                   errors=errors)

class SinglePoleLogNoiseModel(fitter.Fitter):
    def __init__(self, f, data, functions={},
                 mask=None, errors=None):
        model=single_pole_log_parameters_noise_model
        guess=single_pole_log_parameters_noise_guess
        super(SinglePoleLogNoiseModel, self).__init__(f, data, model=model, guess=guess, functions=functions, mask=mask,
                                                   errors=errors)
    def __getattr__(self, attr):
        """
        This allows instances to have a consistent interface while using different underlying models.

        Return a fit parameter or value derived from the fit parameters.
        """
        if attr in self.result.params:
            return self.result.params[attr].value
        if ('log_' + attr) in self.result.params:
            return np.exp(self.result.params['log_'+attr].value)
        try:
            return self._functions[attr](self.result.params)
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __dir__(self):
        return (dir(super(fitter.Fitter, self)) +
                self.__dict__.keys() +
                self.result.params.keys() +
                [x[len('log_'):] for x in self.result.params.keys() if x.startswith('log_')] +
                self._functions.keys())

    def _residual_without_errors(self, params=None):
        """
        This is the residual function used by lmfit. The errors are not used in calculating the residual. Only data
        where mask is True is used for the fit.

        Note that the residual needs to be purely real, and should *not* include abs. The minimizer needs the signs
        of the residuals to properly evaluate the gradients.
        """
        # in the following, .view('float') will take a length N complex array
        # and turn it into a length 2*N float array.
        if params is None:
            params = self.result.params
        return (np.log(self.y_data[self.mask]) - np.log(self.model(params)[self.mask])).view('float')

    def _residual_with_errors(self, params=None):
        """
        This is the residual function used by lmfit. The residual at each point is divided by the corresponding error
        for that point. Only data where mask is True is used for the fit.

        Note that the residual needs to be purely real, and should *not* include abs. The minimizer needs the signs
        of the residuals to properly evaluate the gradients.
        """
        # in the following, .view('float') will take a length N complex array
        # and turn it into a length 2*N float array.
        if params is None:
            params = self.result.params
        errors = self.errors[self.mask]
        return ((self.y_data[self.mask].view('float') - self.model(params)[self.mask].view('float')) +
                np.log(errors.view('float')))

def fit_single_pole_noise(fr,sxx,errors=None,max_num_masked=None,debug=False):
    if errors is None:
        errors = sxx/fr #ad hoc, but seems to work
    reduced_chi2 = []
    if max_num_masked is None:
        max_num_masked = len(fr)/2
    for nmask in range(max_num_masked):
        mask = np.ones(fr.shape,dtype=np.bool)
        if nmask > 0:
            mask[np.abs(sxx).argsort()[-nmask:]] = 0 # mask the nmask highest points
        nf = SinglePoleNoiseModel(fr,sxx,mask=mask,errors=errors)
        reduced_chi2.append(nf.result.redchi)
    reduced_chi2 = np.array(reduced_chi2)
    best_num_to_mask = reduced_chi2.argmin() # best number to mask
    mask = np.ones(fr.shape,dtype=np.bool)
    if best_num_to_mask > 0:
        mask[np.abs(sxx).argsort()[-best_num_to_mask:]] = 0 # mask the 'best_num_to_mask' highest points
    nf = SinglePoleNoiseModel(fr,sxx,mask=mask,errors=errors)
    if debug:
        plt.semilogy(range(max_num_masked),reduced_chi2)
    return nf
