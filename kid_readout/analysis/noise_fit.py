import lmfit
from kid_readout.analysis import fitter


def lorenz(f, fc, a):
    return a / (1 + (f / fc) ** 2)


def simple_noise_model(params, f):
    A = params['A'].value
    fc = params['fc'].value
    nw = params['nw'].value
    return lorenz(f, fc, A) + nw


def simple_noise_guess(f, S):
    params = lmfit.Parameters()
    params.add('A', (S.max() - S.min()), min=0, max=S.max())
    params.add('nw', (S.max() + S.min()) / 2.0, min=S.min() / 2, max=S.max())
    params.add('fc', 500.0, min=10, max=1e4)
    return params


class SingleLorenzModel(fitter.Fitter):
    def __init__(self, f, data, model=simple_noise_model, guess=simple_noise_guess, functions={},
                 mask=None, errors=None):
        super(SingleLorenzModel, self).__init__(f, data, model=model, guess=guess, functions=functions, mask=mask,
                                                errors=errors)
