from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import cPickle
from lmfit import minimize #,Parameters,report_errors
#.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.size'] = 16.0
    
from kid_readout.analysis.noise_function import pkl_mask as default_mask
from kid_readout.analysis.noise_function import guess as default_guess
from kid_readout.analysis.noise_function import model as default_model

class Noise(object):
    def __init__(self,f,p,A=4709632.149,B=1,alpha=-3,beta=-0.1,fc=2e3,i=3,
                 guess_func=default_guess,model_func=default_model,mask_func=None): 
        if mask_func is None:
            self.f = f
            self.p = p
        else:
            f_masked,p_masked = mask_func(f,p)
            self.f = f_masked
            self.p = p_masked
        self._model = model_func
        self.fit(guess_func(A,B,alpha,beta,fc,i))
    
    def __getattr__(self,attr):
        try:
            return self.result.params[attr].value
        except KeyError:
            pass

    def __dir__(self):
        return (dir(super(Noise, self)) +
                self.__dict__.keys() +
                self.result.params.keys())
        
    def fit(self,initial):
        self.result = minimize(self.residual,initial) #ftol=1e-6
        
    def residual(self,params):
        return (self.p - self.model(params))
        
    def model(self,params):
        return self._model(self.f,params)
        
def plot_pkl_noise(fns,A=4709632.149,B=1,alpha=-3,beta=-0.1,fc=2e3,i=3,f_crop=2e4,
                   guess_func=default_guess,model_func=default_model,mask_func=default_mask): 
    fh = open(fns)
    try:
        pkls = cPickle.load(fh)
    except EOFError:
        print "End of File Error"
    else:
        fh.close() 
        for el in range(np.size(pkls)):
            nm=pkls[el]
            f = nm.fr_coarse
            data = nm.prr_coarse
            f = f[1:]
            data = data[1:]
            f =np.array([j for j in f if j<=f_crop])
            p = data[0:len(f)] 
            n = Noise(f,p,A,B,alpha,beta,fc,i,guess_func,model_func,mask_func)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.loglog(n.f,n.p,'g',lw=2)
            ax.loglog(n.f,n.model(n.result.params),'r',lw=2)
            ax.set_xlim([n.f[0],n.f[-1]])
            ax.set_xlabel('Hz')
            ax.set_ylabel('Hz$^2$/Hz')
            ax.grid()
            #text = (('((%.3ff$^%.3f$ + %.3ff$^%.3f$) \frac{1}{\vert (1 + \frac{f}(%.3f))$^%.3f$} + %.3f)' % (1,2,3,4,5,6,7)))  # fix !!!
            #ax.text(0.5,0.5,text,ha='center',va='center',bbox=dict(fc='white',alpha=0.6),
            #        transform = ax.transAxes,fontdict=dict(size='x-small'))
                

    

    
            