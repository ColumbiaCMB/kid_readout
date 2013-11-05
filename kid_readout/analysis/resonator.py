from __future__ import division

import numpy as np
from kid_readout.analysis.generic_resonator import GenericResonator
# To use different defaults, change these three import statements.
# Note that the current model doesn't include a cable delay, so feel
# free to upgrade it to something better.
from kid_readout.analysis.khalil import s21_generic as default_model
from kid_readout.analysis.khalil import guess_generic as default_guess
from kid_readout.analysis.khalil import generic_functions as default_functions

class Resonator(GenericResonator):
    """
    This class represents a single resonator. The idea is that, given
    sweep data f and s21, r = Resonator(f, s21) should just
    work. Modify the import statements to change the defaults.
    """
   
    def __init__(self, f, data, model=default_model, guess=default_guess, functions=default_functions):
        """
        Instantiate a resonator using our current best model.
        """
        super(Resonator, self).__init__(model, functions)
        self.fit(f, data, guess(f, data))
