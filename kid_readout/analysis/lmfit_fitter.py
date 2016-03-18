from lmfit.ui import Fitter

class FitterWithAttributeAccess(Fitter):
    def __getattr__(self, attr):
        """
        This allows instances to have a consistent interface while using different underlying models.

        Return a fit parameter or value derived from the fit parameters.
        """
        if attr.endswith('_error'):
            name = attr[:-len('_error')]
            try:
                return self.current_result.params[name].stderr
            except KeyError:
                print "couldnt find error for ",name,"in self.current_result"
                pass
        try:
            return self.current_params[attr].value
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __dir__(self):
        return (dir(super(Fitter, self)) +
                self.__dict__.keys() +
                self.current_params.keys() +
                [name + '_error' for name in self.current_params.keys()])