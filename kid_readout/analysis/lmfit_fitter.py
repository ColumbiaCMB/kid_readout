from lmfit.ui import Fitter

class FitterWithAttributeAccess(Fitter):
    def __getattr__(self, attr):
        if attr.endswith('_error'):
            name = attr[:-len('_error')]
            try:
                return self.current_result.params[name].stderr
            except KeyError:
                raise AttributeError("couldn't find error for %s in self.current_result" % name)
        try:
            return self.current_params[attr].value
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __dir__(self):
        return sorted(set(dir(Fitter) +
                self.__dict__.keys() +
                self.current_params.keys() +
                [name + '_error' for name in self.current_params.keys()]))
