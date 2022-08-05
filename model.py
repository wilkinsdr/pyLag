"""
pylag.model

Provides Model classes for fitting functions to data

v1.0 05/07/2019 - D.R. Wilkins
"""

import numpy as np
import lmfit


class Model(object):
    def __init__(self, component_name=None, **kwargs):
        self.prefix = component_name + "_" if component_name is not None else ''
        self.params = self.get_params(**kwargs)

    def get_params(self):
        raise AssertionError("I should be overridden!")

    def eval(self, params, x):
        raise AssertionError("I should be overriden")

    def eval_gradient(self, x):
        raise AssertionError("I should be overriden")


class Linear(Model):
    def get_params(self, slope=1., intercept=0.):
        params = lmfit.Parameters()

        params.add('%sslope' % self.prefix, value=slope, min=-1e10, max=1e10)
        params.add('%sintercept' % self.prefix, value=intercept, min=-1e10, max=1e10)

        return params

    def eval(self, params, x):
        slope = params['%sslope' % self.prefix].value
        intercept = params['%sintercept' % self.prefix].value

        return slope*x + intercept


class PowerLaw(Model):
    def get_params(self, slope=1., intercept=0.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=slope, min=1e-10, max=1e10)
        params.add('%sslope' % self.prefix, value=slope, min=-10, max=10)

        return params

    def eval(self, params, x):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value

        return norm * x**slope


class Constant(Model):
    def get_params(self, slope=1., intercept=0.):
        params = lmfit.Parameters()

        params.add('%sconstant' % self.prefix, value=slope, min=-1e10, max=1e10)

        return params

    def eval(self, params, x):
        constant = params['%sconstant' % self.prefix].value

        return constant * np.ones_like(x)