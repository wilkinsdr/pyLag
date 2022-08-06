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

    def __call__(self, *args):
        return self.eval(*args)

class AdditiveModel(Model):
    def __init__(self, components):
        self.components = [c(component_name='add%0d'%n) for n, c in enumerate(components)]

    def get_params(self):
        params = lmfit.Parameters()
        for c in self.components:
            params = params + c.get_params()
        return params

    def eval(self, params, x):
        return np.sum(np.vstack([c.eval(params, x) for c in self.components]), axis=0)

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

    def eval_gradient(self, params, x):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value

        return np.stack([x**slope, norm * x**slope * np.log(x)], axis=-1)


class BendingPowerLaw(Model):
    def get_params(self, norm=1., slope=1.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=-50, max=50)
        params.add('%sslope1' % self.prefix, value=slope, min=-10, max=10)
        params.add('%sfbend' % self.prefix, value=slope, min=-6, max=-2)
        params.add('%sslope2' % self.prefix, value=slope, min=-10, max=10)

        return params

    def eval(self, params, x):
        norm = params['%snorm' % self.prefix].value
        slope1 = params['%sslope1' % self.prefix].value
        fbend = 10. ** params['%sfbend' % self.prefix].value
        slope2 = params['%sslope2' % self.prefix].value

        return np.exp(norm) * x ** slope1 / (
                    1. + (x / fbend) ** (slope1 - slope2))


class Lorentzian(Model):
    def get_params(self, norm=1., centre=-4, width=1e-3):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=-50, max=50)
        params.add('%scentre' % self.prefix, value=centre, min=-6, max=-2)
        params.add('%swidth' % self.prefix, value=width, min=1e-6, max=100)

        return params

    def eval(self, params, x):
        norm = params['%snorm' % self.prefix].value
        centre = 10. ** params['%scentre' % self.prefix].value
        width = params['%swidth' % self.prefix].value

        return norm * (1./np.pi) * 0.5 * width / ((x - centre)**2 + 0.25*width**2)

    def eval_gradient(self, params, x):
        norm = params['%snorm' % self.prefix].value
        centre = 10. ** params['%scentre' % self.prefix].value
        width = params['%swidth' % self.prefix].value

        return np.stack([(1./np.pi) * 0.5 * width / ((x - centre)**2 + 0.25*width**2),
                centre * np.log(10) * (width / np.pi) * (x - centre) / ((x - centre)**2 + 0.25*width**2)**2,
                         (1./(2.*np.pi)) * ( ((x - centre)**2 + 0.25*width**2) - 0.5*width**2 ) / ((x - centre)**2 + 0.25*width**2)**2
        ], axis=-1)


class Constant(Model):
    def get_params(self, slope=1., intercept=0.):
        params = lmfit.Parameters()

        params.add('%sconstant' % self.prefix, value=slope, min=-1e10, max=1e10)

        return params

    def eval(self, params, x):
        constant = params['%sconstant' % self.prefix].value

        return constant * np.ones_like(x)