"""
pylag.model

Provides Model classes for fitting functions to data

v1.0 05/07/2019 - D.R. Wilkins
"""

import numpy as np
import lmfit
import copy

def param2array(params, variable_only=False):
    """
    Convert a Parameters object into an array of the parameter values
    """
    if variable_only:
        return np.array([params[p].value for p in params if params[p].vary])
    else:
        return np.array([params[p].value for p in params])

def array2param(values, params, variable_only=False):
    """
    Create a new Parameters object from a prototype object params, with values from an array
    """
    out_params = copy.copy(params)
    if variable_only:
        for p, v in zip([p for p in out_params if out_params[p].vary], values):
            out_params[p].value = v
    else:
        for p, v in zip(params, values):
            out_params[p].value = v
    return out_params


class Model(object):
    """
    pylag.model.Model

    Base class for deriving models that can be fit to data. Models are defined as a class that inhertis from this class.
    The model class contains definitions of all the model parameters, and the means to evaluate the model for some set
    of parameter values. Parameters are handled via lmfit Parameters objects to store parameter values and limits, and
    to enable parameters to be free or frozen during the fit.

    Each model class should define the following functions that override the functions in this base class:

    get_params(): returns a new Parameters() objects containing all of the aprameters required for this model.

    eval(params, x): evaluates the model for the parameter values stored in params at points x along the x-axis. The
    eval function should return an array containing the model value at each point x.

    Optionally, the class can also provide the method eval_gradient(params, x) to evalate the derivative of the model
    with respect to each parameter. This function should return a 2-dimensional array of dimension (Nx, Npar), containing
    the derivative at each point x with respect to each parameter. Providing an eval_gradient method enables more
    precise analytic derivatives to be used during fitting, rather than having to evauate numberical derivatives.
    """
    def __init__(self, component_name=None, **kwargs):
        self.prefix = component_name + "_" if component_name is not None else ''
        self.params = self.get_params(**kwargs)

    def get_params(self):
        raise AssertionError("I should be overridden!")

    def eval(self, params, x):
        raise AssertionError("I should be overriden")

    def eval_gradient(self, params, x):
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

    def eval_gradient(self, params, x):
        return np.hstack([c.eval_gradient(params, x) for c in self.components])

    def __getitem__(self, item):
        """
        Overload the [] operator to return the specified component number
        """
        return self.components[item]


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

    def eval_gradient(self, params, x):
        slope = params['%sslope' % self.prefix].value
        intercept = params['%sintercept' % self.prefix].value

        return np.stack([x, np.ones_like(x)], axis=-1)


class PowerLaw(Model):
    def get_params(self, slope=1., intercept=0.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=slope, min=-50, max=50)
        params.add('%sslope' % self.prefix, value=slope, min=-10, max=10)

        return params

    def eval(self, params, x):
        norm = np.exp(params['%snorm' % self.prefix].value)
        slope = params['%sslope' % self.prefix].value

        return norm * x**slope

    def eval_gradient(self, params, x):
        norm = np.exp(params['%snorm' % self.prefix].value)
        slope = params['%sslope' % self.prefix].value

        return np.stack([norm * x**slope, norm * x**slope * np.log(x)], axis=-1)


class BendingPowerLaw(Model):
    def get_params(self, norm=1., slope=1.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=-50, max=50)
        params.add('%sslope1' % self.prefix, value=slope, min=-10, max=10)
        params.add('%sfbend' % self.prefix, value=slope, min=-6, max=-2)
        params.add('%sslope2' % self.prefix, value=slope, min=-10, max=10)

        return params

    def eval(self, params, x):
        norm = np.exp(params['%snorm' % self.prefix].value)
        slope1 = params['%sslope1' % self.prefix].value
        fbend = 10. ** params['%sfbend' % self.prefix].value
        slope2 = params['%sslope2' % self.prefix].value

        return norm * x ** slope1 / (1. + (x / fbend) ** (slope1 - slope2))

    def eval_gradient(self, params, x):
        norm = np.exp(params['%snorm' % self.prefix].value)
        slope1 = params['%sslope1' % self.prefix].value
        fbend = 10. ** params['%sfbend' % self.prefix].value
        slope2 = params['%sslope2' % self.prefix].value

        return np.stack([norm * x ** slope1 / (1. + (x / fbend) ** (slope1 - slope2)),
                         norm * x ** slope1 * (np.log(x) + (x / fbend) ** (slope1 - slope2) + np.log(fbend)) / (
                             (1 + (x / fbend) ** (slope1 - slope2))**2),
                         norm * x ** (2*slope1 - slope2) * fbend ** (slope2 - slope1 - 1) * (slope2 - slope1) / (
                             (1 + (x / fbend) ** (slope1 - slope2))**2),
                         norm * x ** slope1 * (x / fbend) ** (slope1 - slope2) * (np.log(x) - np.log(fbend)) / (
                                 (1 + (x / fbend) ** (slope1 - slope2)) ** 2)
                         ], axis=-1)


class Lorentzian(Model):
    def get_params(self, norm=1., centre=-4, width=1e-3):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=-50, max=50)
        params.add('%scentre' % self.prefix, value=centre, min=-6, max=-2)
        params.add('%swidth' % self.prefix, value=width, min=1e-6, max=100)

        return params

    def eval(self, params, x):
        norm = np.exp(params['%snorm' % self.prefix].value)
        centre = 10. ** params['%scentre' % self.prefix].value
        width = params['%swidth' % self.prefix].value

        return norm * (1./np.pi) * 0.5 * width / ((x - centre)**2 + 0.25*width**2)

    def eval_gradient(self, params, x):
        norm = np.exp(params['%snorm' % self.prefix].value)
        centre = 10. ** params['%scentre' % self.prefix].value
        width = params['%swidth' % self.prefix].value

        return np.stack([norm * (1. / np.pi) * 0.5 * width / ((x - centre) ** 2 + 0.25 * width ** 2),
                         centre * np.log(10) * (norm * width / np.pi) * (x - centre) / (
                                     (x - centre) ** 2 + 0.25 * width ** 2) ** 2,
                         norm * (1. / (2. * np.pi)) * (((x - centre) ** 2 + 0.25 * width ** 2) - width ** 2) / (
                                     (x - centre) ** 2 + 0.25 * width ** 2) ** 2
                         ], axis=-1)


class Constant(Model):
    def get_params(self, slope=1., intercept=0.):
        params = lmfit.Parameters()

        params.add('%sconstant' % self.prefix, value=slope, min=-1e10, max=1e10)

        return params

    def eval(self, params, x):
        constant = params['%sconstant' % self.prefix].value

        return constant * np.ones_like(x)

    def eval_gradient(self, params, x):
        return np.ones_like(x)[:, np.newaxis]   # add a dimension to match the shape of gradients from other models
