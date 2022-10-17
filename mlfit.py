"""
pylag.mlfit

Implements the method of Zoghbi et al. 2013, ApJ 777, 24 (https://ui.adsabs.harvard.edu/abs/2013ApJ...777...24Z/abstract)
to fit correlation functions/covariance matrices to light curves in the time domain to estimate power and lag spectra.

These classes employ the method set out by Zoghbi et al. 2013, and make use the algorithms of Rasmussen & Williams
"Gaussian Processes for Machine Learning", the MIT Press, 2006 and implemented in the scikit-learn
GaussainProcessRegressor (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)

If you make use of this code in any publications, please be sure to cite Zoghbi et al. 2013 and Rasmussen & Williams 2006!

v2.0 05/07/2019 - D.R. Wilkins
"""
import numpy as np

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve, cho_factor, blas
from scipy.optimize import minimize

import lmfit
import copy

from .lightcurve import *
from .binning import *
from .model import *
from .plotter import *

class MLFit(object):
    """
    pylag.mlfit.MLFit

    Base class for fitting timing products to light curves in the time domain.

    This class is not typically used on its own, rather the MLPSD and MLCovariance classes inherit their common
    functionality from here (setting up integrals, evaluating likelihood function, fitting, etc.)

    Constructor arguments:
    t: ndarray: array of time points at which light curves are evaluated (for a cross spectrum, this should
    just be the time points from one light curve, not a stacked vector).
    y: ndarray: data vector to which the covariance matrix is fit (either the count rate in each time bin for a
    power spectrum, or a stacked data vector of the two light curves for a cross spectrum)
    noise: ndarray: array of noise terms to be added to the covariance matrix along the leading diagonal. Each
    element should correspond to each bin in the data vector (and shoud be stacked for a cross spectrum).
    psdnorm: float, optional (default=1.): normalisation factor for the power or cross spectrum
    Nf: int, optional (default=10): number of frequency bins to use in the fit between the minimum and Nyquist
    frequencies. If a model fucntion is specified, the model will be evaluated at these frequencies when calculating
    the covariance integrals.
    fbins: Binning, optional (default=None). A Binning object to use defining the frequency bins, instead of
    constructing bins automatically (this will override Nf)
    model: Model, optional (default=None). If set, a Model object to use to model the power spectrum as a
    function of frequency, using the model's parameters instead of treating each frequency bin as a parameter
    component_name: str, optional (default=None): If set, the name of this model component to prefix onto all
    of the parameter names (when multiple components are combined into a single model).
    model_args: dict, optional (default={}): Arguments to pass to constructor of Model object
    eval_sin: bool, optional (default=False): If True, evaluate the integrals of the sin terms (in addition to
    the cos terms). Required for a cross spectrum with lags.
    extend_freq: float, optional (default=None). If True, create an additional low frequecy bin extending to
    this factor times the minimum frequency. Adding a low frequency bin reduces bias in the lowest frequency bin due
    to red noise leak (see Zoghbi et al. 2013).
    """
    def __init__(self, t, y, noise=None, psdnorm=1., Nf=10, fbins=None, model=None, component_name=None, model_args={}, eval_sin=False, extend_freq=None):
        self.time = np.array(t)
        self.dt = np.min(np.diff(self.time))

        # the linear algebra we'll be doing requires the data points as a column vector (in matrix form)
        self.data = y[:, np.newaxis]

        self.noise = noise

        self.psdnorm = psdnorm

        # matrix of pairwise separations of time bins
        self.tau = self.time - np.expand_dims(self.time, 1)

        if fbins is None:
            # set up frequency bins to span
            min_freq = (1. / (np.max(self.time) - np.min(self.time))) / extend_freq
            max_freq = 0.5 / self.dt
            self.fbins = LogBinning(min_freq, max_freq, Nf)
        else:
            self.fbins = fbins

        if extend_freq is not None:
            # create a new set of bins with an extra one at the start, going down to extend_freq * the previous minimum
            self.fbins = Binning(bin_edges=np.insert(self.fbins.bin_edges, 0, extend_freq*self.fbins.bin_edges.min()))

        self.freq = self.fbins.bin_cent
        self.freq_error = self.fbins.x_error()

        # pre-compute the integral of the cosine and sine term in the correlation for each frequency bin
        # so we don't have to do this for every change in parameter values
        # note the factor of 2 to integrate over the negative frequencies too!
        self.cos_integral = np.zeros((len(self.fbins), self.tau.shape[0], self.tau.shape[1]))
        self.sin_integral = np.zeros((len(self.fbins), self.tau.shape[0], self.tau.shape[1])) if eval_sin else None
        diag = np.eye(self.tau.shape[0], dtype=bool)
        for i, (fmin, fmax) in enumerate(zip(self.fbins.bin_start, self.fbins.bin_end)):
            self.cos_integral[i, ~diag] = (1. / (2. * np.pi * self.tau[~diag])) * (
                        np.sin(2. * np.pi * fmax * self.tau[~diag]) - np.sin(2. * np.pi * fmin * self.tau[~diag]))
            self.cos_integral[i, diag] = fmax - fmin
            if eval_sin:
                self.sin_integral[i, ~diag] = (1. / (2. * np.pi * self.tau[~diag])) * (
                            np.cos(2. * np.pi * fmax * self.tau[~diag]) - np.cos(2. * np.pi * fmin * self.tau[~diag]))

        if model is None:
            self.model = None
        elif isinstance(model, Model):
            self.model = model
        else:
            self.model = model(component_name=component_name, **model_args)

        self.comp_name = component_name
        self.prefix = component_name + "_" if component_name is not None else ''
        self.params = self.get_params()

        self.fit_result = None

        self.mcmc_minimizer = None
        self.mcmc_result = None

    def log_likelihood(self, params, eval_gradient=True):
        """
        loglike, grad = pylag.mlfit.MLFit.log_likelihood(params, eval_gradient=True)

        Evaluate log(marginal likelihood), as well as its gradient, for the covariance matrix defined by some set of
        input parameters, applied to the data points we have.

        Based on the Algorithm 2.1 of Rasmussen & Williams "Gaussian Processes for Machine Learning", the MIT Press,
        2006 and implemented in the scikit-learn GaussainProcessRegressor
        (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)

        :param params: Parameters() object containing the set of parameter values the likelihood function is to be
        evaluated for.
        :param eval_gradient: bool, optional (default=True): whether to return the gradient (derviative/Jacobian) of
        the likelihood, or just the likelihood
        
        :return: loglike: float: log(likelihood) value, grad: ndarray: derivative of -log(likelihood)
        """
        c = self.cov_matrix(params)

        # add white noise along the leading diagonal
        # this should be the Poisson noise term when calculating a PSD
        if self.noise is not None:
            c += np.diag(self.noise)

        try:
            L = cho_factor(c, lower=True, check_finite=False)[0]
        except np.linalg.LinAlgError:
            try:
                # try doubling the noise first
                L = cho_factor(c + np.diag(self.noise), lower=True, check_finite=False)[0]
            except np.linalg.LinAlgError:
                #printmsg(2, "WARNING: Couldn't invert covariance matrix with parameters " + param2array(params))
                return (-1e6, np.zeros(len(params)) + 1e6) if eval_gradient else -1e6
        except ValueError:
            return (np.inf, np.zeros(len(params))) if eval_gradient else -np.inf

        alpha = cho_solve((L, True), self.data, check_finite=False)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.data, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= c.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)

        if eval_gradient:
            c_gradient = self.cov_matrix_deriv(params)
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)
            tmp -= cho_solve((L, True), np.eye(c.shape[0]))[:, :, np.newaxis]
            gradient_dims = 0.5 * np.einsum("ijl,ijk->kl", tmp, c_gradient)
            gradient = gradient_dims.sum(-1)

        # note we return -log_likelihood, so we can minimize it!
        return (log_likelihood, gradient) if eval_gradient else log_likelihood

    def get_params(self):
        """
        param = pylag.mlfit.MLFit.get_params()

        Create a new set of parameters for the model used to construct the covariance matrix
        
        :return: param: Parameters() object containing the parameters
        """
        if self.model is None:
            params = lmfit.Parameters()
            for i in range(len(self.fbins)):
                params.add('%sln_psd%01d' % (self.comp_name, i), 1., vary=True, min=-30., max=30.)
            return params
        else:
            return self.model.get_params()

    def set_param(self, param, value=None, min=None, max=None, vary=None):
        """
        pylag.mlfit.MLFit.set_param(param, value, min, max)

        Set the value, lower and upper bounds for a parameter. If any of value, min or max are None, these values will
        not be altered.

        :param param: str: name of the parameter to set
        :param value: float, optional (default=None): new value of parameter
        :param min: float, optional (default=None): lower bound for parameter
        :param max: float, optional (default=None): upper bound for parameter
        """
        if value is not None:
            self.params[param].value = value
        if min is not None:
            self.params[param].min = min
        if max is not None:
            self.params[param].max = max
        if vary is not None:
            self.params[param].vary = vary

    def save_params(self, filename):
        self.params.dump(open(filename, 'w'))

    def load_params(self, filename):
        self.params.load(open(filename, 'r'))

    def _dofit(self, init_params, method='L-BFGS-B', **kwargs):
        """
        result = pylag.mlfit.MLFit._dofit(init_params, method='L-BFGS-B', **kwargs)

        Function to actually perform the minimisation of -log(likelihood).

        This method is not normally called on its own, rather it is called by the fit() or steppar() method.

        :param init_params: Parameters() object containing the parameter values to use as the starting point
        :param method: str, optional (default='L-BFGS-B'): scipy.optimize.minimize minimisation method to use for the fit
        :param kwargs: additional arguments to pass to scipy.optimize.minimize
        
        :return: result: scipy.optimise fit_result object containing the results of the fit
        """
        initial_par_arr = np.array([init_params[p].value for p in init_params if init_params[p].vary])
        bounds = [(init_params[p].min, init_params[p].max) for p in init_params if init_params[p].vary] if method == 'L-BFGS-B' else None

        def objective(par_arr):
            """
            wrapper around log_likelihood method to evaluate for an array of just the variable parameters,
            which can be used directly with scipy.optimise methods.
            """
            fit_params = copy.copy(init_params)
            for par, value in zip([p for p in init_params if init_params[p].vary], par_arr):
                fit_params[par].value = value
            l, g = self.log_likelihood(fit_params, eval_gradient=True)
            print("\r-log(L) = %6.3g" % l + " for parameters: " + ' '.join(['%6.3g' % p for p in param2array(fit_params)]), end="")
            return -l, -g

        result = minimize(objective, initial_par_arr, method=method, jac=True, bounds=bounds, **kwargs)
        print("\r-log(L) = %6.3g" % result.fun + " for parameters: " + " ".join(['%6.3g' % p for p in param2array(array2param(result.x, init_params))]))
        return result

    def fit(self, init_params=None, update_params=True, **kwargs):
        """
        pylag.mlfit.MLFit._dofit(init_params, method='L-BFGS-B', **kwargs)

        Fit the model covariance matrix to the data by minimising -log(likelihood). Once the fit is complete, the
        parameters stored in the member variable params will be updated, the uncertainties will be estimated from
        the Hessian matrix, and the process_fit_results() method will be called from the derived class to update
        calculate power and lag spectra from the best-fitting parameter values.

        The actual minimisation is done by the _dofit() method.

        :param init_params: Parameters() object containing the parameter values to use as the starting point
        :param update_params: bool, optional (default=True): whether to update the stored parameter values after the fit,
        and to calculate the resulting power and/or lag spectra
        :param kwargs: additional arguments to pass to _dofit() method.
        """
        if init_params is None:
            init_params = self.params

        self.fit_result = self._dofit(init_params, **kwargs)
        print(self.fit_result)

        if True or self.fit_result.success and update_params:
            for par, value in zip([p for p in init_params if init_params[p].vary], self.fit_result.x):
                self.params[par].value = value

            hess = self.fit_result.hess_inv(self.fit_result.x) if callable(self.fit_result.hess_inv) else np.diag(self.fit_result.hess_inv)

            # make sure we only get the finite parameter errors
            self.param_error = np.zeros(len(self.params))
            self.param_error[hess>0] =  hess[hess>0] ** 0.5

            self.process_fit_results(self.fit_result, self.params)

    def run_mcmc(self, init_params=None, burn=300, steps=1000, thin=1, walkers=50, **kwargs):
        """
        pylag.mlfit.MLFit.run_mcmc(init_params=None, burn=300, steps=1000, thin=1, walkers=50, **kwargs)

        Run MCMC to obtain the posterior distributions of the model parameters. MCMC calculation is run using the
        Goodman-Weare algorithm, via emcee, called via lmfit's Minimzer class.

        :param init_params: Parameters, optional (default=None): location in parameter space about which to start the
        chains. If none, will use the params member variable, which will either contain the initial values or the results
        of the most recent fit.
        :param burn: int, optional (default=300): number of chain steps to discard (burn) at the start
        :param steps: int, optional (default=300): number of chain steps to run
        :param thin:  int, optional (default=1): keep every nth step in the chain to reduce correlation between adjacent
        chain points
        :param walkers:  int, optional (default=50): number of walkers to use. Should be much larger than the number of
        free parameters
        :param kwargs: passed to lmfit.Minimizer.emcee()
        """
        if init_params is None:
            init_params = self.params

        # we initialise a Minimizer object, but only if there isn't one already, so we can
        if self.mcmc_minimizer is None:
            self.mcmc_minimizer = lmfit.Minimizer(lambda p: self.log_likelihood(p, eval_gradient=False), params=init_params,
                                                  nan_policy='propagate')

        self.mcmc_result = self.mcmc_minimizer.emcee(init_params, burn=burn, steps=steps, thin=thin, nwalkers=walkers, **kwargs)

    def process_mcmc(self, mcmc_result):
        maxprob = np.argmax(mcmc_result.lnprob)
        for p in self.params:
            self.params[p].value = mcmc_result.flatchain[p][maxprob]
        self.param_error = np.array([np.percentile(mcmc_result.flatchain[p], [15.9, 84.2]) for p in self.params])

    def nested_sample(self, params=None, prior_fn=None, log_dir=None, resume=True, frac_remain=None, step='adaptive', **kwargs):
        try:
            import ultranest
            import ultranest.stepsampler
        except ImportError:
            raise ImportError("nested_sample requires package ultranest to be installed")

        if params is None:
                params = self.params

        var_params = [k for k in params.keys() if params[k].vary]

        def objective(par_arr):
            """
            wrapper around log_likelihood method to evaluate for an array of just the variable parameters,
            which can be used directly with scipy.optimise methods.
            """
            fit_params = copy.copy(params)
            for par, value in zip([p for p in params if params[p].vary], par_arr):
                fit_params[par].value = value
            return self.log_likelihood(fit_params, eval_gradient=False)

        if prior_fn is None:
            def prior_fn(quantiles):
                uniform_dist = lambda q, hi, lo: q * (hi - lo) + lo
                return np.array([uniform_dist(q, params[p].max, params[p].min) for q, p in zip(quantiles, var_params)])

        self.nest_sampler = ultranest.ReactiveNestedSampler(var_params, objective, prior_fn, log_dir=log_dir, resume=resume)

        if step == 'adaptive':
            region_filter = kwargs.pop('region_filter', True)
            self.nest_sampler.run(max_ncalls=40000, **kwargs)
            self.nest_sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=1000,
                                            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                                            adaptive_nsteps='move-distance',
                                            region_filter=region_filter)
        elif isinstance(step, int) and step > 0:
            self.nest_sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=step,
                                            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
                                            adaptive_nsteps=False,
                                            region_filter=False)

        self.nest_result = self.nest_sampler.run(frac_remain=frac_remain, **kwargs)
        self.nest_sampler.print_results()

class MLPSD(MLFit):
    """
    pylag.mlfit.MLPSD

    Class to fit the power spectrum to light curves via the autocovariance matrix.

    Fitting functionality is inherited from the MLFit class.

    Constructor arguments:
    lc: LightCurve, optional (default=None): LightCurve object to for which the power spectrum is to be calculated
    t: array, optional (default=None). If lc=None, array containing the observed time bins
    r: array, optional (default=None). If lc=None, array containing the count rate in each time bin
    e: array, optional (default=None). If lc=None, array containing the error for each time bin
    noise: stry, optional (default='errors'). Method for computing the noise (diagonal) term in the covariace matrix.
    The default is to use the error bars of the original light curve.
    kwargs: additional arguments passed to MLFit constructor.
    """
    def __init__(self, lc=None, t=None, r=None, e=None, noise='errors', **kwargs):
        if lc is not None:
            t = np.array(lc.time)
            r = np.array(lc.rate)
            e = np.array(lc.error)

            # remove the time bins with zero counts
            nonzero = (r > 0)
            t = t[nonzero]
            r = r[nonzero]
            e = e[nonzero]

        # to fit a Gaussian process, we should make sure our data has zero mean
        y = r - np.mean(r)

        dt = np.min(np.diff(t))
        #length = (np.max(t) - np.min(t)) // dt
        length = len(r)

        if noise == 'poisson':
            noise_arr = (1. / lc.mean()**2) * np.ones_like(lc.rate)
        elif noise == 'errors':
            # Note: RMS normalisation
            #noise_arr = e ** 2 * length / (2.*dt)
            noise_arr = e**2
        elif isinstance(noise, (float, int)):
            noise_arr = noise * np.ones_like(lc.rate)
        else:
            noise_arr = None

        # RMS normalisation for PSD
        #psdnorm = (np.mean(r) ** 2 * length) / (2 * dt)
        psdnorm = np.mean(r)**2

        MLFit.__init__(self, t, y, noise=noise_arr, psdnorm=psdnorm, **kwargs)

        self.psd = self.get_psd()
        self.psd_error = None

        self.noise_level = (2. / np.mean(r)**2) * np.ones_like(self.fbins.bin_cent)

    def cov_matrix(self, params):
        """
        c = pylag.mlfit.MLPSD.cov_matrix(params)

        Calculate the model covariance matrix for the specified parameter values.

        :param params: Parameters() object containing the set of parameter values for which the covariance matrix is
        to be calculated
        
        :return: c: ndarray (N * N): the covariance matrix
        """
        # if no model is specified, the PSD model is just the PSD value in each frequency bin
        # note the factor of 2 to integrate over the negative frequencies too!
        if self.model is None:
            psd = np.exp(np.array([params[p].value for p in params])) * self.psdnorm
        else:
            psd = self.model(params, self.fbins.bin_cent) * self.psdnorm

        cov = np.sum(np.array([p * c for p, c in zip(psd, self.cos_integral)]), axis=0)

        return cov

    def cov_matrix_deriv(self, params):
        """
        dc = pylag.mlfit.MLPSD.cov_matrix_deriv(params)

        Calculate the first derivative of the covariance matrix wrt the parameters

        :param params: Parameters() object containing the set of parameter values at which the derivative is to be
        calculated
        
        :return: c: ndarray (N * N * Npar): the derivative of the covariance matrix wrt each parameter
        """
        if self.model is None:
            psd = np.exp(np.array([params[p].value for p in params])) * self.psdnorm

            # in this simple case, the covariance matrix is just a linear sum of each frequency term
            # so the derivative is simple - we multiply by p when we're talking about the log
            return np.stack([c * p for c, p in zip(self.cos_integral, psd)], axis=-1)
        else:
            psd_deriv = self.model.eval_gradient(params, self.fbins.bin_cent) * self.psdnorm
            return np.stack([np.sum([c * p for c, p in zip(self.cos_integral, psd_deriv[:, par])], axis=0) for par in range(psd_deriv.shape[-1])], axis=-1)

    def get_psd(self, params=None):
        """
        psd = pylag.mlfit.MLPSD.get_psd(params)

        Calculate the power spectrum in each frequency bin for a given set of parameters.

        :param params: Parameters, optional (default=None): Parameters object containing the parameters from which to
        calculate the power spectrum. If none, will use the current values of the params member variable (which
        will either be the initial values or the results of the last fit).
        
        :return: psd: ndarray: the power spectrum in each frequency bin
        """
        if params is None:
            params = self.params

        if self.model is None:
            return np.array([self.params[p].value for p in self.params])
        else:
            return np.log(self.model(self.params, self.fbins.bin_cent))

    def process_fit_results(self, fit_result, params):
        """
        pylag.mlfit.MLPSD.process_fit_results(fit_result, params)

        Process a scipy.optimise fit result to calculate the best-fitting power spectrum and error from the model.

        :param fit_result: scipy.optimise fit result to be processed
        :param params: Parameters() object containing the best-fitting parameters (including the frozen parameters,
        which are not included in fit_result.x)
        """
        self.psd = self.get_psd()
        if self.model is None:
            self.psd_error = self.param_error
        else:
            # calculate the error on each PSD point from the error on each parameter
            psd_deriv = self.model.eval_gradient(params, self.fbins.bin_cent)
            self.psd_error = np.sum([e * psd_deriv[..., i] for i, e in enumerate(self.param_error)], axis=0) / self.psd
            if np.any(np.isnan(self.psd_error)):
                self.psd_error = None

    def _getplotdata(self):
        x = (self.fbins.bin_cent, self.fbins.x_error())
        y = (np.exp(self.psd), np.vstack([np.exp(self.psd) - np.exp(self.psd - self.psd_error), np.exp(self.psd + self.psd_error) - np.exp(self.psd)])) if self.psd_error is not None else np.exp(self.psd)
        return x, y

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'PSD', 'log'


class MLCrossSpectrum(MLFit):
    """
    pylag.mlfit.MLCrossSpectrum

    Class to fit the cross spectrum to a pair of light curves via the covariance matrix.

    The MLCrossSpectrum will contain an MLPSD object for each of the light curves, which are used to calculate the
    autocovariance (i.e. top left and bottom right) components of the covariance matrix. The power spectrum of each
    light curve can be fit separately, and these fit results can be used to pre-populate and freeze these parts of the
    matrix (or the whole set can be fit simultaneously).

    Fitting functionality is inherited from the MLFit class.

    Constructor arguments:
    lc1: LightCurve, optional (default=None): First LightCurve object
    lc2: LightCurve, optional (default=None): Second LightCurve object
    mlpsd1: (optional, default=None) MLPSD object which can be pre-fit to the first light curve
    mlpsd2: (optional, default=None) MLPSD object which can be pre-fit to the second light curve
    psd_model: (optional, default=None) Model object to model the power spectra as a function of frequency,
    rather than fitting each bin as a free parameter
    cpsd_model: (optional, default=None) Model object to model the cross spectrum as a function of frequency,
    rather than fitting each bin as a free parameter
    lag_model: (optional, default=None) Model object to model the lag as a function of frequency,
    rather than fitting each bin as a free parameter
    freeze_psd: bool (optional, default=True): If True, the power spectra of each light curve is fit separately
    and the autocovariance components of the matrix are frozen to these values during the fit.
    noise: stry, optional (default='errors'). Method for computing the noise (diagonal) term in the covariace matrix.
    The default is to use the error bars of the original light curve.
    cpsd_model_args: dict (optional, default={}): arguments to pass to constructor for cross spectrum model
    lag_model_args: dict (optional, default={}): arguments to pass to constructor for lag model
    kwargs: additional arguments passed to MLFit constructor.
    """
    def __init__(self, lc1, lc2, mlpsd1=None, mlpsd2=None, psd_model=None, cpsd_model=None, lag_model=None, freeze_psd=True, noise='errors', cpsd_model_args={}, lag_model_args={}, **kwargs):
        t = np.array(lc1.time)

        r1 = np.array(lc1.rate)
        r2 = np.array(lc2.rate)

        e1 = np.array(lc1.error)
        e2 = np.array(lc2.error)

        # remove the time bins where either light curve is zero, since these will cause problems with the likelihood function (due to zero error)
        nonzero = np.logical_and(r1>0, r2>0)
        t = t[nonzero]
        r1 = r1[nonzero]
        r2 = r2[nonzero]
        e1 = e1[nonzero]
        e2 = e2[nonzero]

        y1 = r1 - np.mean(r1)
        y2 = r2 - np.mean(r2)

        # to fit a cross spectrum, we stack the data vectors
        y = np.concatenate([y1, y2])

        #cpsdnorm = (lc1.mean() * lc2.mean() * lc1.length) / (2 * np.min(np.diff(lc1.time)))
        cpsdnorm = np.mean(r1) * np.mean(r2)

        self.ac1 = None
        self.ac2 = None

        if mlpsd1 is not None:
            self.mlpsd1 = mlpsd1
            self.ac1 = self.mlpsd1.cov_matrix(self.mlpsd1.params)
        else:
            self.mlpsd1 = MLPSD(t=t, r=r1, e=e1, noise=noise, model=psd_model, component_name='psd1', **kwargs)

        if mlpsd2 is not None:
            self.mlpsd2 = mlpsd2
            self.ac2 = self.mlpsd2.cov_matrix(self.mlpsd2.params)
        else:
            self.mlpsd2 = MLPSD(t=t, r=r2, e=e2, noise=noise, model=psd_model, component_name='psd2', **kwargs)

        # use the noise terms from the individual power spectra, since they only apply across the leading diagonal
        noise_arr = np.concatenate([self.mlpsd1.noise, self.mlpsd1.noise])

        self.freeze_psd = freeze_psd

        if cpsd_model is None:
            self.cpsd_model = None
        elif isinstance(cpsd_model, Model):
            self.cpsd_model = cpsd_model
        else:
            self.cpsd_model = cpsd_model(component_name='cpsd', **cpsd_model_args)

        if lag_model is None:
            self.lag_model = None
        elif isinstance(lag_model, Model):
            self.lag_model = lag_model
        else:
            self.lag_model = lag_model(component_name='lag', **lag_model_args)

        MLFit.__init__(self, t, y, noise=noise_arr, psdnorm=cpsdnorm, eval_sin=True, **kwargs)

    def get_params(self):
        """
        param = pylag.mlfit.MLCrossSpectrum.get_params()

        Create a new set of parameters for the model used to construct the covariance matrix

        :return: param: Parameters() object containing the parameters
        """
        params = lmfit.Parameters()
        if not self.freeze_psd:
            params += self.mlpsd1.get_params()
            params += self.mlpsd2.get_params()
        if self.cpsd_model is None:
            for i in range(len(self.fbins)):
                params.add('%sln_cpsd%01d' % (self.prefix, i), -1., vary=True, min=-30., max=30.)
        else:
            params += self.cpsd_model.get_params()
        if self.lag_model is None:
            for i in range(len(self.fbins)):
                params.add('%slag%01d' % (self.prefix, i), 0., vary=True, min=-np.pi, max=np.pi)
        else:
            params += self.lag_model.get_params()

        return params

    def fit_psd(self):
        """
        pylag.mlfit.MLCrossSpectrum.fit_psd()

        Perform preliminary fits of the power spectra to the individual light curves. This function will pre-populate
        the stored autocovariance components of the matrix (when the psd is frozen) and will set an initial estimate
        of the cross spectral powers to use as the starting point for fitting the full cross spectral model.

        Unless you're fitting the power and cross spectra simultaneously, this method should be run before running fit().
        """
        print("Fitting PSD of light curve 1...")
        self.mlpsd1.fit()
        self.ac1 = self.mlpsd1.cov_matrix(self.mlpsd1.params)

        print("Fitting PSD of light curve 2...")
        self.mlpsd2.fit()
        self.ac2 = self.mlpsd1.cov_matrix(self.mlpsd2.params)

        # set an initial estimate of the cross power spectrum as the average of the two band powers
        # minus a little bit - this helps the fit on its way!
        for i in range(len(self.fbins)):
            self.params['ln_cpsd%01d' % i].value = 0.5 * (self.mlpsd1.psd[i] + self.mlpsd2.psd[i]) - 1.

    def cross_cov_matrix(self, params):
        """
        cc = pylag.mlfit.MLCrossSpectrum.cross_cov_matrix(params)

        Calculate the cross component of the covariance matrix for the model cross spectrum (i.e. the upper right and
        lower left quadrants of the matrix for the terms that use rates from both light curves).

        :param params: Parameters() object containing the set of parameter values for which the covariance matrix is
        to be calculated

        :return: cc: ndarray (N * N): the cross-covariance matrix
        """
        # if no model is specified, the PSD model is just the PSD value in each frequency bin
        if self.cpsd_model is None:
            cpsd = np.exp(np.array([params['%sln_cpsd%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])) * self.psdnorm
        else:
            cpsd = self.cpsd_model(params, self.fbins.bin_cent) * self.psdnorm

        # likewise for the (phase) lags
        if self.lag_model is None:
            lags = np.array([params['%slag%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
        else:
            lags = self.lag_model(params, self.fbins.bin_cent)

        cov = np.sum(np.array([p * (c * np.cos(phi) - s * np.sin(phi)) for p, c, s, phi in zip(cpsd, self.cos_integral, self.sin_integral, lags)]), axis=0)
        return cov

    def cov_matrix(self, params):
        """
        c = pylag.mlfit.MLCrossSpectrum.cov_matrix(params)

        Calculate the cross spectral model covariance matrix for the specified parameter values. The cross spectral
        covariance matrix is the stack of the autocovariance and cross-covariance matrices, and is applicable to the
        stacked data vector.

        If the freeze_psd member variable is True, the stored autocovariance matrices will be used, otherwise they
        will be computed for the current model parameters.

        :param params: Parameters() object containing the set of parameter values for which the covariance matrix is
        to be calculated

        :return: c: ndarray (2N * 2N): the covariance matrix
        """
        if self.freeze_psd:
            if self.ac1 is None or self.ac2 is None:
                raise AssertionError("Autocovariance matrices are not available. Did you fit the PSDs?")
            ac1 = self.ac1
            ac2 = self.ac2
        else:
            ac1 = self.mlpsd1.cov_matrix(params)
            ac2 = self.mlpsd2.cov_matrix(params)

        cc = self.cross_cov_matrix(params)

        return np.vstack([np.hstack([ac1, cc.T]), np.hstack([cc, ac2])])

    def cross_cov_matrix_deriv(self, params):
        """
        dc = pylag.mlfit.MLPSD.cross_cov_matrix_deriv(params)

        Calculate the first derivative of the cross components of the covariance matrix wrt the parameters

        :param params: Parameters() object containing the set of parameter values at which thederivative is to be
        calculated

        :return: c: ndarray (N * N * Npar): the derivative of the covariance matrix wrt each parameter
        """
        if self.cpsd_model is None:
            # if no model is specified, the PSD model is just the PSD value in each frequency bin
            if self.cpsd_model is None:
                cpsd = np.exp(np.array(
                    [params['%sln_cpsd%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])) * self.psdnorm
            else:
                cpsd = self.cpsd_model(params, self.fbins.bin_cent) * self.psdnorm

            # likewise for the (phase) lags
            if self.lag_model is None:
                lags = np.array([params['%slag%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
            else:
                lags = self.lag_model(params, self.fbins.bin_cent)

            if self.cpsd_model is None:
                cpsd_derivs = [(c * np.cos(phi) + s * np.sin(phi)) * p for p, c, s, phi in zip(cpsd, self.cos_integral, self.sin_integral, lags)]
            else:
                # TODO: implement chain rule derivatives for functions
                return NotImplemented

            if self.lag_model is None:
                lag_derivs = [-1 * p * (c * np.sin(phi) + s * np.cos(phi)) for p, c, s, phi in zip(cpsd, self.cos_integral, self.sin_integral, lags)]
            else:
                # TODO: implement chain rule derivatives for functions
                return NotImplemented

            # this is the stack of (1) the derivatives w.r.t. the cross powers (multiplied by p when we're using the log)
            # and (2) the phases
            return np.stack(cpsd_derivs + lag_derivs, axis=-1)

    def cov_matrix_deriv(self, params):
        """
        dc = pylag.mlfit.MLPSD.cov_matrix_deriv(params)

        Calculate the first derivative of the covariance matrix wrt the parameters by stacking the derivatives of the
        four quadrants of the covariance matrix (the autocovariance and cross-covariance matrices)

        :param params: Parameters() object containing the set of parameter values at which thederivative is to be
        calculated

        :return: c: ndarray (2N * 2N * Npar): the derivative of the covariance matrix wrt each parameter
        """
        cc = self.cross_cov_matrix_deriv(params)

        if self.freeze_psd:
            Z = np.zeros_like(self.ac1)
            return np.stack(
                [np.vstack([np.hstack([Z, cc[..., p].T]), np.hstack([cc[..., p], Z])]) for p in
                 range(len(self.params))], axis=-1)

        else:
            ac1 = self.mlpsd1.cov_matrix_deriv(params)
            ac2 = self.mlpsd2.cov_matrix_deriv(params)
            return np.stack(
                [np.vstack([np.hstack([ac1[..., p], cc[..., p].T]), np.hstack([cc[..., p], ac2[..., p]])]) for p in
                 range(len(self.params))], axis=-1)

        return np.stack([np.vstack([np.hstack([ac1[...,p], cc[...,p].T]), np.hstack([cc[...,p], ac2[...,p]])]) for p in range(len(self.params))], axis=-1)

    def get_cpsd(self, params=None):
        """
        cpsd = pylag.mlfit.MLCrossSpectrum.get_cpsd(params)

        Calculate the cross power spectrum in each frequency bin from the model for a given set of parameters.

        :param params: Parameters, optional (default=None): Parameters object containing the parameters from which to
        calculate the cross power spectrum. If none, will use the current values of the params member variable (which
        will either be the initial values or the results of the last fit).

        :return: cpsd: ndarray: the power spectrum in each frequency bin
        """
        if params is None:
            params = self.params

        if self.cpsd_model is None:
            return np.array([self.params['%sln_cpsd%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
        else:
            return np.log(self.cpsd_model(self.params, self.fbins.bin_cent))

    def get_lag(self, params=None, time_lag=True):
        """
        lag = pylag.mlfit.MLCrossSpectrum.get_cpsd(params)

        Calculate the lag in each frequency bin from the model for a given set of parameters.

        :param params: Parameters, optional (default=None): Parameters object containing the parameters from which to
        calculate the lags. If none, will use the current values of the params member variable (which
        will either be the initial values or the results of the last fit).
        :param time_lag: bool, optional (default=True): If True, return the time lag (in seconds), otherwise return the
        phase lag (in radians)

        :return: lag: ndarray: the lag in each frequency bin
        """
        if params is None:
            params = self.params

        if self.lag_model is None:
            lag = np.array([self.params['%slag%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
        else:
            lag = np.log(self.cpsd_model(self.params, self.fbins.bin_cent))

        return lag / (2. * np.pi * self.fbins.bin_cent) if time_lag else lag

    def process_fit_results(self, fit_result, params):
        """
        pylag.mlfit.MLCrossSpectrum.process_fit_results(fit_result, params)

        Process a scipy.optimise fit result to calculate the best-fitting cross spectrum, lag spectrum and errors
        from the model.

        :param fit_result: scipy.optimise fit result to be processed
        :param params: Parameters() object containing the best-fitting parameters (including the frozen parameters,
        which are not included in fit_result.x)
        """
        hess = fit_result.hess_inv(fit_result.x) if callable(fit_result.hess_inv) else np.diag(fit_result.hess_inv)

        self.cpsd = self.get_cpsd()
        if self.cpsd_model is None:
            self.cpsd_error = hess[:len(self.fbins)] ** 0.5
        else:
            return NotImplemented
            # # calculate the error on each PSD point from the error on each parameter
            # psd_deriv = self.model.eval_gradient(params, self.fbins.bin_cent)
            # self.psd_error = np.sum([e * psd_deriv[..., i] for i, e in enumerate(self.param_error)], axis=0) / self.psd
            # if np.any(np.isnan(self.psd_error)):
            #     self.psd_error = None

        self.lag = self.get_lag()
        if self.cpsd_model is None:
            self.lag_error = hess[len(self.fbins):] ** 0.5 / (2. * np.pi * self.fbins.bin_cent)
        else:
            return NotImplemented


class StackedMLPSD(MLPSD):
    """
    pylag.mlfit.StackedMLPSD

    Class for simultaneosuly fitting the power spectrum to multiple lgith curve segments.

    The StackedMLPSD constains MLPSD objects for each light curve (in the mlpsd member variable). The log(likelihood)
    is evaluated as the sum of log(likelihood) for each light curve, and is evaluated using the parameter values tied
    between the set of light cures.

    Constructor arguments:
    :param lclist: list of LightCurve objects to fit
    :param Nf: int, optional (default=10): number of frequency bins to use in the fit between the minimum and Nyquist
    frequencies. The lowest minimum and highest Nyquist frequencies are selected across the set of light curves
    :param fbins: Binning, optional (default=None). A Binning object to use defining the frequency bins, instead of
    constructing bins automatically (this will override Nf)
    :param kwargs: Arguments passed to MLPSD constructor for each of th elight curves
    """
    def __init__(self, lclist, Nf=10, fbins=None, model=None, model_args={}, component_name=None, extend_freq=None, **kwargs):
        if fbins is None:
            # set up frequency bins to span min and max frequencies for the entire list
            T = np.max([lc.time.max() - lc.time.min() for lc in lclist])
            dt = np.min([np.min(np.diff(lc.time)) for lc in lclist])
            min_freq = (1. / T)
            max_freq = 0.5 / dt
            self.fbins = LogBinning(min_freq, max_freq, Nf)
        else:
            self.fbins = fbins

        if extend_freq is not None:
            # create a new set of bins with an extra one at the start, going down to extend_freq * the previous minimum
            self.fbins = Binning(bin_edges=np.insert(self.fbins.bin_edges, 0, extend_freq*self.fbins.bin_edges.min()))

        self.mlpsd = [MLPSD(lc, fbins=self.fbins, model=model, **kwargs) for lc in lclist]

        if model is None:
            self.model = None
        elif isinstance(model, Model):
            self.model = model
        else:
            self.model = model(component_name=component_name, **model_args)

        self.params = self.get_params()

        self.freq = self.fbins.bin_cent
        self.freq_error = self.fbins.x_error()

        self.psd = None
        self.psd_error = None

        self.mcmc_minimizer = None
        self.mcmc_result = None

    def get_params(self):
        """
        param = pylag.mlfit.StackedMLPSD.get_params()

        Create a new set of parameters for the model used to construct the covariance matrix. Note that the same
        parameter values are used to evaluate the covariance matrix for each of the light curves.

        :return: param: Parameters() object containing the parameters
        """
        return self.mlpsd[0].get_params()

    def log_likelihood(self, params, eval_gradient=True):
        """
        loglike, grad = pylag.mlfit.MLFit.log_likelihood(params, eval_gradient=True)

        Evaluate log(marginal likelihood), as well as its gradient, for the covariance matrix defined by some set of
        input parameters. The log(likelihood) for the stack of light curves is the sum of the log(likelihood)
        evaluated for each light curve
        
        :return: mloglike: float: log(likelihood) value, grad: ndarray: derivative of -log(likelihood)
        """
        if eval_gradient:
            segment_loglike = [p.log_likelihood(params, eval_gradient) for p in self.mlpsd]
            # separate and sum the likelihoods and the gradients
            like = np.array([l[0] for l in segment_loglike])
            grad = np.array([l[1] for l in segment_loglike])
            if np.all(np.isfinite(like)):
                return np.sum(like), grad.sum(axis=0)
            else:
                return (-1e6, np.zeros(len(params)) + 1e6)
        else:
            return np.sum([p.log_likelihood(params, eval_gradient) for p in self.mlpsd])
