"""
pylag.gaussian_process

Provides pyLag functionality for fitting light curves using gaussian processes

Classes
-------


v1.0 09/03/2017 - D.R. Wilkins
"""
import numpy as np


from .lightcurve import *
from .gaussian_process import *
import celerite
from celerite import terms
from scipy.optimize import minimize
import emcee


class CeleriteLightCurve(GPLightCurve):
    def __init__(self, filename=None, t=[], r=[], e=[], lc=None, zero_nan=True, num_terms=1, kernel=None, run_fit=True, use_errors=True, noise_kernel=False, fit_noise=True, fit_mean=False, norm=True, lognorm=False, remove_gaps=True, remove_nan=False, zero_time=True, **kwargs):
        if lc is not None:
            if isinstance(lc, list):
                # if we're passed a list, concatenate them into a single LightCurve
                concat_lc = LightCurve().concatenate(lc).sort_time()
                t = concat_lc.time
                r = concat_lc.rate
                e = concat_lc.error
            else:
                t = lc.time
                r = lc.rate
                e = lc.error

        if zero_time:
            t = np.array(t)
            self.tstart = t.min()
            t -= t.min()

        LightCurve.__init__(self, filename, t, r, e, interp_gaps=False, zero_nan=zero_nan, trim=False)


        # need to remove the gaps from the light curve
        # (only store the non-zero time bins)
        if remove_gaps:
            self.remove_gaps(to_self=True)
        elif remove_nan:
            self.remove_nan(to_self=True)

        self.mean_rate = np.mean(self.rate)
        self.std_rate = np.std(self.rate)
        self.var = np.var(self.rate)

        self.sampler = None

        self.lognorm = lognorm
        if self.lognorm:
            self.error = self.error / self.rate
            self.rate = np.log(self.rate)

        self.norm = norm
        if self.norm:
            self.rate = (self.rate - self.rate.mean()) / self.rate.std()
            self.error = self.error / self.rate.std()  # assuming no uncertainty in mean

        self.use_errors = use_errors

        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = terms.RealTerm(1, -10)
            for i in range(num_terms - 1):
                self.kernel += terms.RealTerm(1, -10)
            if noise_kernel:
                self.kernel += terms.JitterTerm(np.log10(np.mean(self.error)))

        self.gp = celerite.GP(kernel, mean=np.mean(self.rate), fit_mean=fit_mean, fit_white_noise=fit_noise)

        if run_fit:
            self.fit()

    def _compute(self):
        if self.use_errors:
            self.gp.compute(self.time, self.error)
        else:
            self.gp.compute(self.time)

    def fit(self, restarts=0):
        initial_params = self.gp.get_parameter_vector()
        bounds = self.gp.get_parameter_bounds()

        def gp_minus_log_likelihood(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)

        def gp_grad_minus_log_likelihood(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.grad_log_likelihood(y)[1]

        r = minimize(gp_minus_log_likelihood, initial_params, jac=gp_grad_minus_log_likelihood, method="L-BFGS-B", bounds=bounds, args=(self.rate, self.gp))
        for i in range(restarts):
            print("restarting fit from ", r.x)
            new_start = []
            for par in r.x:
                new_start.append(par + 0.1*par*scipy.randn())
            r = minimize(gp_minus_log_likelihood, new_start, jac=gp_grad_minus_log_likelihood, method="L-BFGS-B", bounds=bounds, args=(self.rate, self.gp))
        print(r)
        self.gp.set_parameter_vector(r.x)
        self._compute()
        return r

    def run_mcmc(self, nsteps=2000, nburn=500):
        def log_probability(params, y, gp):
            gp.set_parameter_vector(params)
            lp = gp.log_prior()
            if not np.isfinite(lp):
                return -np.inf
            return gp.log_likelihood(y) + lp

        r = self.fit()
        initial = np.array(r.x)

        ndim, nwalkers = len(initial), np.max([32, 2*len(initial)])
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(self.rate, self.gp))

        print("Running burn-in...")
        p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = self.sampler.run_mcmc(p0, nburn)

        print("Running chain...")
        self.sampler.reset()
        self.sampler.run_mcmc(p0, nsteps)

    def predict(self, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))

        r, var = self.gp.predict(self.rate, t, return_var=True)
        e = np.sqrt(var)
        if self.lognorm:
            r = np.exp(r)
            e = r * e
        return LightCurve(t=t, r=r, e=e)

    def predict_posterior(self, n_samples=1, t=None):
        if self.sampler is None:
            raise AssertionError("Must run_mcmc before making posterior predictions!")

        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        e = np.zeros(t.shape)

        samples = self.sampler.flatchain

        if n_samples == 1:
            s = samples[np.random.randint(len(samples))]
            self.gp.set_parameter_vector(s)
            r = self.gp.predict(self.rate, t, return_cov=False)
            if self.lognorm:
                return LightCurve(t=t, r=np.exp(r), e=e)
            else:
                return LightCurve(t=t, r=r, e=e)
        else:
            lclist = []
            for s in samples[np.random.randint(len(samples), size=n_samples)]:
                self.gp.set_parameter_vector(s)
                r = self.gp.predict(self.rate, t, return_cov=False)
                if self.lognorm:
                    lclist.append(LightCurve(t=t, r=np.exp(r), e=e))
                else:
                    lclist.append(LightCurve(t=t, r=r, e=e))
            return lclist

    def sample(self, n_samples=1, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        r = self.gp.sample_conditional(self.rate, t, n_samples)
        e = np.zeros_like(t)
        if n_samples == 1:
            if self.lognorm:
                return LightCurve(t=t, r=np.exp(r[0]), e=e)
            else:
                return LightCurve(t=t, r=r[0], e=e)
        else:
            if self.lognorm:
                return [LightCurve(t=t, r=np.exp(r[n, :]), e=e) for n in range(n_samples)]
            else:
                return [LightCurve(t=t, r=r[n,:], e=e) for n in range(n_samples)]

    def sample_posterior(self, n_samples=1, t=None):
        if self.sampler is None:
            raise AssertionError("Must run_mcmc before sampling the posterior!")

        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        e = np.zeros_like(t)

        samples = self.sampler.flatchain

        if n_samples == 1:
            s = samples[np.random.randint(len(samples))]
            self.gp.set_parameter_vector(s)
            r = self.gp.sample_conditional(self.rate, t, 1)
            if self.lognorm:
                return LightCurve(t=t, r=np.exp(r[0]), e=e)
            else:
                return LightCurve(t=t, r=r[0], e=e)
        else:
            lclist = []
            for s in samples[np.random.randint(len(samples), size=n_samples)]:
                self.gp.set_parameter_vector(s)
                r = self.gp.sample_conditional(self.rate, t, 1)
                if self.lognorm:
                    lclist.append(LightCurve(t=t, r=np.exp(r[0]), e=e))
                else:
                    lclist.append(LightCurve(t=t, r=r[0], e=e))
            return lclist

    def psd(self, freq=None, Nf=100):
        if freq is None:
            freq = LogBinning(1./self.time.max(), 1./(2.*np.min(np.diff(self.time))), Nf).bin_cent
        p = self.kernel.get_psd(2.*np.pi*freq)
        return DataSeries(freq, p, xlabel='Frequency / Hz', xscale='log', ylabel='PSD', yscale='log')

    def psd_posterior(self, freq=None, Nf=100, n_samples=100):
        if freq is None:
            freq = LogBinning(1./self.time.max(), 1./(2.*np.min(np.diff(self.time))), Nf).bin_cent

        samples = self.sampler.flatchain

        psd_samples = []
        for s in samples[np.random.randint(len(samples), size=n_samples)]:
            self.gp.set_parameter_vector(s)
            psd_samples.append(self.kernel.get_psd(2.*np.pi*freq))

        psd_samples = np.vstack(psd_samples)
        psd_mean = np.mean(psd_samples, axis=0)
        psd_std = np.std(psd_samples, axis=0)

        return DataSeries(freq, (psd_mean, psd_std), xlabel='Frequency / Hz', xscale='log', ylabel='PSD', yscale='log')

