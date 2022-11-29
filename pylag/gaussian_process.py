"""
pylag.gaussian_process

Provides pyLag functionality for fitting light curves using gaussian processes

Classes
-------


v1.0 09/03/2017 - D.R. Wilkins
"""
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, WhiteKernel, ConstantKernel as C
import re
import emcee

from .lightcurve import *
from .periodogram import *
from .cross_spectrum import *
from .lag_frequency_spectrum import *
from .lag_energy_spectrum import *


class GPLightCurve(LightCurve):
    def __init__(self, filename=None, t=[], r=[], e=[], lc=None, zero_nan=True, kernel='rq', n_restarts_optimizer=9, run_fit=True, use_errors=True, noise_kernel=False, max_noise=2., freeze_noise=False, lognorm=False, remove_gaps=True, remove_nan=False, zero_time=False, normalise=True):
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
            t -= t.min()

        LightCurve.__init__(self, filename, t, r, e, interp_gaps=False, zero_nan=zero_nan, trim=False)

        # need to remove the gaps from the light curve
        # (only store the non-zero time bins)
        if remove_gaps:
            self.remove_gaps(to_self=True)
        elif remove_nan:
            self.remove_nan(to_self=True)

        self.mean_rate = self.mean()

        if isinstance(kernel, str):
            if kernel == 'rq':
                self.kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(1.0, 1.0, (1e-10, 1e10), (1e-10, 1e10))
            elif kernel == 'se':
                self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-10, 1e10))
            elif kernel == 'matern12':
                self.kernel = C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-10, 1e10), 0.5)
            elif kernel == 'matern32':
                self.kernel = C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-10, 1e10), 1.5)
            elif kernel == 'matern52':
                self.kernel = C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-10, 1e10), 2.5)
        else:
            self.kernel = kernel

        if noise_kernel:
            if lognorm:
                noise_level = np.sqrt(self.mean_rate * self.dt) / (
                            self.mean_rate * self.dt * np.log(self.mean_rate * self.dt))
            else:
                noise_level = np.sqrt(self.mean_rate * self.dt) / (self.mean_rate * self.dt)
            if freeze_noise:
                noise_bounds = (noise_level, noise_level)
            else:
                noise_bounds = (0.1*noise_level, max_noise*noise_level)
            self.kernel += WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_bounds)

        if noise_kernel or not use_errors:
            self.gp_regressor = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalise)
        else:
            alpha = (self.error / self.rate)**2
            self.gp_regressor = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts_optimizer,
                                                         normalize_y=normalise, alpha=alpha)

        # get a list of the kernel parameter names
        # r = re.compile('(k[0-9]+__)+(.+(?<!_bounds)(?<!_k[0-9]))$')
        # self.par_names = list(filter(r.match, self.gp_regressor.kernel.get_params().keys()))
        r = re.compile('(.*?)_bounds$')
        self.par_names = [s.replace('_bounds', '') for s in list(filter(r.match, self.gp_regressor.kernel.get_params().keys()))]

        self.lognorm = lognorm
        if self.lognorm:
            self.error = self.error / self.rate
            self.rate = np.log(self.rate)

        self.sampler = None
        self.log_mcmc = False

        if run_fit:
            self.fit()

    def fit(self):
        t = np.atleast_2d(self.time).T
        self.gp_regressor.fit(t, self.rate)

    def predict(self, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        t_samples = np.atleast_2d(t).T
        r, e = self.gp_regressor.predict(t_samples, return_std=True)
        if self.lognorm:
            r = np.exp(r)
            e = r * e
        return LightCurve(t=t, r=r, e=e)

    def sample(self, n_samples=1, t=None, sample_posterior=False):
        if sample_posterior:
            return self.sample_posterior(n_samples, t)

        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        t_samples = np.atleast_2d(t).T
        r = self.gp_regressor.sample_y(t_samples, n_samples=n_samples, random_state=None)
        e = np.zeros(t.shape)
        if n_samples == 1:
            if self.lognorm:
                return LightCurve(t=t, r=np.exp(r.ravel()), e=e)
            else:
                return LightCurve(t=t, r=r.ravel(), e=e)
        else:
            if self.lognorm:
                return [LightCurve(t=t, r=np.exp(r[:, n]), e=e) for n in range(n_samples)]
            else:
                return [LightCurve(t=t, r=r[:,n], e=e) for n in range(n_samples)]

    def make_param_dict(self, par_values, log_par=True):
        if log_par:
            par_values = np.exp(np.array(par_values))
        return dict(zip(self.par_names, par_values))

    def get_fit_param(self, log_par=True):
        par_array = np.array([self.gp_regressor.kernel_.get_params()[p] for p in self.par_names])
        if log_par:
            return np.log(par_array)
        else:
            return par_array

    def get_param_bounds(self, log_par):
        lower = []
        upper = []

        for l, u in [self.gp_regressor.kernel_.get_params()['%s_bounds' % p] for p in self.par_names]:
            lower.append(l)
            upper.append(u)

        lower = np.array(lower)
        upper = np.array(upper)

        if log_par:
            return np.log(lower), np.log(upper)
        else:
            return lower, upper

    def run_mcmc(self, nsteps=2000, nburn=500, log_par=True):
        def log_probability(params, y, gp, lower=None, upper=None):
            if np.any(np.isinf(params)) or np.any(np.isnan(params)):
                return -np.inf

            if upper is not None and np.any(np.greater(params, upper)):
                return -np.inf
            if lower is not None and np.any(np.less(params, lower)):
                return -np.inf

            try:
                if log_par:
                    p = gp.log_marginal_likelihood(np.exp(params))
                else:
                    p = gp.log_marginal_likelihood(params)
            except ValueError:
                print("ValueError for ", p)
                p = -np.inf
            return p

        self.log_mcmc = log_par

        initial = self.get_fit_param(log_par=log_par)
        lower, upper = self.get_param_bounds(log_par=log_par)

        ndim, nwalkers = len(initial), 32
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(self.rate, self.gp_regressor, lower, upper))

        print("Running burn-in...")
        p0 = initial + 1e-5 * np.random.randn(nwalkers, ndim)
        p0, lp, _ = self.sampler.run_mcmc(p0, nburn)

        print("Running chain...")
        self.sampler.reset()
        self.sampler.run_mcmc(p0, nsteps)

    def predict_posterior(self, n_samples=1, t=None):
        if self.sampler is None:
            raise AssertionError("Must run_mcmc before making posterior predictions!")

        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        t_samples = np.atleast_2d(t).T
        e = np.zeros(t.shape)

        samples = self.sampler.flatchain

        if n_samples == 1:
            s = samples[np.random.randint(len(samples))]
            self.gp_regressor.kernel_.set_params(**self.make_param_dict(s, self.log_mcmc))
            r = self.gp_regressor.predict(t_samples, return_std=False)
            if self.lognorm:
                return LightCurve(t=t, r=np.exp(r), e=e)
            else:
                return LightCurve(t=t, r=r, e=e)
        else:
            lclist = []
            for s in samples[np.random.randint(len(samples), size=n_samples)]:
                self.gp.set_parameter_vector(s)
                r = self.gp_regressor.predict(t_samples, return_std=False)
                if self.lognorm:
                    lclist.append(LightCurve(t=t, r=np.exp(r), e=e))
                else:
                    lclist.append(LightCurve(t=t, r=r, e=e))
            return lclist

    def sample_posterior(self, n_samples=1, t=None):
        if self.sampler is None:
            raise AssertionError("Must run_mcmc before sampling the posterior!")

        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        t_samples = np.atleast_2d(t).T
        e = np.zeros(t.shape)

        samples = self.sampler.flatchain

        if n_samples == 1:
            s = samples[np.random.randint(len(samples))]
            self.gp_regressor.kernel_.set_params(**self.make_param_dict(s, log_par=self.log_mcmc))
            r = self.gp_regressor.sample_y(t_samples, n_samples=1, random_state=None)
            if self.lognorm:
                return LightCurve(t=t, r=np.exp(r.ravel()), e=e)
            else:
                return LightCurve(t=t, r=r.ravel(), e=e)
        else:
            lclist = []
            for s in samples[np.random.randint(len(samples), size=n_samples)]:
                self.gp_regressor.kernel_.set_params(**self.make_param_dict(s, log_par=self.log_mcmc))
                r = self.gp_regressor.sample_y(t_samples, n_samples=1, random_state=None)
                if self.lognorm:
                    lclist.append(LightCurve(t=t, r=np.exp(r.ravel()), e=e))
                else:
                    lclist.append(LightCurve(t=t, r=r.ravel(), e=e))
            return lclist


class GPPeriodogram(Periodogram):
    def __init__(self, lc=None, gplc=None, n_samples=10, bins=None, sample_posterior=False):
        if gplc is not None:
            self.gplc = gplc
        elif lc is not None:
            self.gplc = GPLightCurve(lc=lc)
            self.gplc.fit()
        else:
            raise ValueError("GPPeriodogram requires a light curve!")

        freq, freq_error, per, err = self.calculate(n_samples, bins, sample_posterior=sample_posterior)
        Periodogram.__init__(self, f=freq, per=per, err=err, ferr=freq_error)

    def calculate(self, n_samples=10, bins=None, sample_posterior=False):
        sample_lcs = self.gplc.sample(t=None, n_samples=n_samples, sample_posterior=sample_posterior)
        if not isinstance(sample_lcs, list):
            sample_lcs = [sample_lcs]

        if bins is not None:
            freq = bins.bin_cent
            freq_error = bins.bin_end - bins.bin_cent
            per = np.array([Periodogram(lc).bin(bins).periodogram for lc in sample_lcs])
        else:
            freq = sample_lcs[0].ftfreq()
            freq_error = None
            per = np.array([Periodogram(lc).periodogram for lc in sample_lcs])

        per_avg = np.mean(per, axis=0)
        per_std = np.std(per, axis=0)

        return freq, freq_error, per_avg, per_std


class GPEnergyLCList(EnergyLCList):
    def __init__(self, searchstr=None, lcfiles=None, enlclist=None, enmin=None, enmax=None, lclist=None, concatenate=True, lc_class=GPLightCurve, **kwargs):
        if enlclist is not None:
            self.enmin = enlclist.enmin
            self.enmax = enlclist.enmax
            lclist = enlclist.lclist
        elif lclist is not None and enmin is not None and enmax is not None:
            self.enmin = np.array(enmin)
            self.enmax = np.array(enmax)
        elif lclist is None:
            self.enmin, self.enmax, lclist = self.find_light_curves(searchstr, lcfiles)

        self.en = 0.5*(self.enmin + self.enmax)
        self.en_error = self.en - self.enmin

        self.lclist = []

        self.mcmc_run = False

        if isinstance(lclist[0], list):
            for en_lclist in lclist:
                if concatenate:
                    concat_lc = LightCurve().concatenate(en_lclist).sort_time()
                    self.lclist.append(lc_class(lc=concat_lc, **kwargs))
                else:
                    self.lclist.append([])
                    for lc in en_lclist:
                        self.lclist[-1].append(lc_class(lc=lc, **kwargs))

        elif isinstance(lclist[0], LightCurve):
            for lc in lclist:
                self.lclist.append(lc_class(lc=lc, **kwargs))

    def fit(self):
        if isinstance(lclist[0], list):
            for en_lclist in self.lclist:
                for lc in en_lclist:
                    lc.fit()

        elif isinstance(lclist[0], LightCurve):
            for lc in self.lclist:
                lc.fit()

    def run_mcmc(self, nsteps=2000, nburn=500, log_par=True):
        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                for lc in en_lclist:
                    lc.run_mcmc(nsteps=nsteps, nburn=nburn, log_par=log_par)

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                lc.run_mcmc(nsteps=nsteps, nburn=nburn, log_par=log_par)

        self.mcmc_run = True

    def sample(self, n_samples=1, t=None, sample_posterior=False):
        if n_samples == 1:
            sample_lclist = []
            if isinstance(self.lclist[0], list):
                for en_lclist in self.lclist:
                    sample_lclist.append([])
                    for lc in en_lclist:
                        sample_lclist[-1].append(lc.sample(n_samples=1, t=t, sample_posterior=sample_posterior))

            elif isinstance(self.lclist[0], LightCurve):
                for lc in self.lclist:
                    sample_lclist.append(lc.sample(n_samples=1, t=t, sample_posterior=sample_posterior))

            return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=sample_lclist)

        elif n_samples > 1:
            sample_lclist = []
            if isinstance(self.lclist[0], list):
                for en_lclist in self.lclist:
                    sample_lclist.append([])
                    for lc in en_lclist:
                        sample_lclist[-1].append(lc.sample(n_samples=1, t=t, sample_posterior=sample_posterior))

            elif isinstance(self.lclist[0], LightCurve):
                for lc in self.lclist:
                    sample_lclist.append(lc.sample(n_samples=1, t=t, sample_posterior=sample_posterior))

            sample_lclist = lclist_separate_segments(sample_lclist)

            return [EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=lclist) for lclist in sample_lclist]


class GPLagFrequencySpectrum(LagFrequencySpectrum):
    def __init__(self, bins, lc1=None, lc2=None, gplc1=None, gplc2=None, n_samples=10, low_mem=False, sample_posterior=False, nsteps=2000, nburn=500, log_mcmc=True):
        if gplc1 is not None:
            self.gplc1 = gplc1
        elif lc1 is not None:
            self.gplc1 = GPLightCurve(lc=lc1, run_fit=True)
        else:
            raise ValueError("GPLagFrequencySpectrum requires a pair of light curves!")

        if gplc2 is not None:
            self.gplc2 = gplc2
        elif lc2 is not None:
            self.gplc2 = GPLightCurve(lc=lc2, run_fit=True)
        else:
            raise ValueError("GPLagFrequencySpectrum requires a pair of light curves!")

        if sample_posterior and self.gplc1.sampler is None:
            self.gplc1.run_mcmc(nsteps, nburn, log_par=log_mcmc)
        if sample_posterior and self.gplc2.sampler is None:
            self.gplc2.run_mcmc(nsteps, nburn, log_par=log_mcmc)

        self.lag_samples = None

        if low_mem:
            self.freq, self.freq_error, self.lag, self.error = self.calculate_seq(bins, n_samples, sample_posterior=sample_posterior)
        else:
            self.freq, self.freq_error, self.lag, self.error = self.calculate_batch(bins, n_samples, sample_posterior=sample_posterior)

    def calculate_batch(self, bins, n_samples=10, sample_posterior=False, save_samples=False):
        sample_lcs1 = self.gplc1.sample(t=None, n_samples=n_samples, sample_posterior=sample_posterior)
        sample_lcs2 = self.gplc2.sample(t=None, n_samples=n_samples, sample_posterior=sample_posterior)
        if not isinstance(sample_lcs1, list):
            sample_lcs1 = [sample_lcs1]
        if not isinstance(sample_lcs2, list):
            sample_lcs2 = [sample_lcs2]

        freq = bins.bin_cent
        freq_error = bins.bin_end - bins.bin_cent
        lag = np.array([LagFrequencySpectrum(bins, lc1, lc2, calc_error=False).lag for (lc1, lc2) in zip(sample_lcs1, sample_lcs2)])

        if save_samples:
            self.lag_samples = lag

        lag_avg = np.mean(lag, axis=0)
        lag_std = np.std(lag, axis=0)

        return freq, freq_error, lag_avg, lag_std

    def calculate_seq(self, bins, n_samples=10, sample_posterior=False, save_samples=False):
        freq = bins.bin_cent
        freq_error = bins.bin_end - bins.bin_cent

        lag = []
        for n in range(n_samples):
            sample_lc1 = self.gplc1.sample(t=None, n_samples=1, sample_posterior=sample_posterior)
            sample_lc2 = self.gplc2.sample(t=None, n_samples=1, sample_posterior=sample_posterior)
            lag.append(LagFrequencySpectrum(bins, sample_lc1, sample_lc2, calc_error=False).lag)

        lag = np.array(lag)

        if save_samples:
            self.lag_samples = lag

        lag_avg = np.mean(lag, axis=0)
        lag_std = np.std(lag, axis=0)

        return freq, freq_error, lag_avg, lag_std


class GPLagEnergySpectrum(LagEnergySpectrum):
    def __init__(self, fmin, fmax, lclist=None, gplclist=None, n_samples=10, refband=None, low_mem=False, save_samples=False, sample_posterior=False, nsteps=2000, nburn=500, log_mcmc=True):
        if gplclist is not None:
            self.gplclist = gplclist
        elif lclist is not None:
            self.gplclist = GPEnergyLCList(enlclist=lclist, run_fit=True)
        else:
            raise ValueError("GPLagFrequencySpectrum requires a light curve list!")

        self.en = np.array(self.gplclist.en)
        self.en_error = np.array(self.gplclist.en_error)

        if sample_posterior and not self.gplclist.mcmc_run:
            self.gplclist.run_mcmc(nsteps=nsteps, nburn=nburn, log_par=log_mcmc)

        self.lag_samples = None

        if low_mem:
            self.lag, self.error = self.calculate_seq(fmin, fmax, n_samples, refband, save_samples, sample_posterior=sample_posterior)
        else:
            self.lag, self.error = self.calculate_batch(fmin, fmax, n_samples, refband, save_samples, sample_posterior=sample_posterior)

    def calculate_batch(self, fmin, fmax, n_samples=10, refband=None, save_samples=False, sample_posterior=False):
        sample_lclists = self.gplclist.sample(t=None, n_samples=n_samples, sample_posterior=sample_posterior)
        if not isinstance(sample_lclists, list):
            sample_lclists = [sample_lclists]

        lag = np.array([LagEnergySpectrum(fmin, fmax, lclist, refband=refband, calc_error=False).lag for lclist in sample_lclists])

        lag_avg = np.mean(lag, axis=0)
        lag_std = np.std(lag, axis=0)

        if save_samples:
            self.lag_samples = lag

        return lag_avg, lag_std

    def calculate_seq(self, fmin, fmax, n_samples=10, refband=None, save_samples=False, sample_posterior=False):
        lag = []
        for n in range(n_samples):
            if n % int(n_samples/10) == 0:
                print('Sample %d/%d' % (n, n_samples))
            try:
                sample_lclist = self.gplclist.sample(t=None, n_samples=1, sample_posterior=sample_posterior)
            except:
                continue
            lag.append(LagEnergySpectrum(fmin, fmax, sample_lclist, refband=refband, calc_error=False).lag)

        lag = np.array(lag)
        lag_avg = np.mean(lag, axis=0)
        lag_std = np.std(lag, axis=0)

        if save_samples:
            self.lag_samples = lag

        return lag_avg, lag_std
