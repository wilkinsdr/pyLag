"""
pylag.lens_lag

Implement maximum likelihood determination of time lags among a series of light curves from a lensed quasar

v1.0 13/08/2021 - D.R. Wilkins
"""
import lmfit
import numpy as np

from .mlfit import *


class LensML_lag_offset_model(FFTCorrelationModel):
    def get_params(self, norm=None, slope=2., lag=0.):
        if norm is None:
            norm = -7 if self.log_psd else 1e-3
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=(-50 if self.log_psd else 1e-10), max=(50 if self.log_psd else 1e10))
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        params.add('lag', value=lag, min=-1e4, max=+1e4)
        params.add('offset', value=lag, min=-1e4, max=+1e4)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = np.exp(params['%snorm' % self.prefix].value) if self.log_psd else params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        lag = params['lag'].value
        offset = params['offset'].value

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]
        phase = 2 * np.pi * freq_arr * (lag + offset)

        ft = psd * np.exp(-1j * phase)
        return ft


class LensML_lag_model(FFTCorrelationModel):
    def get_params(self, norm=None, slope=2., lag=0.):
        if norm is None:
            norm = -7 if self.log_psd else 1e-3
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=(-50 if self.log_psd else 1e-10), max=(50 if self.log_psd else 1e10))
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        #params.add('lag', value=lag, min=-1e4, max=+1e4)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = np.exp(params['%snorm' % self.prefix].value) if self.log_psd else params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        lag = params['lag'].value

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]
        phase = 2 * np.pi * freq_arr * lag

        ft = psd * np.exp(-1j * phase)
        return ft


class LensML_offset_model(FFTCorrelationModel):
    def get_params(self, norm=None, slope=2., lag=0.):
        if norm is None:
            norm = -7 if self.log_psd else 1e-3
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=(-50 if self.log_psd else 1e-10), max=(50 if self.log_psd else 1e10))
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        #params.add('offset', value=lag, min=-1e4, max=+1e4)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = np.exp(params['%snorm' % self.prefix].value) if self.log_psd else params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        offset = params['offset'].value

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]
        phase = 2 * np.pi * freq_arr * offset

        ft = psd * np.exp(-1j * phase)
        return ft


class LensMLCovariance(MLCovariance):
    def __init__(self, lc1list_a, lc2list_a, lc1list_b, lc2list_b, reflclist_a, reflclist_b, noise='mean_error', params=None, **kwargs):
        if params is not None:
            self.params = params
        else:
            self.params = lmfit.Parameters()

            cov_matrix_a = CrossCovarianceMatrixModel(FFTAutoCorrelationModel_plpsd, LensML_lag_model,
                                                    lc1list_a[0].time, lc2list_a[0].time, noise1=1., noise2=1., prefix='a_', **kwargs)
            self.params += cov_matrix_a.get_params()

            cov_matrix_b = CrossCovarianceMatrixModel(FFTAutoCorrelationModel_plpsd, LensML_lag_model,
                                                      lc1list_b[0].time, lc2list_b[0].time, noise1=1., noise2=1.,
                                                      prefix='b_', **kwargs)
            self.params += cov_matrix_b.get_params()

            cov_matrix_ab = CrossCovarianceMatrixModel(FFTAutoCorrelationModel_plpsd, LensML_lag_offset_model,
                                                      lc1list_a[0].time, lc2list_a[0].time, noise1=1., noise2=1.,
                                                      prefix='ab_', **kwargs)
            self.params += cov_matrix_ab.get_params()

            cov_matrix_ref = CrossCovarianceMatrixModel(FFTAutoCorrelationModel_plpsd, LensML_offset_model,
                                                      lc1list_a[0].time, lc2list_a[0].time, noise1=1., noise2=1.,
                                                      prefix='ref_', **kwargs)
            self.params += cov_matrix_ref.get_params()

        self.ml_covariance_a = [
            MLCrossCovariance(lc1, lc2, FFTAutoCorrelationModel_plpsd, LensML_lag_model, noise1=noise,
                              noise2=noise, params=self.params, prefix='a_', **kwargs)
            for lc1, lc2 in zip(lc1list_a, lc2list_a)]

        self.ml_covariance_b = [
            MLCrossCovariance(lc1, lc2, FFTAutoCorrelationModel_plpsd, LensML_lag_model, noise1=noise,
                              noise2=noise, params=self.params, prefix='b_', **kwargs)
            for lc1, lc2 in zip(lc1list_b, lc2list_b)]

        self.ml_covariance_ab = [
            MLCrossCovariance(lc1, lc2, FFTAutoCorrelationModel_plpsd, LensML_lag_offset_model, noise1=noise,
                              noise2=noise, params=self.params, prefix='ab_', **kwargs)
            for lc1, lc2 in zip(lc1list_a, lc2list_b)]

        self.ml_covariance_ref = [
            MLCrossCovariance(lc1, lc2, FFTAutoCorrelationModel_plpsd, LensML_offset_model, noise1=noise,
                              noise2=noise, params=self.params, prefix='ref_', **kwargs)
            for lc1, lc2 in zip(lc1list_a, lc2list_a)]

        self.minimizer = None
        self.fit_result = None

    def log_likelihood(self, params, eval_gradient=False, delta=1e-3):
        # the likelihood is the product of the likelihood for the individual light curve pairs
        # so the log-likelihood is the sum
        if eval_gradient:
            segment_loglike = [mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_a] + \
                              [mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_b] + \
                              [mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_ab] + \
                              [mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_ref]
            # separate and sum the likelihoods and the gradients
            like = np.array([l[0] for l in segment_loglike])
            grad = np.array([l[1] for l in segment_loglike])
            if np.all(np.isfinite(like)):
                return np.sum(like), grad.sum(axis=0)
            else:
                return -np.inf, np.zeros(len(params))
        else:
            like = np.sum([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_a] + \
                              [mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_b] + \
                              [mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_ab] + \
                              [mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_ref])

            # print(like, np.isfinite(like))
            #
            # if not np.isfinite(like):
            #     print(params)
            #     print(np.sum([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_a]))
            #     print(np.sum([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_b]))
            #     print(np.sum([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_ab]))
            #     print(np.sum([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance_ref]))

            return like if np.isfinite(like) else -np.inf
