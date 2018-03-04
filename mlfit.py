import numpy as np
import scipy.fftpack

class MLCovarianceMatrix(object):
    def __init__(self, lc):
        # initialise matrix of the time intervals between observations
        dt = np.zeros((len(lc), len(lc)))
        for i in range(len(lc)):
            for j in range(len(lc)):
                dt[i, j] = lc.time[j] - lc.time[i]

    def param_deriv:
        pars2 = list(self.pars)
        if pars[dpar] != 0:
            pars2[dpar] = self.pars[dpar] * (1 + step)
        else:
            pars2[dpar] = step
        return (self.calculate(pars2, *self.args) - self.calculate(self.pars, *self.args)) / (pars2[dpar] - self.pars[dpar])

class FFTMLCovarianceMatrix(MLCovarianceMatrix):
    def __init__(self, lc):
        MLCovarianceMatrix.__init__(self, lc)

        dtau = np.abs(dt[dt > 0]).min()
        ntau = 2 * int(np.abs(dt[dt > 0]).max() / dtau)
        self.tau_max = (ntau - 1) * dtau
        self.fft_freq = scipy.fftpack.fftfreq(ntau, d=dtau)

class PSDMLCovarianceMatrix(FFTMLCovarianceMatrix):
    def __init__(self, lc, pars, fit_freq):
        FFTMLCovarianceMatrix.__init__(self, lc)
        self.calculate()

    @staticmethod
    def calculate(pars, fit_freq):
        psd_interp = interp1d(fit_freq, pars)
        psd = psd_interp(np.abs(freq[1:]))
        psd = np.insert(psd, 0, 1.)
        autocorr = scipy.fftpack.ifft(psd).real
        autocorr -= autocorr.min()
        # autocorr /= autocorr[0]

        cov = np.zeros(dt.shape)

        for i in range(1, len(lc)):
            for j in range(i):
                if abs(dt[i, j]) > (tau_max / 2):
                    cov[i, j] = 0
                    # print("WARNING: time shift out of range")
                else:
                    it = int(abs(dt[i, j]) / dtau)
                    # print(dt[i,j], it, tau_max)
                    cov[i, j] = autocorr[it].real

        cov = cov + cov.T + autocorr[0].real * np.identity(dt.shape[0])

        return cov