"""
pylag.wavelet

Classes for wavelet spectral timing analysis, based on pyCWT

v1.0 - 29/04/2025 D.R. Wilkins
"""
from .pycwt import *
from .pycwt.wavelet import _check_parameter_wavelet
from .pycwt.helpers import (ar1, ar1_spectrum, fft, fft_kwargs, find, get_cache_dir,
                      rednoise)
from .pycwt.mothers import Morlet, Paul, DOG, MexicanHat
import numpy as np
from .plotter import DataSeries

import matplotlib.pyplot as plt


class WaveletCoherence(object):
    def __init__(self, lc1, lc2, dj=1 / 12, s0=-1, J=-1, wavelet='morlet', normalize=True, **kwargs):
        # Makes sure input signals are numpy arrays.
        self.y1 = np.array(lc1.rate)
        self.y2 = np.array(lc2.rate)
        # Calculates the standard deviation of both input signals.
        self.std1 = self.y1.std()
        self.std2 = self.y2.std()
        # Normalizes both signals, if appropriate.
        if normalize:
            self.y1 = (self.y1 - self.y1.mean()) / self.std1
            self.y2 = (self.y2 - self.y2.mean()) / self.std2

        self.t = lc1.time
        self.dt = np.min(np.diff(self.t))

        self.wavelet = _check_parameter_wavelet(wavelet)

        self.dj = dj
        if s0 == -1:
            # Number of scales
            self.s0 = 2 * self.dt / self.wavelet.flambda()
        else:
            self.s0 = s0
        if J == -1:
            # Number of scales
            self.J = int(np.round(np.log2(self.y1.size * self.dt / self.s0) / self.dj))

        self.coh, self.plag, self.coi, self.freq, self.W1, self.W2, self.W12, self.S1, self.S2, self.S12 \
            = self.calculate(self.y1, self.y2, self.dt, self.dj, self.s0, self.J, self.wavelet, **kwargs)

    @staticmethod
    def calculate(y1, y2, dt, dj=1 / 12, s0=-1, J=-1, wavelet='morlet', **kwargs):
        """Wavelet coherence transform (WCT).

        The WCT finds regions in time frequency space where the two time
        series co-vary, but do not necessarily have high power.

        Parameters
        ----------
        y1, y2 : numpy.ndarray, list
            Input signals.
        dt : float
            Sample spacing.
        dj : float, optional
            Spacing between discrete scales. Default value is 1/12.
            Smaller values will result in better scale resolution, but
            slower calculation and plot.
        s0 : float, optional
            Smallest scale of the wavelet. Default value is 2*dt.
        J : float, optional
            Number of scales less one. Scales range from s0 up to
            s0 * 2**(J * dj), which gives a total of (J + 1) scales.
            Default is J = (log2(N*dt/so))/dj.
        sig : bool
            set to compute signficance, default is True
        significance_level (float, optional) :
            Significance level to use. Default is 0.95.
        normalize (boolean, optional) :
            If set to true, normalizes CWT by the standard deviation of
            the signals.

        Returns
        -------
        WCT : magnitude of coherence
        aWCT : phase angle of coherence
        coi (array like):
            Cone of influence, which is a vector of N points containing
            the maximum Fourier period of useful information at that
            particular time. Periods greater than those are subject to
            edge effects.
        freq (array like):
            Vector of Fourier equivalent frequencies (in 1 / time units)    coi :
        sig :  Significance levels as a function of scale
           if sig=True when called, otherwise zero.

        See also
        --------
        cwt, xwt

        """
        # Calculates the CWT of the time-series making sure the same parameters
        # are used in both calculations.
        _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
        W1, sj, freq, coi, _, _ = cwt(y1, dt, **_kwargs)
        W2, sj, freq, coi, _, _ = cwt(y2, dt, **_kwargs)

        scales1 = np.ones([1, y1.size]) * sj[:, None]
        scales2 = np.ones([1, y2.size]) * sj[:, None]

        # Smooth the wavelet spectra before truncating.
        S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
        S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

        # Now the wavelet transform coherence
        W12 = W1 * W2.conj()
        scales = np.ones([1, y1.size]) * sj[:, None]
        S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
        coh = np.abs(S12) ** 2 / (S1 * S2)
        plag = np.angle(S12)

        return coh, plag, coi, freq, W1, W2, W12, S1, S2, S12

    def get_timeavg_coherence(self, coi=True):
        if coi:
            ti, fi = np.meshgrid(self.t, self.freq)
            coi_mask = np.ones_like(fi)
            for i in range(fi.shape[1]):
                coi_mask[self.freq < 1. / self.coi[i], i] = np.nan

            # don't include scales outside the COI in the mean
            W12 = self.W12 * coi_mask
            W1 = self.W1 * coi_mask
            W2 = self.W2 * coi_mask

        else:
            W12 = self.W12
            W1 = self.W1
            W2 = self.W2

        avgcoh = np.abs(np.nanmean(W12, axis=1)) ** 2 / (
                    np.nanmean(np.abs(W1) ** 2, axis=1) * np.nanmean(np.abs(W2) ** 2, axis=1))

        return avgcoh

    def timeavg_coherence(self, **kwargs):
        avgcoh = self.get_timeavg_coherence(**kwargs)
        return DataSeries(self.freq, avgcoh, xlabel='Frequency (Hz)', ylabel='Coherence', xscale='log', yscale='linear')

    def plot(self, levels=5, cmap='gray_r', figsize=(12, 6), dpi=72):
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # First sub-plot, the original time series
        ax = plt.axes([0.1, 0.65, 0.65, 0.3])
        # ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
        ax.plot(self.t, self.y1, 'C0', linewidth=1.5)
        ax.plot(self.t, self.y2, 'C1', linewidth=1.5)
        ax.set_ylabel(r'{} [{}]'.format('Count Rate', 'ct/s'))

        # Second sub-plot, the wavelet coherence spectrum
        bx = plt.axes([0.1, 0.1, 0.65, 0.45], sharex=ax)
        if isinstance(levels, int):
            levels = np.linspace(0, 1, levels)
        bx.contourf(self.t, np.log2(self.freq), self.coh, levels, extend='both', cmap=cmap)
        extent = [self.t.min(), self.t.max(), min(self.freq), max(self.freq)]
        # shade the cone of influence
        bx.fill(np.concatenate([self.t, self.t[-1:] + self.dt, self.t[-1:] + self.dt,
                                self.t[:1] - self.dt, self.t[:1] - self.dt]),
                np.concatenate([np.log2(1. / self.coi), [1e-9], np.log2(self.freq[-1:]),
                                np.log2(self.freq[-1:]), [1e-9]]),
                'w', alpha=0.5, hatch=None)
        bx.set_xlabel('Time (s)')
        bx.set_ylabel('Frequency (Hz)')
        #
        Yticks = 2 ** np.arange(np.ceil(np.log2(self.freq.min())),
                                np.ceil(np.log2(self.freq.max())))
        bx.set_yticks(np.log2(Yticks))
        bx.set_yticklabels(['%0.1E' % tick for tick in Yticks])

        # Third sub-plot, the average coherence over all time bins
        avgcoh = self.get_timeavg_coherence(coi=True)

        cx = plt.axes([0.77, 0.1, 0.2, 0.45], sharey=bx)
        cx.plot(avgcoh, np.log2(self.freq), 'k-', linewidth=1.5)
        cx.set_xlabel(r'Coherence')
        cx.set_xlim([0, 1])
        cx.set_ylim(np.log2([self.freq.min(), self.freq.max()]))
        cx.set_yticks(np.log2(Yticks))
        cx.set_yticklabels(['%0.1E' % tick for tick in Yticks])
        plt.setp(cx.get_yticklabels(), visible=False)

        plt.show()

