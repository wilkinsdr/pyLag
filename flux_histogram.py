"""
pylag.flux_histogram

Class to plot flux histograms from light curves
"""
from .lightcurve import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


class FluxHistogram(object):
    def __init__(self, lc, bins=50, log=False, plot=True):
        if isinstance(lc, list):
            lc = LightCurve().concatenate(lc)
        lc = lc.remove_gaps()

        r = lc.rate
        if log:
            r = np.log10(r)

        self.mu, self.sigma = self.fit_gaussian(r)

        if plot:
            self._fig, self._ax = plt.subplots()
            self.freq, self.bins, _ = self._ax.hist(r, bins=bins, normed=1)
            if log:
                self._ax.set_xlabel('log(Count Rate)')
            else:
                self._ax.set_xlabel('Count Rate')
            self._ax.set_ylabel('Frequency')
        else:
            self.freq, self.bins = np.histogram(r, bins=bins, normed=1)

        self.gaussian_fit = mlab.normpdf(self.bins, self.mu, self.sigma)
        if plot:
            self._ax.plot(self.bins, self.gaussian_fit, 'r--', linewidth=2)
            self._fig.show()

    @staticmethod
    def fit_gaussian(r):
        mu, sigma = norm.fit(r)
        return mu, sigma

    def write_hist(self, filename, fmt='%15.10g', delimiter=' '):
        bin_cent = 0.5 * (self.bins[1:] + self.bins[:-1])
        bin_err = self.bins[1:] - bin_cent
        data = [bin_cent, bin_err, self.bins[:-1], self.bins[1:], self.freq, self.gaussian_fit]
        np.savetxt(filename, list(zip(*data)), fmt=fmt, delimiter=delimiter)
