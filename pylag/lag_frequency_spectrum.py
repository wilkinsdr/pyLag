"""
pylag.lag_frequency_spectrum

Provides pyLag class for automating calculation of the lag-frequency spectrum

Classes
-------
LagFrequencySpectrum : Calculate the lag-frequency spectrum from a set of light
                       curves in two energy bands

v1.0 09/03/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .cross_spectrum import *
from .coherence import *
from .binning import *
from .plotter import *

import numpy as np

from .util import printmsg


class LagFrequencySpectrum(object):
    """
    pylag.LagFrequencySpectrum

    Class for computing the lag-frequency spectrum from pairs of light curves
    with some frequency binning.

    The lag-frequency spectrum can either be calculated between a single pair of
    light curves or between multiple pairs of light curves (passing a list of
    LightCurve objects for both lc1 and lc2) to produce a stacked lag-frequency
    spectrum. If stacking in this way, the cross spectrum is calculated for each
    pair of light curves in turn, then the data points are sorted into bins. The
    final cross spectrum in each bin is the average over all of the individual
    frequency points from all of the light curves that fall into that bin.

    In order to adopt the convention that a positive lag indicates variability
    in a harder X-ray band lagging that in a softer band, the hard band light
    curve or list of light curves should be passed as lc1 and the soft band as
    lc2.

    The resulting lag-frequency spectrum is stored in the member variables.

    This class automates calculation of the lag-frequency spectrum and its errors
    from the CrossSpectrum (or StackedCrossSpectrum) and the Coherence classes.

    Member Variables
    ----------------
    freq       : ndarray
                 numpy array containing the central frequency of each bin
    freq_error : ndarray
                 numpy array containing the error bar of each frequency bin (the
                 central frequency minus the minimum)
    lag        : ndarray
                 numpy array containing the time lag of lc1 relative to lc2 in
                 each frequency bin
    error      : ndarray
                 numpy array containing the error in the lag measurement in each
                 bin

    Constructor: pylag.LagEnergySpectrum(lclist, fmin, fmax, enmin, enmax)

    Constructor Arguments
    ---------------------
    bins : Binning, optional (default=None)
           pyLag Binning object specifying the frequency bins in which the
           lag spectrum will be evaluated
    lc1  : LightCurve or list of LightCurve objects
           pyLag LightCurve object for the primary or hard band (complex
           conjugated during cross spectrum calculation). If a list of LightCurve
           objects is passed, the stacked lag-frequency spectrum will be calculated
    lc2  : LightCurve or list of LightCurve objects
           pyLag LightCurve object for the reference or soft band
    """

    def __init__(self, bins=10, lc1=None, lc2=None, lc1files=None, lc2files=None, interp_gaps=False, calc_error=True, resample_errors=False, n_samples=10, calculate_args={}, **kwargs):
        self.lag = np.array([])
        self.error = np.array([])

        if lc1files is not None:
            lc1 = get_lclist(lc1files, interp_gaps=interp_gaps, **kwargs)
        if lc2files is not None:
            lc2 = get_lclist(lc2files, interp_gaps=interp_gaps, **kwargs)

        self.cross_spec, self.coherence = self.calculate_crossspec(lc1, lc2, calc_error, **calculate_args)

        self.fmin = self.cross_spec.fmin()
        self.fmax = self.cross_spec.fmax()

        if isinstance(bins, Binning):
            self._bins = bins
            self._Nf = len(self._bins)
        elif isinstance(bins, int):
            self._Nf = bins
            self._bins = LogBinning(self.fmin, self.fmax, bins)
        self.freq = self._bins.bin_cent
        self.freq_error = self._bins.x_error()

        self.lag, self.error = self.calculate_lag(self._bins)

        if resample_errors:
            self.error = self.resample_errors(lc1, lc2, self._bins, n_samples)

    @staticmethod
    def calculate_crossspec(lc1, lc2, calc_error=True, **kwargs):
        if isinstance(lc1, list) and isinstance(lc2, list):
            printmsg(1, "Constructing lag-frequency spectrum from %d pairs of light curves" % len(lc1))
            cross_spec = StackedCrossSpectrum(lc1, lc2, bins=None, **kwargs)
        elif isinstance(lc1, LightCurve) and isinstance(lc2, LightCurve):
            printmsg(1, "Computing lag-frequency spectrum from pair of light curves")
            cross_spec = CrossSpectrum(lc1, lc2, **kwargs)

        if calc_error:
            coherence = Coherence(lc1, lc2, **kwargs)
        else:
            coherence = None

        return cross_spec, coherence

    def calculate_lag(self, bins, cross_spec=None, coherence=None):
        if cross_spec is None:
            cross_spec = self.cross_spec
        if coherence is None:
            coherence = self.coherence

        _, lag = cross_spec.bin(bins).lag_spectrum()
        if coherence is not None:
            error = coherence.lag_error(bins=bins)
        else:
            error = None

        return lag, error

    def resample_errors(self, lc1, lc2, bins, n_samples=10, **kwargs):
        lags = []
        for n in range(n_samples):
            if isinstance(lc1, list) and isinstance(lc2, list):
                lc1_resample = []
                lc2_resample = []
                for lc in lc1:
                    lc1_resample.append(lc.resample_noise())
                for lc in lc2:
                    lc2_resample.append(lc.resample_noise())
            elif isinstance(lc1, LightCurve) and isinstance(lc2, LightCurve):
                lc1_resample = lc1.resample_noise()
                lc2_resample = lc2.resample_noise()

            cs, _ = LagFrequencySpectrum.calculate_crossspec(lc1_resample, lc2_resample, calc_error=False, **kwargs)
            l, _ = self.calculate_lag(bins, cross_spec=cs, coherence=None)
            lags.append(l)

        lags = np.array(lags)
        error = np.std(lags, axis=0)

        return error


    def _getplotdata(self):
        return (self.freq, self.freq_error), (self.lag, self.error)

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'Lag / s', 'linear'

    def _get_bins(self,):
        return self._bins

    def _set_bins(self, value):
        self._bins = value
        self.freq = self._bins.bin_cent
        self.freq_error = self._bins.x_error()
        self.lag, self.error = self.calculate_lag(self._bins)

    bins = property(_get_bins, _set_bins)

    def _get_Nf(self,):
        return self._Nf

    def _set_Nf(self, value):
        self._Nf = value
        self._bins = LogBinning(self.fmin, self.fmax, self._Nf)
        self.freq = self._bins.bin_cent
        self.freq_error = self._bins.x_error()
        self.lag, self.error = self.calculate_lag(self._bins)

    Nf = property(_get_Nf, _set_Nf)


