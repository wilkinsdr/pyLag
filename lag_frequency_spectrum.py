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
import matplotlib.pyplot as plt


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

    def __init__(self, bins, lc1=None, lc2=None, lc1files=None, lc2files=None, interp_gaps=False):
        self.freq = bins.bin_cent
        self.freq_error = bins.x_error()

        self.lag = np.array([])
        self.error = np.array([])

        if lc1files is not None:
            lc1 = get_lclist(lc1files, interp_gaps=interp_gaps)
        if lc2files is not None:
            lc2 = get_lclist(lc2files, interp_gaps=interp_gaps)

        if isinstance(lc1, list) and isinstance(lc2, list):
            print("Constructing lag-frequency spectrum from %d pairs of light curves" % len(lc1))
            cross_spec = StackedCrossSpectrum(lc1, lc2, bins)
            coh = Coherence(lc1, lc2, bins)
        elif isinstance(lc1, LightCurve) and isinstance(lc2, LightCurve):
            print("Computing lag-frequency spectrum from pair of light curves")
            cross_spec = CrossSpectrum(lc1, lc2).bin(bins)
            coh = Coherence(lc1, lc2, bins)

        f, self.lag = cross_spec.lag_spectrum()
        self.error = coh.lag_error()
        self.coh = coh.coh

    def _getplotdata(self):
        return (self.freq, self.freq_error), (self.lag, self.error)

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'Lag / s', 'linear'
