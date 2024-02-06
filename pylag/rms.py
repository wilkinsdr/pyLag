"""
pylag.rms

Provides pyLag class for computing RMS spectra

Classes
-------
Rms         : Calculate the RMS from a light curve, integrated over some frequency rage
RmsSpectrum : Compute to RMS spectrum from light curves in different energy bands

v1.0 01/03/2024 - D.R. Wilkins
"""
from .lightcurve import *
from .periodogram import *
from .binning import *
from .util import printmsg

import numpy as np


class Rms(object):
    """
    pylag.Rms

    Class to calculate the RMS variability from the periodogram of a light curve.
    Can be calculated in frequency bins or over a specified frequency range.

    Once calculated, the RMS is accessible via the rms member variable,
    either as a numpy array containing the coherence for each frequency bin or
    as a single float if the coherence is calculated over a single frequency
    range.

    Member Variables
    ----------------
    freq : ndarray or float
           numpy array storing the sample frequencies at which the coherence is
           evaluated or the mean of the frequency range if a single coherence
           value is calculated
    rms  : ndarray or float
           numpy array (complex) storing the calculated RMS in each
           frequency bin or the single coherence value if calculated over a
           single frequency range
    num_freq  : ndarray or float
                The total number of sample frequencies in each bin summed across
                all light curves

    Constructor: pylag.Covariance(lc=None, reflc=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0., bias=True)

    Constructor Arguments
    ---------------------
    lc       : LightCurve or list of LightCurve objects
               pyLag LightCurve object for the primary band (complex
               conjugated during cross spectrum calculation). If a list of LightCurve
               objects is passed, the covariance will be calculated for the stacked
               cross spectrum
    bins     : Binning, optional (default=None)
               pyLag Binning object specifying the frequency bins in which the
               coherence is to be calculated. If no binning is specified, a frequency
               range can be specfied to obtain a single coherence value over the range
    fmin     : float
               Lower bound of frequency range
    fmin     : float
               Upper bound of frequency range
    bkg      : float, optional (default=0.)
               Background count rate in the primary band for caclualtion of Poisson
               noise in bias terms
    absolute : boolean (default=True)
               Calculate the absolute RMS in units of count rate, rather than the
               fractional RMS
    """

    def __init__(self, lc=None, bins=None, fmin=None, fmax=None, bkg=0., absolute=True):
        self.bkg = bkg

        self.cov = np.array([])
        self.num_freq = np.array([])

        self.bins = bins

        if bins is not None:
            self.freq = bins.bin_cent
            self.freq_error = bins.x_error()
        elif fmin is not None and fmax is not None and (fmin > 0 and fmax > 0):
            self.freq = np.mean([fmin, fmax])
            self.freq_error = None

        # if we're passed a single pair of light curves, get the cross spectrum
        # and periodograms and count the number of sample frequencies in either
        # the bins or specified range
        if isinstance(lc, LightCurve):
            self.per = Periodogram(lc)
            self.lcmean = lc.mean()

        # if we're passed lists of light curves, get the stacked cross spectrum
        # and periodograms and count the number of sample frequencies across all
        # the light curves
        elif isinstance(lc, list):
            self.per = StackedPeriodogram(lc, bins)
            self.lcmean = stacked_mean_count_rate(lc)

        self.pnoise = 2 * (self.lcmean + self.bkg) / self.lcmean ** 2

        if bins is not None or (fmin is not None and fmax is not None):
            self.rms, self.error = self.calculate(bins, fmin, fmax, absolute)
        else:
            self.cov, self.error = np.nan, np.nan

    def calculate(self, bins=None, fmin=None, fmax=None, absolute=True):
        """
        pylag.Rms.calculate(bins=None, fmin=None, fmax=None, absolute=True)

        calculate the RMS either in each bin or over a specified frequency
        range. The result is returned either as a numpy array if calculated over
        separate bins or as a single float value when calculating for a frequency
        range.

        Arguments
        ---------
        bins     : Binning, optional (default=None)
                   pyLag Binning object specifying the bins in which coherence is to
                   be calculated
        fmin     : float
                   Lower bound of frequency range
        fmax     : float
                   Upper bound of frequency range
        absolute : boolean (default=True)
                   Calculate the absolute RMS in units of count rate, rather than the
                   fractional RMS

        Returns
        -------
        coh : ndarray or float
              The calculated coherence either as a numpy array containing the
              value for each frequency bin or a single float if the coherence is
              calculated over a single frequency range
        """
        if bins is not None:
            per = self.per.bin(self.bins).periodogram
            num_freq = self.per.num_freq_in_bins(bins)
            delta_f = bins.delta_x()
        elif fmin > 0 and fmax > 0:
            per = self.per.freq_average(fmin, fmax)
            num_freq = self.per.num_freq_in_range(fmin, fmax)
            delta_f = fmax - fmin

        rms = np.sqrt(delta_f * (per - self.pnoise))
        if absolute:
            rms *= self.lcmean

        # rms = (per - pnoise) * self.lcmean ** 2 * self.delta_f
        rmssq_noise = self.pnoise * delta_f
        if absolute:
            rmssq_noise *= self.lcmean ** 2

        err = np.sqrt((2. * rms**2 * rmssq_noise + rmssq_noise**2) / (2 * num_freq * rms**2))

        return rms, err

    def _getplotdata(self):
        return (self.freq, self.freq_error), (self.cov, self.error)

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'RMS', 'log'


class RmsSpectrum(object):
    """
    pylag.RmsSpectrum

    Class for computing the RMS spectrum from a set of light curves, one
    in each energy band (or a set of light curve segments in each energy band),
    The RMS at each energy is averaged over some	frequency range.

    The resulting RMS spectrum is stored in the member variables.

    This class automates calculation of the RMS spectrum and its errors
    from using the Covariance class for each energy.

    Member Variables
    ----------------
    en       : ndarray
               numpy array containing the central energy of each band
    en_error : ndarray
               numpy array containing the error bar of each energy band (the
               central energy minus the minimum)
    rms      : ndarray
               numpy array containing the RMS of each energy band
    error    : ndarray
               numpy array containing the error in the covariance in each
               band

    Constructor: pylag.RmsSpectrum(lclist, fmin, fmax, lcfiles, interp_gaps=False, absolute=True)

    Constructor Arguments
    ---------------------
    fmin        : float
                  Lower bound of frequency range
    fmin        : float
                  Upper bound of frequency range
    lclist      : list of LightCurve objects or list of lists of LightCurve objects
                  optional (default=None)
                  This is either a 1-dimensional list containing the pylag
                  LightCurve objects for the light curve in each of the energy bands,
                  i.e. [en1_lc, en2_lc, ...]
                  or a 2-dimensional list (i.e. list of lists) if multiple observation
                  segments are to be stacked. In this case, the outer index
                  corresponds to the energy band. For each energy band, there is a
                  list of LightCurve objects that represent the light curves in that
                  energy band from each observation segment.
                  i.e. [[en1_obs1, en1_obs2, ...], [en2_obs1, en2_obs2, ...], ...]
    enmin       : ndarray or list, optional (default=None)
                  numpy array or list containing the lower energy bound of each band
                  (each entry corresponds to one light curve, in order)
    enmax       : ndarray or list, optional (default=None)
                  numpy array or list containing the upper energy bound of each band
                  (each entry corresponds to one light curve, in order)
    lcfiles     : string, optional (default='')
                  If not empty, the filesystem will be searched using this glob to
                  automatically build the list of light curves and energies
    interp_gaps : boolean (default=False)
                  Interpolate over gaps in the light curves
    absolute    : boolean (default=True)
                  Calculate the absolute RMS in units of count rate, rather than the
                  fractional RMS
    """

    def __init__(self, fmin, fmax, lclist=None, lcfiles='', interp_gaps=False,
                 absolute=True, resample_errors=False, n_samples=100):
        self.en = np.array([])
        self.en_error = np.array([])
        self.rms = np.array([])
        self.error = np.array([])

        self._freq_range = (fmin, fmax)

        self.absolute = absolute
        self.return_sed = True

        if lcfiles != '':
            lclist = EnergyLCList(lcfiles, interp_gaps=interp_gaps)

        self.en = np.array(lclist.en)
        self.en_error = np.array(lclist.en_error)

        printmsg(1, "Constructing RMS spectrum in %d energy bins" % len(lclist))
        self.rms_obj = self.calculate_rms(lclist.lclist)

        self.rms, self.error = self.calculate_spectrum(self._freq_range[0], self._freq_range[1], absolute=self.absolute)

        if resample_errors:
            printmsg(1, "Estimating errors from %d resamples" % n_samples)
            self.error = self.resample_errors(lclist, fmin, fmax, absolute, n_samples)

        self.sed, self.sed_error = self.calculate_sed()

    def calculate_rms(self, lclist):
        """
        rms, error = pylag.RmsSpectrum.calculate(lclist, fmin, fmax, refband=None, energies=None)

        Pre-calculate the RMS objects from a list of energy channels, one in
        each energy band, to be used in the calculation of the RMS spectrum.

        Arguments
        ---------
        lclist   : list of LightCurve objects
                   1-dimensional list containing the pylag
                   LightCurve objects for the light curve in each of the energy
                   bands, i.e. [en1_lc, en2_lc, ...]

        Returns
        -------
        rms_obj  : ndarray
                   List of Rms objects for each energy channel
        """
        rms_obj = []
        for lc in lclist:
            rms_obj.append(Rms(lc, fmin=None, fmax=None))

        return rms_obj

    def calculate_spectrum(self, fmin, fmax, rms_obj=None, absolute=True):
        """
        rms, error = pylag.RmsSpectrum.calculate_spectrum(fmin, fmax, rms_obj=None, absolute=True)

        calculate the RMS spectrum by averaging the RMS for each band
        over the requested frequency range.

        Requires that the Rms objects for each band have been pre-calculated

        Arguments
        ---------
        fmin        : float
                      Lower bound of frequency range
        fmax        : float
                      Upper bound of frequency range
        rms_obj     : list of Rms objects, optional (default=None)
                      The Rms object for each band from which the spectrum is to be calculated.
                      If None, the Rms objects that were pre-calculated when this objected was
                      constructed will be used
        absolute    : boolean (default=True)
                      Calculate the absolute RMS in units of count rate, rather than the
                      fractional RMS

        Returns
        -------
        rms   : ndarray
                numpy array containing the RMS of each energy band
        error : ndarray
                numpy array containing the error in each covariance measurement
        """
        if rms_obj is None:
            rms_obj = self.rms_obj

        rms, error = zip(*[r.calculate(fmin=fmin, fmax=fmax, absolute=absolute) for r in rms_obj])
        return np.array(rms), np.array(error)

    def resample_errors(self, lclist, fmin, fmax,  absolute, n_samples, mode='std'):
        rms = []
        for n in range(n_samples):
            this_lclist = lclist.resample_noise()
            rms_obj = self.calculate_rms(this_lclist.lclist)
            this_rms, _ = self.calculate_spectrum(fmin, fmax, rms_obj, absolute)
            rms.append(this_rms)

        rms = np.array(rms)

        if mode =='std':
            rms_errors = np.nanstd(rms, axis=0)

        return rms_errors

    def calculate_sed(self):
        """
        sed, err = pylag.RmsSpectrum.calculate_sed()

        Calculatethe RMS in units of E*F_E (the spectral energy distribution),
        the	equivalent of eeufspec in XSPEC.

        Return Values
        -------------
        sed : ndarray
              The covariance spectrum in units of E*F_E
        err : ndarray
              The error on the covariance spectrum
        """
        sed = self.en ** 2 #* self.rms / (2 * self.en_error)
        err = self.en ** 2 #* self.error / (2 * self.en_error)
        return sed, err

    def _getplotdata(self):
        return (self.en, self.en_error), ((self.sed, self.sed_error) if self.return_sed else (self.rms, self.error))

    def _getplotaxes(self):
        return 'Energy / keV', 'log', 'RMS', 'log'

    def writeflx(self, filename):
        data = [self.en - self.en_error, self.en + self.en_error, self.rms, self.error]
        np.savetxt(filename, list(zip(*data)), fmt='%15.10g', delimiter=' ')

    def _get_freq_range(self,):
        return self._freq_range

    def _set_freq_range(self, value):
        self._freq_range = value
        self.rms, self.error = self.calculate_spectrum(self._freq_range[0], self._freq_range[1], absolute=self.absolute)
        self.sed, self.sed_error = self.calculate_sed()

    freq_range = property(_get_freq_range, _set_freq_range)
