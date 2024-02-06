"""
pylag.covariance

Provides pyLag class for computing covariance spectra

Classes
-------
Covariance         : Calculate the coherence between a light curve and a reference
CovarianceSpectrum : Compute to covariance spectrum from light curves in different
                     energy bands

v1.0 09/03/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .cross_spectrum import *
from .periodogram import *
from .binning import *
from .util import printmsg

import numpy as np
import re
import glob


class Covariance(object):
    """
    pylag.Covariance

    Class to calculate the covariance between two light curves. Can be calculated
    in frequency bins or over a specified frequency range.

    Once calculated, the coherence is accessible via the coh member variable,
    either as a numpy array containing the coherence for each frequency bin or
    as a single float if the coherence is calculated over a single frequency
    range.

    Member Variables
    ----------------
    freq : ndarray or float
           numpy array storing the sample frequencies at which the coherence is
           evaluated or the mean of the frequency range if a single coherence
           value is calculated
    coh  : ndarray or float
           numpy array (complex) storing the calculated coherence in each
           frequency bin or the single coherence value if calculated over a
           single frequency range
    num_freq  : ndarray or float
                The total number of sample frequencies in each bin summed across
                all light curves

    Constructor: pylag.Covariance(lc=None, reflc=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0., bias=True)

    Constructor Arguments
    ---------------------
    lc   :   LightCurve or list of LightCurve objects
             pyLag LightCurve object for the primary band (complex
             conjugated during cross spectrum calculation). If a list of LightCurve
             objects is passed, the covariance will be calculated for the stacked
             cross spectrum
    reflc  : LightCurve or list of LightCurve objects
             pyLag LightCurve object for the reference band
    bins   : Binning, optional (default=None)
             pyLag Binning object specifying the frequency bins in which the
             coherence is to be calculated. If no binning is specified, a frequency
             range can be specfied to obtain a single coherence value over the range
    fmin   : float
             Lower bound of frequency range
    fmin   : float
             Upper bound of frequency range
    bkg1   : float, optional (default=0.)
             Background count rate in the primary band for caclualtion of Poisson
             noise in bias terms
    bkg2   : float, optional (default=0.)
             Background count rate in the reference band for caclualtion of Poisson
             noise in bias terms
    bias   : boolean, optional (default=True)
             If true, the bias due to Poisson noise will be subtracted from the
             magnitude of the cross spectrum and periodograms
    """

    def __init__(self, lc=None, reflc=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0., bias=True):
        self.bkg1 = bkg1
        self.bkg2 = bkg2

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
        if isinstance(lc, LightCurve) and isinstance(reflc, LightCurve):
            self.cross_spec = CrossSpectrum(lc, reflc)
            self.per = Periodogram(reflc)
            self.per_ref = Periodogram(reflc)
            self.reflcmean = reflc.mean()
            self.lcmean = lc.mean()

        # if we're passed lists of light curves, get the stacked cross spectrum
        # and periodograms and count the number of sample frequencies across all
        # the light curves
        elif isinstance(lc, list) and isinstance(reflc, list):
            self.cross_spec = StackedCrossSpectrum(lc, reflc, bins)
            self.per = StackedPeriodogram(lc, bins)
            self.per_ref = StackedPeriodogram(reflc, bins)
            self.reflcmean = stacked_mean_count_rate(reflc)
            self.lcmean = stacked_mean_count_rate(lc)

        self.pnoise = 2 * (self.lcmean + self.bkg1) / self.lcmean ** 2
        self.pnoise_ref = 2 * (self.reflcmean + self.bkg2) / self.reflcmean ** 2

        if bins is not None or (fmin is not None and fmax is not None):
            self.cov, self.error = self.calculate(bins, fmin, fmax, bias)
        else:
            self.cov, self.error = np.nan, np.nan

    def calculate(self, bins=None, fmin=None, fmax=None, bias=True):
        """
        pylag.Coherence.calculate(bins=None, fmin=None, fmax=None)

        calculate the covariance either in each bin or over a specified frequency
        range. The result is returned either as a numpy array if calculated over
        separate bins or as a single float value when calculating for a frequency
        range.

        Arguments
        ---------
        bins : Binning, optional (default=None)
               pyLag Binning object specifying the bins in which coherence is to
               be calculated
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range
        bias : boolean, optional (default=True)
               If true, the bias due to Poisson noise will be subtracted from the
               magnitude of the cross spectrum and periodograms

        Returns
        -------
        coh : ndarray or float
              The calculated coherence either as a numpy array containing the
              value for each frequency bin or a single float if the coherence is
              calculated over a single frequency range
        """
        if bins is not None:
            cross_spec = self.cross_spec.bin(self.bins).crossft
            per = self.per.bin(self.bins).periodogram
            per_ref = self.per_ref.bin(self.bins).periodogram
            num_freq = self.per_ref.num_freq_in_bins(bins)
            delta_f = bins.delta_x()
        elif fmin > 0 and fmax > 0:
            cross_spec = self.cross_spec.freq_average(fmin, fmax)
            per = self.per.freq_average(fmin, fmax)
            per_ref = self.per_ref.freq_average(fmin, fmax)
            num_freq = self.per_ref.num_freq_in_range(fmin, fmax)
            delta_f = fmax - fmin

        if bias:
            nbias = (
                    self.pnoise_ref * (per - self.pnoise) + self.pnoise * (per_ref - self.pnoise_ref) + self.pnoise * self.pnoise_ref) / num_freq
        else:
            nbias = 0

        cov = self.lcmean * np.sqrt(delta_f * (np.abs(cross_spec) ** 2 - nbias) / (per_ref - self.pnoise_ref))

        # rms = (per - pnoise) * self.lcmean ** 2 * delta_f
        rms_noise = self.pnoise * self.lcmean ** 2 * delta_f
        rms_ref = (per_ref - self.pnoise_ref) * self.reflcmean ** 2 * delta_f
        rms_ref_noise = self.pnoise_ref * self.reflcmean ** 2 * delta_f

        err = np.sqrt((cov ** 2 * rms_ref_noise + rms_ref * rms_noise + rms_noise * rms_ref_noise) / (
            2 * num_freq * rms_ref))

        return cov, err

    def _getplotdata(self):
        return (self.freq, self.freq_error), (self.cov, self.error)

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'Covariance', 'log'


class CovarianceSpectrum(object):
    """
    pylag.CovarianceSpectrum

    Class for computing the covariance spectrum from a set of light curves, one
    in each energy band (or a set of light curve segments in each energy band),
    relative to a reference band that is the summed time series over all energy
    bands. The covariance at each energy is averaged over some	frequency range.
    For each energy band, the present energy range is subtracted from the reference
    band to avoid correlated noise.

    The resulting covariance spectrum is stored in the member variables.

    This class automates calculation of the covariance spectrum and its errors
    from using the Covariance class for each energy.

    Member Variables
    ----------------
    en       : ndarray
               numpy array containing the central energy of each band
    en_error : ndarray
               numpy array containing the error bar of each energy band (the
               central energy minus the minimum)
    cov      : ndarray
               numpy array containing the covariance of each energy band relative
               to the reference band
    error    : ndarray
               numpy array containing the error in the covariance in each
               band

    Constructor: pylag.Covariance(lclist, fmin, fmax, enmin, enmax, lcfiles, interp_gaps=False, refband=None)

    Constructor Arguments
    ---------------------
    fmin        : float
                  Lower bound of frequency range
    fmin          : float
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
    refband     : list of floats
                  If specified, the reference band will be restricted to this
                  energy range [min, max]. If not specified, the full band will
                  be used for the reference
    bias   		: boolean, optional (default=True)
                  If true, the bias due to Poisson noise will be subtracted from
                  the magnitude of the cross spectrum and periodograms
    """

    def __init__(self, fmin, fmax, lclist=None, lcfiles='', interp_gaps=False, refband=None,
                 bias=True, resample_errors=False, n_samples=100):
        self.en = np.array([])
        self.en_error = np.array([])
        self.cov = np.array([])
        self.error = np.array([])

        self._freq_range = (fmin, fmax)

        self.return_sed = True
        self.bias = bias

        if lcfiles != '':
            lclist = EnergyLCList(lcfiles, interp_gaps=interp_gaps)

        self.en = np.array(lclist.en)
        self.en_error = np.array(lclist.en_error)

        if isinstance(lclist[0], LightCurve):
            printmsg(1, "Constructing covariance spectrum in %d energy bins" % len(lclist))
            self.covariances = self.calculate_covariances(lclist.lclist, refband, self.en)
        elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
            printmsg(1, "Constructing covariance spectrum from %d light curves in each of %d energy bins" % (
                len(lclist[0]), len(lclist)))
            self.covariances = self.calculate_covariances_stacked(lclist.lclist, refband, self.en)

        self.cov, self.error = self.calculate_spectrum(self._freq_range[0], self._freq_range[1], bias=self.bias)

        if resample_errors:
            printmsg(1, "Estimating errors from %d resamples" % n_samples)
            self.error = self.resample_errors(lclist, self._freq_range[0], self._freq_range[1], refband, self.en, bias, n_samples)

        self.sed, self.sed_error = self.calculate_sed()

    @staticmethod
    def calculate_covariances(lclist, refband=None, energies=None):
        """
        cov, error = pylag.CovarianceSpectrum.calculate_covariances(lclist, fmin, fmax, refband=None, energies=None)

        calculate the covariances for each light curve vs the reference band to
        preparate for the calcation of the covariance spectrum

        The covariance is calculated with respect to a reference light curve that is
        computed as the sum of all energy bands, but subtracting the energy band
        of interest for each lag/energy point, so to avoid correlated noise
        between the subject and reference light curves.

        Arguments
        ---------
        lclist   : list of LightCurve objects
                   1-dimensional list containing the pylag
                   LightCurve objects for the light curve in each of the energy
                   bands, i.e. [en1_lc, en2_lc, ...]
        refband  : list of floats
                 : If specified, the reference band will be restricted to this
                   energy range [min, max]. If not specified, the full band will
                   be used for the reference
        energies : ndarray (default=None)
                 : If a specific range of energies is to be used for the reference
                   band rather than the full band, this is the list of central
                   energies of the bands represented by each light curve

        Returns
        -------
        covariances : list of Covariance objects
                      Covariance foe each energy band
        """
        reflc = LightCurve(t=lclist[0].time)
        for energy_num, lc in enumerate(lclist):
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    continue
            reflc = reflc + lc

        covariances = []
        for energy_num, lc in enumerate(lclist):
            thisref = reflc - lc
            # if we're only using a specific reference band, we did not need to
            # subtract the current band if it's outside the range
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    thisref = reflc
            covariances.append(Covariance(lc, thisref, fmin=None, fmax=None))

        return covariances

    @staticmethod
    def calculate_covariances_stacked(lclist, refband=None, energies=None):
        """
        cov, error = pylag.CovarianceSpectrum.calculate_stacked(lclist, fmin, fmax, refband=None, energies=None)

        calculate the covariances from a list of light curves vs the reference
        band. The covariance is calculated from the cross
        spectrum and coherence stacked over multiple light curve segments in
        each energy band.

        The covariance is calculated with respect to a reference light curve that is
        computed as the sum of all energy bands, but subtracting the energy band
        of interest for each lag/energy point, so to avoid correlated noise
        between the subject and reference light curves.

        Arguments
        ---------
        lclist   : list of lists of LightCurve objects
                   This is a 2-dimensional list (i.e. list of lists). The outer index
                   corresponds to the energy band. For each energy band, there is a
                   list of LightCurve objects that represent the light curves in that
                   energy band from each observation segment.
                   i.e. [[en1_obs1, en1_obs2, ...], [en2_obs1, en2_obs2, ...], ...]
        refband  : list of floats
                 : If specified, the reference band will be restricted to this
                   energy range [min, max]. If not specified, the full band will
                   be used for the reference
        energies : ndarray (default=None)
                 : If a specific range of energies is to be used for the reference
                   band rather than the full band, this is the list of central
                   energies of the bands represented by each light curve

        Returns
        -------
        covariances : list of Covariance objects
                      Covariance foe each energy band
        """
        reflc = []
        # initialise a reference light curve for each of the observations/light
        # curve segments
        for lc in lclist[0]:
            reflc.append(LightCurve(t=lc.time))
        # sum all of the energies (outer index) together to produce a reference
        # light curve for each segment (inner index)
        for energy_num, energy_lcs in enumerate(lclist):
            # if a reference band is specifed, skip any energies that do not fall
            # in that range
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    continue
            for segment_num, segment_lc in enumerate(energy_lcs):
                reflc[segment_num] = reflc[segment_num] + segment_lc

        covariances = []
        for energy_num, energy_lclist in enumerate(lclist):
            # subtract this energy band from the reference light curve for each
            # light curve segment to be stacked (subtracting the current band
            # means we don't have correlated noise between the subject and
            # reference bands)
            ref_lclist = []
            for segment_num, segment_lc in enumerate(energy_lclist):
                # if a reference band is specifed and this energy falls outside that
                # band, no need to subtract the current band
                if refband is not None:
                    if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                        ref_lclist.append(reflc[segment_num])
                        continue
                ref_lclist.append(reflc[segment_num] - segment_lc)
            # now get the covariance
            covariances.append(Covariance(energy_lclist, ref_lclist, fmin=None, fmax=None))

        return covariances

    def calculate_spectrum(self, fmin, fmax, covariances=None, bias=True):
        """
        cov, error = pylag.CovarianceSpectrum.calculate_spectrum(fmin, fmax, covariances=None, bias=True)

        calculate the covariance spectrum by averaging the covariances for each band
        over the requested frequency range.

        Requires that the covariances for each band vs the reference have been pre-calculated

        Arguments
        ---------
        fmin        : float
                      Lower bound of frequency range
        fmax        : float
                      Upper bound of frequency range
        covariances : list of Covariance objects, optional (default=None)
                      The Covariance object for each band vs. the reference from
                      which the spectrum is to be calculated. If None, the Covariances
                      that were pre-calculated when this objected was constructed will be
                      used
        bias	    : boolean, optional (default=True)
                      If true, the bias due to Poisson noise will be subtracted from
                      the magnitude of the cross spectrum and periodograms

        Returns
        -------
        cov   : ndarray
                numpy array containing the covariance of each energy band with respect
                to the reference band
        error : ndarray
                numpy array containing the error in each covariance measurement
        """
        if covariances is None:
            covariances = self.covariances

        cov, error = zip(*[c.calculate(fmin=fmin, fmax=fmax, bias=bias) for c in covariances])
        return np.array(cov), np.array(error)

    def resample_errors(self, lclist, fmin, fmax, refband, energies, bias, n_samples, mode='std'):
        cov = []
        for n in range(n_samples):
            this_lclist = lclist.resample_noise()
            if isinstance(lclist[0], LightCurve):
                covariances = self.calculate_covariances(this_lclist.lclist, refband, energies)
            elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
                covariances = self.calculate_covariances_stacked(this_lclist.lclist, refband, energies)
            this_cov, _ = self.calculate_spectrum(fmin, fmax, covariances, bias)
            cov.append(this_cov)

        cov = np.array(cov)

        if mode =='std':
            cov_errors = np.nanstd(cov, axis=0)

        return cov_errors

    def calculate_sed(self):
        """
        sed, err = pylag.CovarianceSpectrum.calculate_sed()

        Calculatethe covariance in units of E*F_E (the spectral energy distribution),
        the	equivalent of eeufspec in XSPEC.

        Return Values
        -------------
        sed : ndarray
              The covariance spectrum in units of E*F_E
        err : ndarray
              The error on the covariance spectrum
        """
        sed = self.en ** 2 * self.cov / (2 * self.en_error)
        err = self.en ** 2 * self.error / (2 * self.en_error)
        return sed, err

    def _getplotdata(self):
        return (self.en, self.en_error), ((self.sed, self.sed_error) if self.return_sed else (self.cov, self.error))

    def _getplotaxes(self):
        return 'Energy / keV', 'log', 'Covariance', 'log'

    def writeflx(self, filename):
        data = [self.en - self.en_error, self.en + self.en_error, self.cov, self.error]
        np.savetxt(filename, list(zip(*data)), fmt='%15.10g', delimiter=' ')

    def _get_freq_range(self,):
        return self._freq_range

    def _set_freq_range(self, value):
        self._freq_range = value
        self.cov, self.error = self.calculate_spectrum(self._freq_range[0], self._freq_range[1], bias=self.bias)
        self.sed, self.sed_error = self.calculate_sed()

    freq_range = property(_get_freq_range, _set_freq_range)


