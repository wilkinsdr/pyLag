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
            self.delta_f = bins.delta_x()
        elif fmin > 0 and fmax > 0:
            self.freq = np.mean([fmin, fmax])
            self.freq_error = None
            self.delta_f = fmax - fmin

        # if we're passed a single pair of light curves, get the cross spectrum
        # and periodograms and count the number of sample frequencies in either
        # the bins or specified range
        if isinstance(lc, LightCurve) and isinstance(reflc, LightCurve):
            self.cross_spec = CrossSpectrum(lc, reflc)
            self.per = Periodogram(reflc)
            self.per_ref = Periodogram(reflc)
            if bins is not None:
                self.num_freq = reflc.bin_num_freq(bins)
            elif fmin > 0 and fmax > 0:
                self.num_freq = reflc.num_freq_in_range(fmin, fmax)
            self.reflcmean = reflc.mean()
            self.lcmean = lc.mean()

        # if we're passed lists of light curves, get the stacked cross spectrum
        # and periodograms and count the number of sample frequencies across all
        # the light curves
        elif isinstance(lc, list) and isinstance(reflc, list):
            self.cross_spec = StackedCrossSpectrum(lc, reflc, bins)
            self.per = StackedPeriodogram(lc, bins)
            self.per_ref = StackedPeriodogram(reflc, bins)
            if bins is not None:
                self.num_freq = np.zeros(bins.num)

                for l in reflc:
                    self.num_freq += l.bin_num_freq(bins)
            elif fmin > 0 and fmax > 0:
                self.num_freq = 0
                for l in reflc:
                    self.num_freq += l.num_freq_in_range(fmin, fmax)
            self.reflcmean = stacked_mean_count_rate(reflc)
            self.lcmean = stacked_mean_count_rate(lc)

        self.cov, self.error = self.calculate(bins, fmin, fmax, bias)

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
        elif fmin > 0 and fmax > 0:
            cross_spec = self.cross_spec.freq_average(fmin, fmax)
            per = self.per.freq_average(fmin, fmax)
            per_ref = self.per_ref.freq_average(fmin, fmax)

        if bias:
            pnoise = 2 * (self.lcmean + self.bkg1) / self.lcmean ** 2
            pnoise_ref = 2 * (self.reflcmean + self.bkg2) / self.reflcmean ** 2
            nbias = (
                    pnoise_ref * (per - pnoise) + pnoise * (per_ref - pnoise_ref) + pnoise * pnoise_ref) / self.num_freq
        else:
            pnoise = 0
            pnoise_ref = 0
            nbias = 0

        cov = self.lcmean * np.sqrt(self.delta_f * (np.abs(cross_spec) ** 2 - nbias) / (per_ref - pnoise_ref))

        # rms = (per - pnoise) * self.lcmean ** 2 * self.delta_f
        rms_noise = pnoise * self.lcmean ** 2 * self.delta_f
        rms_ref = (per_ref - pnoise_ref) * self.reflcmean ** 2 * self.delta_f
        rms_ref_noise = pnoise_ref * self.reflcmean ** 2 * self.delta_f

        err = np.sqrt((cov ** 2 * rms_ref_noise + rms_ref * rms_noise + rms_noise * rms_ref_noise) / (
            2 * self.num_freq * rms_ref))

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

        self.return_sed = True

        if lcfiles != '':
            lclist = EnergyLCList(lcfiles, interp_gaps=interp_gaps)

        self.en = np.array(lclist.en)
        self.en_error = np.array(lclist.en_error)

        if resample_errors:
            printmsg(1, "Constructing covariance spectrum from resampled light curves in %d energy bins" % len(lclist))
            self.cov, self.error = self.calculate_resample(lclist, fmin, fmax, refband, self.en, bias, n_samples)
        elif isinstance(lclist[0], LightCurve):
            printmsg(1, "Constructing covariance spectrum in %d energy bins" % len(lclist))
            self.cov, self.error = self.calculate(lclist.lclist, fmin, fmax, refband, self.en, bias=bias)
        elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
            printmsg(1, "Constructing covariance spectrum from %d light curves in each of %d energy bins" % (
                len(lclist[0]), len(lclist)))
            self.cov, self.error = self.calculate_stacked(lclist.lclist, fmin, fmax, refband, self.en, bias=bias)

        self.sed, self.sed_error = self.calculate_sed()

    def calculate(self, lclist, fmin, fmax, refband=None, energies=None, bias=True):
        """
        cov, error = pylag.CovarianceSpectrum.calculate(lclist, fmin, fmax, refband=None, energies=None)

        calculate the covariance spectrum from a list of light curves, one in
        each energy band, averaged over some frequency range.

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
        fmin     : float
                   Lower bound of frequency range
        fmax     : float
                   Upper bound of frequency range
        refband  : list of floats
                 : If specified, the reference band will be restricted to this
                   energy range [min, max]. If not specified, the full band will
                   be used for the reference
        energies : ndarray (default=None)
                 : If a specific range of energies is to be used for the reference
                   band rather than the full band, this is the list of central
                   energies of the bands represented by each light curve
        bias	 : boolean, optional (default=True)
                   If true, the bias due to Poisson noise will be subtracted from
                   the magnitude of the cross spectrum and periodograms

        Returns
        -------
        cov   : ndarray
                numpy array containing the lag of each energy band with respect
                to the reference band
        error : ndarray
                numpy array containing the error in each lag measurement
        """
        reflc = LightCurve(t=lclist[0].time)
        for energy_num, lc in enumerate(lclist):
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    continue
            reflc = reflc + lc

        cov = []
        error = []
        for energy_num, lc in enumerate(lclist):
            thisref = reflc - lc
            # if we're only using a specific reference band, we did not need to
            # subtract the current band if it's outside the range
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    thisref = reflc
            cov_obj = Covariance(lc, thisref, fmin=fmin, fmax=fmax, bias=bias)
            cov.append(cov_obj.cov)
            error.append(cov_obj.error)

        return np.array(cov), np.array(error)

    def calculate_stacked(self, lclist, fmin, fmax, refband=None, energies=None, bias=True):
        """
        cov, error = pylag.CovarianceSpectrum.calculate_stacked(lclist, fmin, fmax, refband=None, energies=None)

        calculate the covariance spectrum from a list of light curves, averaged
        over some frequency range. The covariance is calculated from the cross
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
        fmin     : float
                   Lower bound of frequency range
        fmax     : float
                   Upper bound of frequency range
        refband  : list of floats
                 : If specified, the reference band will be restricted to this
                   energy range [min, max]. If not specified, the full band will
                   be used for the reference
        energies : ndarray (default=None)
                 : If a specific range of energies is to be used for the reference
                   band rather than the full band, this is the list of central
                   energies of the bands represented by each light curve
        bias	 : boolean, optional (default=True)
                   If true, the bias due to Poisson noise will be subtracted from
                   the magnitude of the cross spectrum and periodograms

        Returns
        -------
        cov   : ndarray
                numpy array containing the covariance of each energy band with respect
                to the reference band
        error : ndarray
                numpy array containing the error in each lag measurement
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

        cov = []
        error = []
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
            # now get the covariance and error
            cov_obj = Covariance(energy_lclist, ref_lclist, fmin=fmin, fmax=fmax, bias=bias)
            cov.append(cov_obj.cov)
            error.append(cov_obj.error)

        return np.array(cov), np.array(error)

    def calculate_resample(self, lclist, fmin, fmax, refband, energies, bias, n_samples):
        cov = []
        for n in range(n_samples):
            this_lclist = lclist.resample_noise()
            if isinstance(lclist[0], LightCurve):
                this_cov, _ = self.calculate(this_lclist.lclist, fmin, fmax, refband, energies, bias)
            elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
                this_cov, _ = self.calculate_stacked(this_lclist.lclist, fmin, fmax, refband, energies, bias)
            cov.append(this_cov)

        cov = np.array(cov)

        cov_values = np.mean(cov, axis=0)
        cov_errors = np.std(cov, axis=0)

        return cov_values, cov_errors

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

    @staticmethod
    def find_lightcurves(searchstr, **kwargs):
        """
        enmin, enmax, lclist = pylag.LagEnergySpectrum.find_lightcurves(searchstr)

        Search the filesystem for light curve files and return a list of light
        curve segments for each available observation segment in each energy
        band. A 2-dimensional list of LightCurve objects from each segment
        (inner index) in each energy band (outer index) is returned,
        i.e. [[en1_obs1, en1_obs2, ...], [en2_obs1, en2_obs2, ...], ...]
        suitable for calcualation of a stacked lag-energy spectrum. Lists of
        lower and upper energies for each bin are also returned.

        If only one light curve is found for each energy band, a 1 dimensional
        list of light curves is returned.

        Light curves are sorted by lower energy bound, then alphabetically by
        filename such that if a common prefix convention is adopted identifying
        the segment, the segments listed in each energy bin will match up.

        Light curve filenames must have the substring enXXX-YYY where XXX and YYY
        are the lower and upper bounds of the energy bin in eV.
        e.g. obs1_src_en300-400.lc

        Note: Make sure that the search string returns only the light curves to
        be used in the observation and that there are the same number of segments
        in each energy band!

        Arguments
        ---------
        searchstr : string or list of strings
                    Wildcard for searching the filesystem to find the light curve
                    filesystem. If a list is passed, the results of searches with
                    all of them will be included

        Returns
        -------
        enmin  : ndarray
                 numpy array countaining the lower energy bound of each band
        enmax  : ndarray
                 numpy array containing the upper energy bound of each band
        lclist : list of list of LightCurve objects
                 The list of light curve segments in each energy band for
                 computing the lag-energy spectrum
        """
        if isinstance(searchstr, list):
            lcfiles = []
            for s in searchstr:
                lcfiles += sorted(glob.glob(s))
            lcfiles = sorted(lcfiles)
        else:
            lcfiles = sorted(glob.glob(searchstr))
        enlist = list(set([re.search('(en[0-9]+\-[0-9]+)', lc).group(0) for lc in lcfiles]))

        enmin = []
        enmax = []
        for estr in enlist:
            matches = re.search('en([0-9]+)\-([0-9]+)', estr)
            enmin.append(float(matches.group(1)))
            enmax.append(float(matches.group(2)))
        # zip the energy bins to sort them, then unpack
        entuples = sorted(zip(enmin, enmax))
        enmin, enmax = zip(*entuples)

        lclist = []
        for emin, emax in zip(enmin, enmax):
            estr = 'en%d-%d' % (emin, emax)
            energy_lightcurves = sorted([lc for lc in lcfiles if estr in lc])
            # see how many light curves match this energy - if there's only one, we
            # don't want nested lists so stacking isn't run
            if len(energy_lightcurves) > 1:
                energy_lclist = []
                for lc in energy_lightcurves:
                    energy_lclist.append(LightCurve(lc, **kwargs))
                lclist.append(energy_lclist)
            else:
                lclist.append(LightCurve(energy_lightcurves[0], **kwargs))

        return np.array(enmin) / 1000., np.array(enmax) / 1000., lclist

    def _getplotdata(self):
        return (self.en, self.en_error), ((self.sed, self.sed_error) if self.return_sed else (self.cov, self.error))

    def _getplotaxes(self):
        return 'Energy / keV', 'log', 'Covariance', 'log'

    def writeflx(self, filename):
        data = [self.en - self.en_error, self.en + self.en_error, self.cov, self.error]
        np.savetxt(filename, list(zip(*data)), fmt='%15.10g', delimiter=' ')


