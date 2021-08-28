"""
pylag.coherence

Provides pyLag class for computing the coherence between two light curves, for
calculating lag errors

Classes
-------
Coherence : Calculate the coherence between a pair of light curves

v1.0 09/03/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .cross_spectrum import *
from .periodogram import *
from .binning import *

import numpy as np


class Coherence(object):
    """
    pylag.Coherence

    Class to calculate the coherence between two light curves from which cross
    spectrum and lag errors can be calculated. Can be calculated in frequency
    bins or over a specified frequency range.

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

    Constructor: pylag.Coherence(lc1=None, lc2=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0., bias=True)

    Constructor Arguments
    ---------------------
    lc1  : LightCurve or list of LightCurve objects
           pyLag LightCurve object for the primary or hard band (complex
           conjugated during cross spectrum calculation). If a list of LightCurve
           objects is passed, the coherence will be calculated for the stacked
           cross spectrum
    lc2  : LightCurve or list of LightCurve objects
           pyLag LightCurve object for the reference or soft band
    bins : Binning, optional (default=None)
           pyLag Binning object specifying the frequency bins in which the
           coherence is to be calculated. If no binning is specified, a frequency
           range can be specfied to obtain a single coherence value over the range
    fmin : float
           Lower bound of frequency range
    fmin : float
           Upper bound of frequency range
    bkg1 : float, optional (default=0.)
           Background count rate in the primary band for caclualtion of Poisson
           noise in bias terms
    bkg2 : float, optional (default=0.)
           Background count rate in the reference band for caclualtion of Poisson
           noise in bias terms
    bias : boolean, optional (default=True)
           If true, the bias due to Poisson noise will be subtracted from the
           magnitude of the cross spectrum and periodograms
    """

    def __init__(self, lc1=None, lc2=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0., bias=True, **kwargs):
        self.bkg1 = bkg1
        self.bkg2 = bkg2

        self.coh = np.array([])
        self.num_freq = np.array([])

        self.bins = bins

        if bins is not None:
            self.freq = bins.bin_cent
            self.freq_error = bins.x_error()
        elif fmin > 0 and fmax > 0:
            self.freq = np.mean([fmin, fmax])
            self.freq_error = None

        # if we're passed a single pair of light curves, get the cross spectrum
        # and periodograms and count the number of sample frequencies in either
        # the bins or specified range
        if isinstance(lc1, LightCurve) and isinstance(lc2, LightCurve):
            self.cross_spec = CrossSpectrum(lc1, lc2, **kwargs)
            self.per1 = Periodogram(lc1, **kwargs)
            self.per2 = Periodogram(lc2, **kwargs)
            if bins is not None:
                self.num_freq = lc1.bin_num_freq(bins)
                # apply binning to cross spectrum and periodogram
                self.cross_spec = self.cross_spec.bin(bins)
                self.per1 = self.per1.bin(bins, calc_error=False)
                self.per2 = self.per2.bin(bins, calc_error=False)
            elif fmin > 0 and fmax > 0:
                self.num_freq = lc1.num_freq_in_range(fmin, fmax)
            self.lc1mean = lc1.mean()
            self.lc2mean = lc2.mean()

        # if we're passed lists of light curves, get the stacked cross spectrum
        # and periodograms and count the number of sample frequencies across all
        # the light curves
        elif isinstance(lc1, list) and isinstance(lc2, list):
            self.cross_spec = StackedCrossSpectrum(lc1, lc2, bins, **kwargs)
            self.per1 = StackedPeriodogram(lc1, bins, calc_error=False, **kwargs)
            self.per2 = StackedPeriodogram(lc2, bins, calc_error=False, **kwargs)
            if bins is not None:
                self.num_freq = np.zeros(bins.num)

                for lc in lc1:
                    self.num_freq += lc.bin_num_freq(bins)
            elif fmin > 0 and fmax > 0:
                self.num_freq = 0
                for lc in lc1:
                    self.num_freq += lc.num_freq_in_range(fmin, fmax)
            self.lc1mean = stacked_mean_count_rate(lc1)
            self.lc2mean = stacked_mean_count_rate(lc2)

        self.coh = self.calculate(bins, fmin, fmax, bias)

        # delete the cross spectrum and periodograms to save some memory once
        # we've finished the calculation
        del self.cross_spec, self.per1, self.per2

    def calculate(self, bins=None, fmin=None, fmax=None, bias=True):
        """
        pylag.Coherence.calculate(bins=None, fmin=None, fmax=None)

        calculate the coherence either in each bin or over a specified frequency
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
            # note the binning is already taken care of in the constructor
            # since the StackedPeriodogram needs to be binned when created
            cross_spec = self.cross_spec.crossft
            per1 = self.per1.periodogram
            per2 = self.per2.periodogram
        elif fmin > 0 and fmax > 0:
            cross_spec = self.cross_spec.freq_average(fmin, fmax)
            per1 = self.per1.freq_average(fmin, fmax)
            per2 = self.per2.freq_average(fmin, fmax)

        if bias:
            pnoise1 = 2 * (self.lc1mean + self.bkg1) / self.lc1mean ** 2
            pnoise2 = 2 * (self.lc2mean + self.bkg2) / self.lc2mean ** 2
            nbias = (pnoise2 * (per1 - pnoise1) + pnoise1 * (per2 - pnoise2) + pnoise1 * pnoise2) / self.num_freq
        else:
            nbias = 0

        coh = (np.abs(cross_spec) ** 2 - nbias) / (per1 * per2)
        return coh

    def phase_error(self):
        return np.sqrt((1 - self.coh) / (2 * self.coh * self.num_freq))

    def lag_error(self):
        return self.phase_error() / (2 * np.pi * self.freq)

    def _getplotdata(self):
        return (self.freq, self.freq_error), self.coh

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'Coherence', 'linear'
    
    
class CoherenceSpectrum(object):
    """
    pylag.CoherenceSpectrum

    Class for computing the coherence as a function of energy from a set of light curves,
    one in each energy band (or a set of light curve segments in each energy band),
    relative to a reference band that is the summed time series over all energy
    bands. The lag at each energy is averaged over some	frequency range. For each
    energy band, the present energy range is subtracted from the reference band
    to avoid correlated noise.

    The resulting coherence spectrum is stored in the member variables.

    This class automates calculation of the spectrum from the Coherence classes.

    Member Variables
    ----------------
    en       : ndarray
               numpy array containing the central energy of each band
    en_error : ndarray
               numpy array containing the error bar of each energy band (the
               central energy minus the minimum)
    coh      : ndarray
               numpy array containing the coherence in each band across the
               specified frequency range
    error    : ndarray
               for future use

    Constructor: pylag.CoherenceSpectrum(lclist, fmin, fmax, enmin, enmax, lcfiles, interp_gaps=False, refband=None)

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
    """

    def __init__(self, fmin, fmax, lclist=None, lcfiles='', interp_gaps=False, refband=None,
                 bias=True):
        self.en = np.array([])
        self.en_error = np.array([])
        self.error = np.array([])
        self.coh = np.array([])

        if lcfiles != '':
            lclist = EnergyLCList(lcfiles, interp_gaps=interp_gaps)

        self.en = np.array(lclist.en)
        self.en_error = np.array(lclist.en_error)

        if isinstance(lclist[0], LightCurve):
            print("Constructing coherence spectrum in %d energy bins" % len(lclist))
            self.coh = self.calculate(lclist.lclist, fmin, fmax, refband, self.en, bias)
        elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
            print("Constructing coherence spectrum from %d light curves in each of %d energy bins" % (
                len(lclist[0]), len(lclist)))
            self.coh = self.calculate_stacked(lclist.lclist, fmin, fmax, refband, self.en, bias)

    def calculate(self, lclist, fmin, fmax, refband=None, energies=None, bias=True):
        """
        coh = pylag.CoherenceSpectrum.calculate(lclist, fmin, fmax, refband=None, energies=None)

        calculate the coherence as a function of energy from a list of light curves,
        one in each energy band, averaged over some frequency range.

        The coherence is calculated with respect to a reference light curve that is
        computed as the sum of all energy bands, but subtracting the energy band
        of interest for each point, so to avoid correlated noise
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
        bias     : boolean, optional (default=True)
                   If true, the bias due to Poisson noise will be subtracted when
                   calculating coherence

        Returns
        -------
        coh   : ndarray
                numpy array containing the coherence of each energy band with respect
                to the reference band
        """
        reflc = LightCurve(t=lclist[0].time)
        for energy_num, lc in enumerate(lclist):
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    continue
            reflc = reflc + lc

        error = []
        coh = []
        for lc in lclist:
            thisref = reflc - lc
            # if we're only using a specific reference band, we did not need to
            # subtract the current band if it's outside the range
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    thisref = reflc
            coh.append(coherence_obj.coh)

        return np.array(coh)

    def calculate_stacked(self, lclist, fmin, fmax, refband=None, energies=None, bias=True):
        """
        coh = pylag.CoherenceSpectrum.CalculateStacked(lclist, fmin, fmax, refband=None, energies=None)

        calculate the coherence as a function of energy from a list of light curves,
        averaged over some frequency range. The coherence spectrum is calculated from
        the cross spectrum and stacked over multiple light curve
        segments in each energy band.
        The coherence is calculated with respect to a reference light curve that is
        computed as the sum of all energy bands, but subtracting the energy band
        of interest for each point, so to avoid correlated noise
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
        bias     : boolean, optional (default=True)
                   If true, the bias due to Poisson noise will be subtracted when
                   calculating coherence

        Returns
        -------
        coh   : ndarray
                numpy array containing the coherence of each energy band with respect
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

        error = []
        coh = []
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
            # now get the lag and error from the stacked cross spectrum and
            # coherence between the sets of lightcurves for this energy band and
            # for the reference
            coherence_obj = Coherence(energy_lclist, ref_lclist, fmin=fmin, fmax=fmax, bias=bias)
            coh.append(coherence_obj.coh)

        return np.array(coh)

    def _getplotdata(self):
        return (self.en, self.en_error), self.coh

    def _getplotaxes(self):
        return 'Energy / keV', 'log', 'Coherence', 'linear'


class ResampledCoherence(Coherence):
    def __init__(self, lc1=None, lc2=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0., bias=True, n_samples=10):
        if bins is not None:
            self.freq = bins.bin_cent
            self.freq_error = bins.x_error()
        elif fmin > 0 and fmax > 0:
            self.freq = np.mean([fmin, fmax])
            self.freq_error = None

        cohs = []

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

            coh_obj = Coherence(lc1_resample, lc2_resample, bins, fmin, fmax, bkg1, bkg2, bias)
            cohs.append(coh_obj.coh)

        cohs = np.array(cohs)
        self.coh = np.mean(cohs, axis=0)
        self.error = np.std(cohs, axis=0)

    def _getplotdata(self):
        return (self.freq, self.freq_error), (self.coh, self.error)
