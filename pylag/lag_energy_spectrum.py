"""
pylag.lag_energy_spectrum

Provides pyLag class for automating calculation of the lag-energy spectrum

Classes
-------
LagEnergySpectrum : Calculate the lag-energy spectrum from a set of light curves
                    in different energy bands

v1.0 09/03/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .cross_spectrum import *
from .coherence import *
from .binning import *
from .plotter import *

import numpy as np
import glob
import re

from .util import printmsg


class LagEnergySpectrum(object):
    """
    pylag.LagEnergySpectrum

    Class for computing the lag-energy spectrum from a set of light curves, one
    in each energy band (or a set of light curve segments in each energy band),
    relative to a reference band that is the summed time series over all energy
    bands. The lag at each energy is averaged over some	frequency range. For each
    energy band, the present energy range is subtracted from the reference band
    to avoid correlated noise.

    The resulting lag-enery spectrum is stored in the member variables.

    This class automates calculation of the lag-energy spectrum and its errors
    from the CrossSpectrum (or StackedCrossSpectrum) and the Coherence classes.

    Member Variables
    ----------------
    en       : ndarray
               numpy array containing the central energy of each band
    en_error : ndarray
               numpy array containing the error bar of each energy band (the
               central energy minus the minimum)
    lag      : ndarray
               numpy array containing the time lag of each energy band relative
               to the reference band
    error    : ndarray
               numpy array containing the error in the lag measurement in each
               band
    coh      : ndarray
               numpy array containing the coherence in each band across the
               specified frequency range

    Constructor: pylag.LagEnergySpectrum(lclist, fmin, fmax, enmin, enmax, lcfiles, interp_gaps=False, refband=None)

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
                 bias=True, calc_error=True):
        self.en = np.array([])
        self.en_error = np.array([])
        self.lag = np.array([])
        self.error = np.array([])
        self.coh = np.array([])

        self._freq_range = (fmin, fmax)

        if lcfiles != '':
            lclist = EnergyLCList(lcfiles, interp_gaps=interp_gaps)

        self.en = np.array(lclist.en)
        self.en_error = np.array(lclist.en_error)

        if isinstance(lclist[0], LightCurve):
            printmsg(1, "Constructing lag energy spectrum in %d energy bins" % len(lclist))
            self.cross_spec, self.coherence = self.calculate_crossspec(lclist.lclist, refband, self.en, bias, calc_error)
        elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
            printmsg(1, "Constructing lag energy spectrum from %d light curves in each of %d energy bins" % (
                len(lclist[0]), len(lclist)))
            self.cross_spec, self.coherence = self.calculate_crossspec_stacked(lclist.lclist, refband, self.en, bias, calc_error)

        self.lag, self.error = self.calculate_lag(self._freq_range[0], self._freq_range[1])

    def calculate_crossspec(self, lclist, refband=None, energies=None, bias=True, calc_error=True):
        """
        cross_spec, coherence = pylag.LagEnergySpectrum.ccalculate_crossspec(self, lclist, refband=None, energies=None, bias=True, calc_error=True)

        Calculate the cross spectra and coherence in preparation for calculating the lag-energy spectrum
        from a list of light curves, one in each energy band.

        The lags/cross spectra are calculated with respect to a reference light curve that is
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
        bias     : boolean, optional (default=True)
                   If true, the bias due to Poisson noise will be subtracted when
                   calculating coherence

        Returns
        -------
        lag   : ndarray
                numpy array containing the lag of each energy band with respect
                to the reference band
        error : ndarray
                numpy array containing the error in each lag measurement
        coh   : ndarray
                numpy array containing the coherence in each energy band
        """
        reflc = LightCurve(t=lclist[0].time)
        for energy_num, lc in enumerate(lclist):
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    continue
            reflc = reflc + lc

        cross_spec = []
        coherence = []
        for lc in lclist:
            thisref = reflc - lc
            # if we're only using a specific reference band, we did not need to
            # subtract the current band if it's outside the range
            if refband is not None:
                if energies[energy_num] < refband[0] or energies[energy_num] > refband[1]:
                    thisref = reflc
            cross_spec.append(CrossSpectrum(lc, thisref))
            if calc_error:
                coherence.append(Coherence(lc, reflc, fmin=None, fmax=None, bias=bias))

        return cross_spec, coherence

    def calculate_crossspec_stacked(self, lclist, refband=None, energies=None, bias=True, calc_error=True):
        """
        cross_spec, coherence = pylag.LagEnergySpectrum.calculate_crossspec_stacked(self, lclist, refband=None, energies=None, bias=True, calc_error=True)

        Calculate the lag-energy spectrum from a list of light curves, averaged
        over some frequency range. The lag-energy spectrum is calculated from
        the cross spectrum and coherence stacked over multiple light curve
        segments in each energy band.

        The lag is calculated with respect to a reference light curve that is
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
        bias     : boolean, optional (default=True)
                   If true, the bias due to Poisson noise will be subtracted when
                   calculating coherence

        Returns
        -------
        cross_spec : list of CrossSpectrum objects
                     The cross spectrum for each energy band vs. the reference
        coherence  : list of Coherence objects
                     The Coherence between each energy band and the reference
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

        cross_spec = []
        coherence = []
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
            # now get stacked cross spectrum and
            # coherence between the sets of lightcurves for this energy band and
            # for the reference
            cross_spec.append(StackedCrossSpectrum(energy_lclist, ref_lclist))
            if calc_error:
                coherence.append(Coherence(energy_lclist, ref_lclist, fmin=None, fmax=None, bias=bias))

        return cross_spec, coherence

    def calculate_lag(self, fmin, fmax, cross_spec=None, coherence=None):
        """
        lag, error = pylag.LagEnergySpectrum.calculate_lag(fmin, fmax, , cross_spec=None, coherence=None)

        Ralculate the lag-energy spectrum by averaging the cross spectra for each energy band
        over the requested frequency range.

        Requires that the cross spectrum and coherence for each band vs the reference have been pre-calculated

        Arguments
        ---------
        fmin        : float
                      Lower bound of frequency range
        fmax        : float
                      Upper bound of frequency range
        cross_spec  : list of CrossSpectrum objects, optional (default=None)
                      The CrossSpectrum object for each band vs. the reference from
                      which the lags are to be calculated. If None, the CrossSpectrum objects
                      that were pre-calculated when this objected was constructed will be
                      used
        coherence   : list of CrossSpectrum objects, optional (default=None)
                      The Coherence object for each band vs. the reference from
                      which the errors are to be calculated. If None, the Coherence objects
                      that were pre-calculated when this objected was constructed will be
                      used

        Returns
        -------
        lag   : ndarray
                numpy array containing the lag of each energy band with respect
                to the reference band
        error : ndarray
                numpy array containing the error in each lag measurement
        """
        if cross_spec is None:
            cross_spec = self.cross_spec
        if coherence is None:
            coherence = self.coherence

        lag = np.array([cs.lag_average(fmin, fmax) for cs in cross_spec])
        if len(coherence) > 0:
            error = np.array([coh.lag_error(fmin=fmin, fmax=fmax) for coh in coherence])
        else:
            error = None

        return lag, error

    def _getplotdata(self):
        return (self.en, self.en_error), (self.lag, self.error)

    def _getplotaxes(self):
        return 'Energy / keV', 'log', 'Lag / s', 'linear'

    def _get_freq_range(self,):
        return self._freq_range

    def _set_freq_range(self, value):
        self._freq_range = value
        self.lag, self.error = self.calculate_lag(self._freq_range[0], self._freq_range[1])

    freq_range = property(_get_freq_range, _set_freq_range)
