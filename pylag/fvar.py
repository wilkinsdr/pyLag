"""
pylag.fvar

Provides pyLag class for calculating fractional variability spectra

Classes
-------
Fvar : Calculate the fractional variability from a set of light curves in
       different energy bands

v1.0 04/05/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .plotter import *

import numpy as np
import glob
import re


def fvar(lclist, tbin):
    """
    fvar, error = pylag.fvar(lclist, bin)

    Returns the fractional variability (the sqrt of the excess variance) in a
    light curve or list of light curves. If a list of light curves is passed,
    the time bins from each light curve are concatenated to calculate the fvar
    across the whole set of light curves.

    Arguments
    ---------
    lclist : LightCurve or list of LightCurves
             The LightCurve object (or list of multiple LightCurve objects) from
             which the fvar is to be calculated
    tbin   : float
             Time bin size to be used for the excess variance

    Returns
    -------
    fvar  : float
            The fractional variability in the light curve
    error : float
            The error in the fvar measurement
    """
    if isinstance(lclist, LightCurve):
        lclist = [lclist]

    bin_mean = []
    bin_stderr = []
    bins = 0
    for lc in lclist:
        for tmin in np.arange(lc.time.min(), lc.time.max(), tbin):
            tmax = tmin + tbin
            binlc = lc.time_segment(tmin, tmax)
            bin_count = len(binlc)
            if bin_count == 0:
                continue
            bins += 1
            bin_mean.append(binlc.mean())
            bin_stderr.append((1. / (bin_count * (bin_count - 1))) * sum((np.array(binlc.rate) - binlc.mean()) ** 2.))

    bin_mean = np.array(bin_mean)
    bin_stderr = np.array(bin_stderr)

    binned_mean = np.mean(bin_mean)
    binned_variance = (1. / (bins - 1)) * sum((bin_mean - binned_mean) ** 2.)
    mean_err = np.mean(bin_stderr)

    excess_variance = (binned_variance - mean_err) / binned_mean ** 2.
    f = np.sqrt(excess_variance)

    err = (1. / f) * np.sqrt(1. / (2 * bins)) * binned_variance / binned_mean ** 2.

    return f, err


class FvarSpectrum(object):
    """
    pylag.FvarSpectrum

    Class for computing the fractional variability spectrum from a set of light
    curves, one in each energy band (or a set of light curve segments in each
    energy band).

    The resulting Fvarspectrum is stored in the member variables.

    Member Variables
    ----------------
    en       : ndarray
               numpy array containing the central energy of each band
    en_error : ndarray
               numpy array containing the error bar of each energy band (the
               central energy minus the minimum)
    lag      : ndarray
               numpy array containing the fractional variability in each energy
               band
    error    : ndarray
               numpy array containing the error in the fvar measurement in each
               band

    Constructor: pylag.FvarSpectrum(tbin, lclist, enmin, enmax, lcfiles, interp_gaps=False)

    Constructor Arguments
    ---------------------
    tbin        : float
                  Time bin size to be used for fvar/excess variance calculation
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
    """

    def __init__(self, tbin, lclist=None, enmin=None, enmax=None, lcfiles='', interp_gaps=False):
        self.en = np.array([])
        self.en_error = np.array([])
        self.fvar = np.array([])
        self.error = np.array([])

        if lcfiles != '':
            enmin, enmax, lclist = self.find_lightcurves(lcfiles, interp_gaps=interp_gaps)

        self.en = (0.5 * (np.array(enmin) + np.array(enmax)))
        self.en_error = self.en - np.array(enmin)

        if isinstance(lclist[0], LightCurve):
            print("Constructing fvar spectrum in %d energy bins" % len(lclist))
        elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
            print(
                "Constructing fvar spectrum from %d light curves in each of %d energy bins" % (
                    len(lclist[0]), len(lclist)))

        self.fvar, self.error = self.calculate(lclist, tbin)

    def calculate(self, lclist, tbin):
        """
        pylag.fvar.calculate(lclist, tbin)

        Perform fvar spectrum calculation.

        Arguments
        ---------
        lclist : list or list of lists of LightCurves
                 This is either a 1D list containing the light curve in each
                 energy band or 2D list (i.e. list of lists). The outer index
                 corresponds to the energy band. For each energy band, there is a
                 list of LightCurve objects that represent the light curves in that
                 energy band from each observation segment.
        tbin   : float
                 Time bin size to be used for fvar calculation
        """
        en_fvar = []
        error = []

        for enlc in lclist:
            f, e = fvar(enlc, tbin)
            en_fvar.append(f)
            error.append(e)

        return np.array(en_fvar), np.array(error)

    @staticmethod
    def find_lightcurves(searchstr, **kwargs):
        """
        enmin, enmax, lclist = pylag.LagEnergySpectrum.FindLightCurves(searchstr)

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
        searchstr : string
                  : Wildcard for searching the filesystem to find the light curve
                    filesystem

        Returns
        -------
        enmin :  ndarray
                 numpy array countaining the lower energy bound of each band
        enmax :  ndarray
                 numpy array containing the upper energy bound of each band
        lclist : list of list of LightCurve objects
                 The list of light curve segments in each energy band for
                 computing the lag-energy spectrum
        """
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
        return (self.en, self.en_error), (self.fvar, self.error)

    def _getplotaxes(self):
        return 'Energy / keV', 'log', 'fvar', 'log'
