"""
pylag.periodogram

Provides pyLag classes for calculating periodograms/power spectra from light
curves

Classes
-------
Periodogram        : Calculates the periodogram from a light curve
StackedPeriodogram : Calculates the stacked periodogram from multiple light curves

v1.0 09/03/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .binning import *

import numpy as np


class Periodogram(object):
    """
    pylag.Periodogram

    Class to calculate the periodogram from a light curve.

    The constructor is passed a pylag LightCurve object. The periodogram is then
    calculated immediately at each sample frequency present in the discete
    Fourier transform of the input light curve. The result is stored in the
    member variables of this class from where it can be analysed or binned further.

    Light curves should be 'fixed' before calculating the periodogram. At a
    minimum, nan values must be replaced by zeros. Short gaps in the light curve
    can also be interpolated over.

    Member Variables
    ----------------
    freq        : ndarray
                  numpy array storing the sample frequencies at which the
                  periodogram is evaluated
    periodogram : ndarray
                  numpy array storing the calculated periodogram
    error       : ndarray
                  numpy array for the standard error on the periodogram. This is
                  only meaningful for binned periodograms. If not initialised,
                  this will be set to zero for each sample frequency

    Constructor: pylag.Periodogram(lc=None, f=[], per=[], err=None, norm=True)

    Constructor Arguments
    ---------------------
    lc   : LightCurve, optional (default=None)
           pyLag LightCurve object from which the periodogram is computed
    f    : ndarray or list, optional (default=[])
           If no light curve is specified, the sample frequency array can be
           manually initialised using this array (used if storing the result
           from an external calculation)
    per  : ndarray or list, optional (default=[])
           If no light curve is specified, the periodogram can be manually
           initialised using this array (used if storing the result from an
           external calculation)
    err  : ndarray or list, optional (default=None)
           The error on the periodogram at each sample frequency is manually
           initialised using this array
    norm : boolean, optional (default=True)
           If True, the calculated periodogram is normalised to be consistent
           with the PSD (this only takes effect if the periodogram is calculated
           from an input light curve)
    """

    def __init__(self, lc=None, f=[], per=[], err=None, ferr=None, norm=True, **kwargs):
        if lc is not None:
            if not isinstance(lc, LightCurve):
                raise ValueError(
                    "pyLag CrossSpectrum ERROR: Can only compute cross spectrum between two LightCurve objects")

            self.freq, self.periodogram = self.calculate(lc, norm, **kwargs)

        else:
            self.freq = np.array(f)
            self.periodogram = np.array(per)

        # these will only be set once the periodogram is binned
        self.freq_error = ferr
        self.error = err

    def calculate(self, lc, norm=True, uneven=False, **kwargs):
        """
        pylag.Periodogram.calculate(lc, norm=True)

        calculate the periodogram from a light curve and store it in the member
        variables. Sample frequency array is copied from the light curve. The
        discrete Fourier transform is obtained from the FT method in the
        LightCurve class.

        Arguments
        ---------
        lc   : LightCurve
               pyLag LightCurve object from which the periodogram is computed
        norm : boolean, optional (default=True)
               If True, the calculated periodogram is normalised to be consistent
               with the PSD

        Returns
        -------
        f   : ndarray
              numpy array containing the sample frequencies at which the
              periodogram is evaluated
        per : ndarray
              numpy array containing the periodogram at each frequency
        """
        if norm:
            psdnorm = 2 * lc.dt / (lc.mean() ** 2 * lc.length)
        else:
            psdnorm = 1

        if uneven:
            f, ft = lc.ft_uneven(**kwargs)
        else:
            f, ft = lc.ft()

        per = psdnorm * np.abs(ft) ** 2
        return f, per

    def bin(self, bins, calc_error=True):
        """
        perbin = pylag.Periodogram.bin(bins)

        bin the periodogram using a Binning object then return the binned spectrum
        as a new Periodogram object

        Arguments
        ---------
        bins       : Binning
                     pyLag Binning object to perform the Binning
        calc_error : bool, optional (default=True)
                     Whether the error on each bin is required in returned periodogram

        Returns
        -------
        perbin : Periodogram
                 pyLag Periodogram object storing the newly binned periodogram

        """
        if not isinstance(bins, Binning):
            raise ValueError("pyLag Periodogram bin ERROR: Expected a Binning object")

        if calc_error:
            binned_error = bins.std_error(self.freq, self.periodogram)
        else:
            binned_error = None

        return Periodogram(f=bins.bin_cent, per=bins.bin(self.freq, self.periodogram),
                           err=binned_error, ferr=bins.x_error())

    def points_in_bins(self, bins):
        """
        points_in_bins = pylag.Periodogram.points_in_bins(bins)

        bin the periodogram by frequency using a Binning object then return
        the list of data points that fall in each frequency bin.

        Arguments
        ---------
        bins : Binning
               pyLag Binning object to perform the Binning

        Returns
        -------
        points_in_bins : list
                     List of data point values that fall into each frequency bin

        """
        if not isinstance(bins, Binning):
            raise ValueError("pyLag CrossSpectrum bin ERROR: Expected a Binning object")

        return bins.points_in_bins(self.freq, self.periodogram)

    def freq_average_slow(self, fmin, fmax):
        """
        per_avg = pylag.Periodogram.freq_average(fmin, fmax)

        calculate the average value of the periodogram over a specified frequency
        interval.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        per_avg : float
                  The average value of the periodogram over the frequency range
        """
        return np.mean([p for f, p in zip(self.freq, self.periodogram) if fmin <= f < fmax])

    def freq_average(self, fmin, fmax):
        """
        per_avg = pylag.Periodogram.freq_average(fmin, fmax)

        calculate the average value of the periodogram over a specified frequency
        interval.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        per_avg : float
                  The average value of the periodogram over the frequency range
        """
        bin_edges = [fmin, fmax]
        per_mean, _, _ = binned_statistic(self.freq, self.periodogram, statistic='mean', bins=bin_edges)
        return per_mean[0]

    def points_in_freqrange(self, fmin, fmax):
        """
        range_points = pylag.Periodogram.points_in_freqrange(fmin, fmax)

        Return the list of periodogram points that fall in a specified frequency
        interval.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        range_points : list
                       List of periodogram points (complex) in the frequency range

        """
        return [p for f, p in zip(self.freq, self.periodogram) if fmin <= f < fmax]

    def _getplotdata(self):
        return (self.freq, self.freq_error), (self.periodogram, self.error)

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'Periodogram', 'log'


# --- STACKED DATA PRODUCTS ----------------------------------------------------

class StackedPeriodogram(Periodogram):
    """
    pylag.StackedPeriodogram(Periodogram)

    calculate the average periodogram from multiple pairs of light curves
    with some frequency binning.

    The periodogram is calculated for each pair of light curves in turn, then
    the data points are sorted into bins. The final periodogram in each bin
    is the average over all of the individual frequency points from all of the
    light curves that fall into that bin.

    The resulting periodogram is accessible in the same manner as a single
    cross spectrum and analysis can be conducted in the same way.

    Constructor: pylag.StackedPeriodogram(lc1_list, lc2_list, bins)

    Constructor Arguments
    ---------------------
    lc_list : list (of LightCurve objects)
              List containing the pyLag LightCurve objects
    bins    : Binning, optional (default=None)
              pyLag Binning object specifying the binning. If no binning is
              specified, routines accessing the cross spectrum as a function of
              frequency will not be accessible, but the cross spectrum can be
              averaged over specified frequency ranges
    """

    def __init__(self, lc_list, bins=None, calc_error=True, **kwargs):
        self.periodograms = []
        for lc in lc_list:
            self.periodograms.append(Periodogram(lc, **kwargs))

        self.bins = bins
        freq = []
        per = []
        err = []
        ferr = []

        if bins is not None:
            freq = bins.bin_cent
            ferr = bins.x_error()
            per, err = self.calculate_binned(calc_error)
        else:
            freq = self.periodograms[0].freq
            ferr = np.zeros_like(freq)
            per, err = self.calculate_stacked()

        Periodogram.__init__(self, f=freq, per=per, err=err, ferr=ferr)

    def calculate_slow(self, calc_error=True):
        """
        per, err = pylag.StackedPeriodogram.calculate()

        Calculates the average periodogram in each frequency bin. The final
        periodogram in each bin is the average over all of the individual
        frequency points from all of the light curves that fall into that bin.

        Returns
        -------
        per : ndarray
              The average periodogram in each frequency bin
        err : ndarray
              The standard error of the periodogram in each bin
        """
        per_points = []
        for b in self.bins.bin_cent:
            per_points.append([])

        for per in self.periodograms:
            this_per = per.points_in_bins(self.bins)

            # add the individual frequency points for this cross spectrum (for each
            # frequency bin) into the accumulated lists for each frequency bin
            for i, points in enumerate(this_per):
                per_points[i] += points

        # now take the mean of all the points that landed in each bin
        per = []
        err = []
        for freq_points in per_points:
            per.append(np.mean(freq_points))
            err.append(np.std(freq_points) / np.sqrt(len(freq_points)))

        return np.array(per), np.array(err)

    def calculate_stacked(self):
        """
        per, err = pylag.StackedPeriodogram.calculate_stacked()

        Calculates the average periodogram at in each frequency by stacking
        the periodograms from the individual light curves. Requires input
        light curves to have the same time binning and length.

        Returns
        -------
        per : ndarray
              The average periodogram in each frequency bin
        err : ndarray
              The standard error of the periodogram in each bin
        """
        try:
            stacked_per = np.mean(np.vstack([p.periodogram for p in self.periodograms]), axis=0)
            err = np.std(np.vstack([p.periodogram for p in self.periodograms]), axis=0)
        except ValueError:
            # if the time bins don't line up and we're not averaging the periodogram into bins, it doesn't
            # make sense to evaluate the periodogram at individual frequency points, so just return NaN
            return np.nan, np.nan

        return stacked_per, err

    def calculate_binned(self, calc_error=True):
        """
        per, err = pylag.StackedPeriodogram.calculate_binned()

        Calculates the average periodogram in each frequency bin. The final
        periodogram in each bin is the average over all of the individual
        frequency points from all of the light curves that fall into that bin.

        Returns
        -------
        per : ndarray
              The average periodogram in each frequency bin
        err : ndarray
              The standard error of the periodogram in each bin
        """
        freq_list = np.hstack([p.freq for p in self.periodograms])
        per_list = np.hstack([p.periodogram for p in self.periodograms])

        if calc_error:
            error = self.bins.std_error(freq_list, per_list)
        else:
            error = None

        return self.bins.bin(freq_list, per_list), error

    def freq_average_slow(self, fmin, fmax):
        """
        per_avg = pylag.StackedPeriodogram.freq_average(fmin, fmax)

        calculate the average value of the periodogram over a specified
        frequency interval. The final periodogram is the average over all of
        the individual frequency points from all of the light curves that fall
        into the range.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        per_avg : complex
                  The average value of the cross spectrum over the frequency range

        """
        per_points = []
        for per in self.periodograms:
            per_points += per.points_in_freqrange(fmin, fmax)

        return np.mean(per_points)

    def freq_average(self, fmin, fmax):
        """
        per_avg = pylag.StackedPeriodogram.freq_average(fmin, fmax)

        calculate the average value of the periodogram over a specified
        frequency interval. The final periodogram is the average over all of
        the individual frequency points from all of the light curves that fall
        into the range.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        per_avg : complex
                  The average value of the cross spectrum over the frequency range

        """
        freq_list = np.hstack([p.freq for p in self.periodograms])
        per_list = np.hstack([p.periodogram for p in self.periodograms])

        bin_edges = [fmin, fmax]
        per_mean, _, _ = binned_statistic(freq_list, per_list, statistic='mean', bins=bin_edges)
        return per_mean[0]
