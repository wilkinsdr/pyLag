"""
pylag.cross_spectrum

Provides pyLag classes for calculating cross spectrum; the first step in X-ray
timing/lag analysis

Classes
-------
CrossSpectrum        : Calculates the cross spectrum between two light curves
StackedCrossSpectrum : Calculates the stacked cross spectrum from multiple pairs
                       of light curvs

v1.0 09/03/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .binning import *

import numpy as np
from scipy.stats import binned_statistic


class CrossSpectrum(object):
    """
    pylag.CrossSpectrum

    Class to calculate the cross spectrum from a pair of light curves; the first
    step in X-ray timing/lag analysis.

    Light curves must have the same time binning and be of the same length.

    The constructor is passed two pylag LightCurve objects. The cross spectrum
    is then	calculated immediately at each sample frequency present in the discete
    Fourier transform of the input light curves. The result is stored in the
    member variables of this class from where it can be analysed or binned further.

    This class provides functionality for calculating the phase/time lags between
    the light curves, either in each frequency bin or averaged over a range of
    frequencies.

    In the cross spectrum calculation, the complex conjugate is taken of the lc1
    DFT. This results in a positive lag denoting lc1 lagging BEHIND lc2. Thus, in
    order to adopt the common sign convention, lc1 is the HARD band light curve
    and lc2 is the SOFT band or REFERENCE light curve.

    Light curves should be 'fixed' before calculating the cross spectrum. At a
    minimum, nan values must be replaced by zeros. Short gaps in the light curve
    can also be interpolated over.

    Member Variables
    ----------------
    freq : ndarray
           numpy array storing the sample frequencies at which the
           cross spectrum is evaluated
    cs   : ndarray
           numpy array (complex) storing the calculated cross spectrum

    Constructor: pylag.CrossSpectrum(lc1=None, lc2=None, f=[], cs=[], norm=True)

    Constructor Arguments
    ---------------------
    lc1    : LightCurve, optional (default=None)
             pyLag LightCurve object for the primary or hard band (complex
             conjugated during cross spectrum calculation)
    lc2    : LightCurve, optional (default=None)
             pyLag LightCurve object for the reference or soft band
    f      : ndarray or list, optional (default=[])
             If no light curve is specified, the sample frequency array can be
             manually initialised using this array (used if storing the result
             from an external calculation)
    cs     : ndarray or list, optional (default=[])
             If no light curve is specified, the cross spectrum can be manually
             initialised using this array (used if storing the result from an
             external calculation)
    norm   : boolean, optional (default=True)
             If True, the calculated cross spectrum is normalised to be consistent
             with the PSD normalisation (this only takes effect if the cross
             spectrum is calculated from input light curves)
    uneven : boolean, optional (default=False)
             True if the light curves have gaps or uneven time sampling. If True,
             Fourier transforms will be directly evaluated using the method of
             Scargle 1989 for unevenly sampled time series
    """

    def __init__(self, lc1=None, lc2=None, f=[], cs=[], ferr=None, norm=True, **kwargs):
        if lc1 is not None and lc2 is not None:
            if not (isinstance(lc1, LightCurve) and isinstance(lc2, LightCurve)):
                raise ValueError(
                    "pyLag CrossSpectrum ERROR: Can only compute cross spectrum between two LightCurve objects")

            if lc1 != lc2:
                lc1, lc2 = extract_sim_lightcurves(lc1, lc2)
                #raise AssertionError(
                #    "pyLag CrossSpectrum ERROR: Light curves must be the same length and have same time binning to compute cross spectrum")

            self.freq, self.crossft = self.calculate(lc1, lc2, norm, **kwargs)

        else:
            self.freq = np.array(f)
            self.crossft = np.array(cs)
            self.ferr = ferr

    def calculate(self, lc1, lc2, norm=True, uneven=False, **kwargs):
        """
        f, crossft = pylag.CrossSpectrum.calculate(lc1, lc2, norm=True)

        calculate the cross spectrum from a pair of light curves and store it in
        the member variables. Sample frequency array is copied from the first
        light curve. The discrete Fourier transforms are obtained from the FT
        method in the LightCurve class.

        In the cross spectrum calculation, the complex conjugate is taken of the lc1
        DFT. This results in a positive lag denoting lc1 lagging BEHIND lc2. Thus, in
        order to adopt the common sign convention, lc1 is the HARD band light curve
        and lc2 is the SOFT band or REFERENCE light curve.

        Arguments
        ---------
        lc1  : LightCurve
               pyLag LightCurve object for the primary or hard band (complex
               conjugated during cross spectrum calculation)
        lc2  : LightCurve
               pyLag LightCurve object for the reference or soft band
        norm : boolean, optional (default=True)
               If True, the calculated cross spectrum is normalised to be consistent
               with the PSD normalisation

        Returns
        -------
        f       : ndarray
                  numpy array containing the sample frequencies at which the
                  cross spectrum is evaluated
        crossft : ndarray
                  numpy array containing the (complex) cross spectrum at each
                  frequency
        """
        if norm:
            crossnorm = 2 * lc1.dt / (lc1.mean() * lc2.mean() * lc1.length)
        else:
            crossnorm = 1

        if uneven:
            f1, ft1 = lc1.ft_uneven(**kwargs)
            _, ft2 = lc2.ft_uneven(**kwargs)
        else:
            f1, ft1 = lc1.ft()
            _, ft2 = lc2.ft()

        crossft = crossnorm * np.conj(ft1) * ft2
        return f1, crossft

    def bin(self, bins):
        """
        csbin = pylag.CrossSpectrum.bin(bins)

        bin the cross spectrum by frequency using a Binning object then return
        the binned spectrum as a new CrossSpectrum object

        Arguments
        ---------
        bins : Binning
               pyLag Binning object to perform the Binning

        Returns
        -------
        csbin : CrossSpectrum
                pyLag CrossSpectrum object storing the newly binned spectrum

        """
        if not isinstance(bins, Binning):
            raise ValueError("pyLag CrossSpectrum bin ERROR: Expected a Binning object")

        return CrossSpectrum(f=bins.bin_cent, cs=bins.bin(self.freq, self.crossft), ferr=bins.x_error())

    def points_in_bins(self, bins):
        """
        points_in_bins = pylag.CrossSpectrum.points_in_bins(bins)

        bin the cross spectrum by frequency using a Binning object then return
        the list of data points that fall in each frequency bin.

        Arguments
        ---------
        bins : Binning
               pyLag Binning object to perform the Binning

        Returns
        -------
        points_in_bins : list
                     List of data point values (complex) that fall into each
                     frequency bin

        """
        if not isinstance(bins, Binning):
            raise ValueError("pyLag CrossSpectrum bin ERROR: Expected a Binning object")

        return bins.points_in_bins(self.freq, self.crossft)

    def freq_average(self, fmin, fmax):
        """
        csavg = pylag.CrossSpectrum.freq_average(fmin, fmax)

        calculate the average value of the cross spectrum over a specified
        frequency interval.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        csavg : complex
                The average value of the cross spectrum over the frequency range

        """
        return np.mean([c for f, c in zip(self.freq, self.crossft) if fmin <= f < fmax])

    def points_in_freqrange(self, fmin, fmax):
        """
        range_points = pylag.CrossSpectrum.points_in_freqrange(fmin, fmax)

        Return the list of cross spectrum points that fall in a specified
        frequency interval.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        range_points : list
                       List of cross spectrum points (complex) in the frequency
                       range

        """
        return [c for f, c in zip(self.freq, self.crossft) if fmin <= f < fmax]

    def lag_spectrum(self):
        """
        freq, lag = pylag.CrossSpectrum.LagSpectrum()

        Return the lag/frequency spectrum: The time lag between correlated
        variability in the two light curves as a function of Fourier frequency

        Returns
        -------
        freq : ndarray
               numpy array containing the sample frequencies
        lag  : ndarray
               numpy array containing the time lag (in seconds) at each sample
               frequency

        """
        lag = np.angle(self.crossft) / (2 * np.pi * self.freq)
        return self.freq, lag

    def lag_average(self, fmin, fmax):
        """
        lagavg = pylag.CrossSpectrum.freq_average(fmin, fmax)

        calculate the average value of the time lag over a specified frequency
        interval.

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Return
        ------
        lagavg : float
                 The average value of the time lag over the frequency range
                 (in seconds)

        """
        avgcross = self.freq_average(fmin, fmax)
        lag = np.angle(avgcross) / (2 * np.pi * np.mean([fmin, fmax]))
        return lag

    def cross_power(self, psdslope=0.):
        from .plotter import DataSeries
        if self.ferr is not None:
            xdata = (self.freq, self.ferr)
        else:
            xdata = self.freq
        return DataSeries(x=xdata, y=np.abs(self.freq**-psdslope * self.crossft), xlabel='Frequency / Hz', xscale='log', ylabel='Cross Power', yscale='log')


# --- STACKED DATA PRODUCTS ----------------------------------------------------

class StackedCrossSpectrum(CrossSpectrum):
    """
    pylag.StackedCrossSpectrum(CrossSpectrum)

    calculate the average cross spectrum from multiple pairs of light curves
    with some frequency binning.

    In the cross spectrum calculation, the complex conjugate is taken of the lc1
    DFT. This results in a positive lag denoting lc1 lagging BEHIND lc2. Thus, in
    order to adopt the common sign convention, lc1 is the HARD band light curve
    and lc2 is the SOFT band or REFERENCE light curve.

    The cross spectrum is calculated for each pair of light curves in turn, then
    the data points are sorted into bins. The final cross spectrum in each bin
    is the average over all of the individual frequency points from all of the
    light curves that fall into that bin.

    The resulting cross spectrum is accessible in the same manner as a single
    cross spectrum and analysis can be conducted in the same way.

    Constructor: pylag.StackedCrossSpectrum(lc1_list, lc2_list, bins)

    Constructor Arguments
    ---------------------
    lc1_list : list (of LightCurve objects)
               List containing the pyLag LightCurve objects for the subject or
               hard band.
    lc2_list : list (of LightCurve objects)
               List containing the pyLag LightCurve objects for the reference or
               soft band.
    bins     : Binning, optional (default=None)
               pyLag Binning object specifying the binning. If no binning is
               specified, routines accessing the cross spectrum as a function of
               frequency will not be accessible, but the cross spectrum can be
               averaged over specified frequency ranges
    """

    def __init__(self, lc1_list, lc2_list, bins=None, norm=True, **kwargs):
        self.cross_spectra = []
        for lc1, lc2 in zip(lc1_list, lc2_list):
            self.cross_spectra.append(CrossSpectrum(lc1, lc2, norm=norm, **kwargs))

        self.bins = bins

        freq = []
        crossft = []

        if bins is not None:
            freq = bins.bin_cent
            crossft = self.calculate()

        CrossSpectrum.__init__(self, f=freq, cs=crossft)

    def calculate_slow(self):
        """
        pylag.StackedCrossSpectrum.StackBinnedCrossSpectrum()

        Calculates the average cross spectrum in each frequency bin. The final
        cross spectrum in each bin is the average over all of the individual
        frequency points from all of the light curves that fall into that bin.

        Returns
        -------
        cross_spec : ndarray
                     The average cross spectrum (complex) in each frequency bin
        """
        cross_spec_points = []
        for b in self.bins.bin_cent:
            cross_spec_points.append([])

        for cs in self.cross_spectra:
            this_cross_spec = cs.points_in_bins(self.bins)

            # add the individual frequency points for this cross spectrum (for each
            # frequency bin) into the accumulated lists for each frequency bin
            for i, points in enumerate(this_cross_spec):
                cross_spec_points[i] += points

        # now take the mean of all the points that landed in each bin
        cross_spec = []
        for freq_points in cross_spec_points:
            cross_spec.append(np.mean(freq_points))

        return np.array(cross_spec)

    def calculate(self):
        """
        pylag.StackedCrossSpectrum.StackBinnedCrossSpectrum()

        Calculates the average cross spectrum in each frequency bin. The final
        cross spectrum in each bin is the average over all of the individual
        frequency points from all of the light curves that fall into that bin.

        Returns
        -------
        cross_spec : ndarray
                     The average cross spectrum (complex) in each frequency bin
        """
        freq_list = np.hstack([c.freq for c in self.cross_spectra])
        crossft_list = np.hstack([c.crossft for c in self.cross_spectra])
        return self.bins.bin(freq_list, crossft_list)

    def freq_average_slow(self, fmin, fmax):
        """
        csavg = pylag.CrossSpectrum.freq_average(fmin, fmax)

        calculate the average value of the cross spectrum over a specified
        frequency interval. The final cross spectrum is the average over all of
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
        csavg : complex
                The average value of the cross spectrum over the frequency range

        """
        cross_spec_points = []
        for cs in self.cross_spectra:
            cross_spec_points += cs.points_in_freqrange(fmin, fmax)

        return np.mean(cross_spec_points)

    def freq_average(self, fmin, fmax):
        """
        csavg = pylag.CrossSpectrum.freq_average(fmin, fmax)

        calculate the average value of the cross spectrum over a specified
        frequency interval. The final cross spectrum is the average over all of
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
        csavg : complex
                The average value of the cross spectrum over the frequency range

        """
        freq_list = np.hstack([c.freq for c in self.cross_spectra])
        crossft_list = np.hstack([c.crossft for c in self.cross_spectra])

        bin_edges = [fmin, fmax]
        real_mean, _, _ = binned_statistic(freq_list, crossft_list.real, statistic='mean', bins=bin_edges)
        imag_mean, _, _ = binned_statistic(freq_list, crossft_list.imag, statistic='mean', bins=bin_edges)
        return np.complex(real_mean, imag_mean)
