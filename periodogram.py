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
from pylag.lightcurve import *
from pylag.binning import *

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
	def __init__(self, lc=None, f=[], per=[], err=None, ferr=None, norm=True):
		if(lc != None):
			if not isinstance(lc, LightCurve):
				raise ValueError("pyLag CrossSpectrum ERROR: Can only compute cross spectrum between two LightCurve objects")

			self.freq, self.periodogram = self.Calculate(lc, norm)

		else:
			self.freq = np.array(f)
			self.periodogram = np.array(per)

		self.freq_error = ferr
		self.error = err
		# if(len(err)>0):
		# 	self.error = np.array(err)
		# else:
		# 	self.error = np.zeros(len(self.freq))

	def Calculate(self, lc, norm=True):
		"""
		pylag.Periodogram.Calculate(lc, norm=True)

		Calculate the periodogram from a light curve and store it in the member
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

		Return Values
		-------------
		f   : ndarray
		      numpy array containing the sample frequencies at which the
			  periodogram is evaluated
		per : ndarray
		      numpy array containing the periodogram at each frequency
		"""
		if norm:
			psdnorm = 2*lc.dt / (lc.Mean()**2 * lc.length)
		else:
			psdnorm = 1

		f, ft = lc.FT()
		per = psdnorm * np.abs(ft)**2
		return f, per

	def Bin(self, bins):
		"""
		perbin = pylag.Periodogram.Bin(bins)

		Bin the periodogram using a Binning object then return the binned spectrum
		as a new Periodogram object

		Arguments
		---------
		bins : Binning
			   pyLag Binning object to perform the Binning

		Return Values
		-------------
		perbin : Periodogram
		         pyLag Periodogram object storing the newly binned periodogram

		"""
		if not isinstance(bins, Binning):
			raise ValueError("pyLag CrossSpectrum bin ERROR: Expected a Binning object")

		return Periodogram(f=bins.bin_cent, per=bins.Bin(self.freq, self.periodogram), err=bins.StdError(self.freq, self.periodogram), ferr=bins.XError())

	def BinPoints(self, bins):
		"""
		bin_points = pylag.Periodogram.BinPoints(bins)

		Bin the periodogram by frequency using a Binning object then return
		the list of data points that fall in each frequency bin.

		Arguments
		---------
		bins : Binning
			   pyLag Binning object to perform the Binning

		Return Values
		-------------
		bin_points : list
		             List of data point values that fall into each frequency bin

		"""
		if not isinstance(bins, Binning):
			raise ValueError("pyLag CrossSpectrum bin ERROR: Expected a Binning object")

		return bins.BinPoints(self.freq, self.periodogram)

	def FreqAverage(self, fmin, fmax):
		"""
		per_avg = pylag.Periodogram.FreqAverage(fmin, fmax)

		Calculate the average value of the periodogram over a specified frequency
		interval.

		Arguments
		---------
		fmin : float
			   Lower bound of frequency range
		fmin : float
			   Upper bound of frequency range

		Return Values
		-------------
		per_avg : float
		          The average value of the periodogram over the frequency range
		"""
		return np.mean(  [p for f,p in zip(self.freq,self.periodogram) if f>=fmin and f<fmax] )

	def FreqRangePoints(self, fmin, fmax):
		"""
		range_points = pylag.Periodogram.FreqRangePoints(fmin, fmax)

		Return the list of periodogram points that fall in a specified frequency
		interval.

		Arguments
		---------
		fmin : float
			   Lower bound of frequency range
		fmin : float
			   Upper bound of frequency range

		Return Values
		-------------
		range_points : list
		               List of periodogram points (complex) in the frequency range

		"""
		return [p for f,p in zip(self.freq,self.periodogram) if f>=fmin and f<fmax]

	def _getplotdata(self):
		return (self.freq, self.freq_error), (self.periodogram, self.error)

	def _getplotaxes(self):
		return 'Frequency / Hz', 'log', 'Periodogram', 'log'


### --- STACKED DATA PRODUCTS --------------------------------------------------

class StackedPeriodogram(Periodogram):
	"""
	pylag.StackedPeriodogram(Periodogram)

	Calculate the average periodogram from multiple pairs of light curves
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
	def __init__(self, lc_list, bins=None):
		self.periodograms = []
		for lc in lc_list:
			self.periodograms.append( Periodogram(lc) )

		self.bins = bins
		freq = []
		per = []
		err = []
		ferr = []

		if(bins != None):
			freq = bins.bin_cent
			ferr = bins.XError()
			per, err = self.Calculate()

		Periodogram.__init__(self, f=freq, per=per, err=err, ferr=ferr)

	def Calculate(self):
		"""
		per, err = pylag.StackedPeriodogram.Calculate()

		Calculates the average periodogram in each frequency bin. The final
		periodogram in each bin is the average over all of the individual
		frequency points from all of the light curves that fall into that bin.

		Return Values
		-------------
		per : ndarray
		      The average periodogram in each frequency bin
		err : ndarray
			  The standard error of the periodogram in each bin
		"""
		per_points = []
		for b in self.bins.bin_cent:
			per_points.append([])

		for per in self.periodograms:
			this_per = per.BinPoints(self.bins)

			# add the individual frequency points for this cross spectrum (for each
			# frequency bin) into the accumulated lists for each frequency bin
			for i, points in enumerate(this_per):
				per_points[i] += points

		# now take the mean of all the points that landed in each bin
		per = []
		err = []
		for freq_points in per_points:
			per.append( np.mean(freq_points) )
			err.append( np.std(freq_points) / np.sqrt(len(freq_points)) )

		return np.array(per), np.array(err)

	def FreqAverage(self, fmin, fmax):
		"""
		per_avg = pylag.StackedPeriodogram.FreqAverage(fmin, fmax)

		Calculate the average value of the periodogram over a specified
		frequency interval. The final periodogram is the average over all of
		the individual frequency points from all of the light curves that fall
		into the range.

		Arguments
		---------
		fmin : float
			   Lower bound of frequency range
		fmin : float
			   Upper bound of frequency range

		Return Values
		-------------
		per_avg : complex
		          The average value of the cross spectrum over the frequency range

		"""
		per_points = []
		for per in self.periodograms:
			per_points += per.FreqRangePoints(fmin, fmax)

		return np.mean(per_points)
