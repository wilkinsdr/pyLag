"""
pylag.coherence

Provides pyLag class for computing the coherence between two light curves, for
calculating lag errors

Classes
-------
Coherence : Calculate the coherence between a pair of light curves

v1.0 09/03/2017 - D.R. Wilkins
"""
from pylag.lightcurve import *
from pylag.cross_spectrum import *
from pylag.periodogram import *
from pylag.binning import *

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
	def __init__(self, lc1=None, lc2=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0., bias=True):
		self.bkg1 = bkg1
		self.bkg2 = bkg2

		self.coh = np.array([])
		self.num_freq = np.array([])

		self.bins = bins

		if(bins != None):
			self.freq = bins.bin_cent
			self.freq_error = bins.XError()
		elif(fmin>0 and fmax>0):
			self.freq = np.mean([fmin, fmax])
			self.freq_error = None

		# if we're passed a single pair of light curves, get the cross spectrum
		# and periodograms and count the number of sample frequencies in either
		# the bins or specified range
		if(isinstance(lc1, LightCurve) and isinstance(lc2, LightCurve)):
			self.cross_spec = CrossSpectrum(lc1, lc2)
			self.per1 = Periodogram(lc1)
			self.per2 = Periodogram(lc2)
			if(bins != none):
				self.num_freq = lc1.BinNumFreq(bins)
			elif(fmin>0 and fmax>0):
				self.num_freq = lc1.NumFreqInRange(fmin, fmax)
			self.lc1mean = lc1.Mean()
			self.lc2mean = lc2.Mean()

		# if we're passed lists of light curves, get the stacked cross spectrum
		# and periodograms and count the number of sample frequencies across all
		# the light curves
		elif(isinstance(lc1, list) and isinstance(lc2, list)):
			self.cross_spec = StackedCrossSpectrum(lc1, lc2, bins)
			self.per1 = StackedPeriodogram(lc1, bins)
			self.per2 = StackedPeriodogram(lc2, bins)
			if(bins != None):
				self.num_freq = np.zeros(bins.num)

				for lc in lc1:
					self.num_freq += lc.BinNumFreq(bins)
			elif(fmin>0 and fmax>0):
				self.num_freq = 0
				for lc in lc1:
					self.num_freq += lc.NumFreqInRange(fmin, fmax)
			self.lc1mean = StackedMeanCountRate(lc1)
			self.lc2mean = StackedMeanCountRate(lc2)

		self.coh = self.Calculate(bins, fmin, fmax, bias)

	def Calculate(self, bins=None, fmin=None, fmax=None, bias=True):
		"""
		pylag.Coherence.Calculate(bins=None, fmin=None, fmax=None)

		Calculate the coherence either in each bin or over a specified frequency
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
		fmin : float
			   Upper bound of frequency range

		Return Values
		-------------
		coh : ndarray or float
		      The calculated coherence either as a numpy array containing the
			  value for each frequency bin or a single float if the coherence is
			  calculated over a single frequency range
		"""
		if(bins != None):
			cross_spec = self.cross_spec.Bin(self.bins).crossft
			per1 = self.per1.Bin(self.bins).periodogram
			per2 = self.per2.Bin(self.bins).periodogram
		elif(fmin>0 and fmax>0):
			cross_spec = self.cross_spec.FreqAverage(fmin, fmax)
			per1 = self.per1.FreqAverage(fmin, fmax)
			per2 = self.per2.FreqAverage(fmin, fmax)

		if bias:
			pnoise1 = 2*(self.lc1mean + self.bkg1) / self.lc1mean**2
			pnoise2 = 2*(self.lc2mean + self.bkg2) / self.lc2mean**2
			nbias = (pnoise2*(per1 - pnoise1) + pnoise1*(per2 - pnoise2) + pnoise1*pnoise2) / self.num_freq
		else:
			nbias = 0

		coh = (np.abs(cross_spec)**2 - nbias) / (per1 * per2)
		return coh

	def PhaseError(self):
		return np.sqrt( (1 - self.coh) / (2 * self.coh * self.num_freq) )

	def LagError(self):
		return self.PhaseError() / (2*np.pi*self.freq)

	def _getplotdata(self):
		return self.freq, self.coh, self.freq_error, None

	def _getplotaxes(self):
		return 'Frequency / Hz', 'log', 'Coherence', 'linear'
