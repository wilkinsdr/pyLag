"""
pyLag - X-ray spectral timing analysis in Python

v1.0 09/03/2017 - D.R. Wilkins
"""
import pyfits
import numpy as np
import scipy.fftpack

### --- RAW DATA / LIGHT CURVES ------------------------------------------------

class LightCurve(object):
	"""
	pylag.LightCurve

	Class to read, store and manage X-ray light curves. Light curves are read
	from OGIP standard FITS files and stored in numpy arrays for analysis.

	A number of operations are supported on the stored light curves including
	interpolation over gaps in the time series and Fourier transforms.

	Member Variables
	----------------
	time   : ndarray
	         The time axis points at which the light curve is sampled
	rate   : ndarray
	         The count rate
	error  : ndarray
	         The error in the count rate
	dt     : float
	         The time step between bins
	length : int
	         The number of time bins in the light curve

	Constructor: pylag.LightCurve(filename, t=[], r=[], e=[], fix=False)

	Constructor Arguments
	---------------------
	filename : string, optional (default=None)
	           If set, the light curve will be read from the specified FITS file
			   Specifying an input file will overwrite all other values set for
			   the light curve.
	t   : ndarray or list, optional (default=[])
	      If set, the time axis points with which the light curve will be initialised
	r   : ndarray or list, optional (default=[])
	      If set, the count rate for each point in the light curve
	e   : ndarray or list, optional (default=[])
	      If set, the error for each point in the light curve
	interp_gaps : boolean, optional (default=False)
	              If true, after reading the light curve, the routine to
		          interpolate over gaps will be run automatically
	zero_nan    : boolean, optional (default=True)
	              If true, after reading the light curve, the routine to replace
				  nan values with zeros in the coun rate will be run

	Overloaded Operators
	--------------------
	LightCurve + LightCurve : Add the count rates from two light curves into a
	 						  new LightCurve object (e.g. for summing observations
							  from two detectors).
							  Errors from the ERROR columns are combined in
							  quadrature.
							  Light curves must have the same length and time
							  binning.

  	LightCurve - LightCurve : Subtract the second light curve from the first and
	                          return the result in a new LightCurve object
							  (e.g. for background subtraction).
  							  Errors from the ERROR columns are combined in
  							  quadrature.
  							  Light curves must have the same length and time
  							  binning.
	"""
	def __init__(self, filename=None, t=[], r=[], e=[], interp_gaps=False, zero_nan=True):
		self.time = np.array(t)
		self.rate = np.array(r)
		self.error = np.array(e)

		if(filename != None):
			self.ReadFITS(filename)

		self.dt = self.time[1] - self.time[0]
		self.length = len(self.rate)

		if interp_gaps:
			self.InterpGaps()
		if zero_nan:
			self.ZeroNaN()

	def ReadFITS(self, filename):
		"""
		pylag.LightCurve.ReadFITS(filename)

		Read the light curve from an OGIP standard FITS file.

		Paremeters
		----------
		filename : string
		           The path of the FITS file from which the light curve is loaded
		"""
		try:
			print "Reading light curve from ", filename
			fitsfile = pyfits.open(filename)
			tabdata = fitsfile['RATE'].data

			self.time = np.array(tabdata['TIME'])
			self.rate = np.array(tabdata['RATE'])
			self.error = np.array(tabdata['ERROR'])

			self.dt = self.time[1] - self.time[0]

			fitsfile.close()

		except:
			raise AssertionError("pyLag LightCurve ERROR: Could not read light curve from FITS file")

	def ZeroTime(self):
		"""
		pylag.LightCurve.ZeroTime()

		Shift the time axis of the light curve to start at zero. The modified
		time axis is stored	back in the original object.
		"""
		self.time = self.time - self.time.min()

	def InterpGaps(self):
		"""
		pylag.LightCurve.InterpGaps()

		Interpolate over gaps within the light curve for fixing gaps left by GTI
		filters when performing timing analysis.

		The missing data points are filled in by linear interpolation between the
		start end end points of the gap and the patched light curve is stored
		back in the original object.
		"""
		in_gap = False
		gap_count = 0
		max_gap = 0

		for i in range(len(self.rate)):
			if not in_gap:
				if np.isnan(self.rate[i]):
					in_gap = True
					gap_start = i-1
					gap_count += 1

			elif in_gap:
				if not np.isnan(self.rate[i]):
					gap_end = i
					in_gap = False

					self.rate[gap_start:gap_end] = np.interp(self.time[gap_start:gap_end], [self.time[gap_start], self.time[gap_end]], [self.rate[gap_start], self.rate[gap_end]])

					gap_length = gap_end - gap_start
					if(gap_length > max_gap):
						max_gap = gap_length

		print "Patched ", gap_count, " gaps"
		print "Longest gap was ", max_gap, " bins"

	def ZeroNaN(self):
		"""
		pylag.LightCurve.InterpGaps()

		Replace all instances of nan with zero in the light curve. The modified
		light curve is stored back in the original object.
		"""
		self.rate[np.isnan(self.rate)] = 0
		self.error[np.isnan(self.error)] = 0

	def TimeSegment(self, start, end):
		"""
		lc = pylag.LightCurve.TimeSegment(start, end)

		Returns the Fourier transform (FFT) of the light curve.
		Only the positive frequencies are returned (N/2 points for light curve length N)

		Arguments
		---------
		start : float
		        The start time from which the extracted light curve begins
		end   : float
		        The end time to which the extracted light curve runs

		Return Values
		-------------
		lc : LightCurve
		     The extracted light curve segment as a new LightCurve object
		"""
		this_t = np.array([t for t in self.time if t>start and t<=end])
		this_rate = np.array([r for t,r in zip(self.time,self.rate) if t>start and t<=end])
		this_error = np.array([e for t,e in zip(self.time,self.error) if t>start and t<=end])
		return LightCurve(t=this_t, r=this_rate, e=this_error)

	def Mean(self):
		"""
		mean = pylag.LightCurve.Mean()

		Returns the mean count rate over the light curve

		Return Values
		-------------
		mean : float
		       The mean count rate
		"""
		return np.mean(self.rate)

	def FT(self):
		"""
		freq, ft = pylag.LightCurve.FT()

		Returns the discrete Fourier transform (FFT) of the light curve.
		Only the positive frequencies are returned (N/2 points for light curve length N)

		Return Values
		-------------
		freq : ndarray
		       The sample frequencies
		ft   : ndarray
		       The discrete Fourier transfrom of the light curve
		"""
		ft = np.fft.fft(self.rate)
		freq = np.fft.fftfreq(self.length, d=self.dt)

		return freq[:self.length/2], ft[:self.length/2]

	def BinNumFreq(self, bins):
		"""
		numfreq = pylag.LightCurve.BinNumFreq(bins)

		Returns the number of sample frequencies that fall into bins specified
		in a pyLag Binning object

		Arguments
		---------
		bins : Binning
			   pyLag Binning object defining the bins into which sample frequencies
			   are to be counted

		Return Values
		-------------
		numfreq : ndarray
		          The number of frequencies falling into each bin
		"""
		freq = np.fft.fftfreq(self.length, d=self.dt)

		return bins.BinNumPoints(freq)

	def NumFreqInRange(self, fmin, fmax):
		"""
		num_freq = pylag.LightCurve.NumFreqInRange()

		Returns the number of sample frequencies that fall into bins specified
		in a pyLag Binning object

		Arguments
		---------
		fmin : float
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

		Return Values
		-------------
		numfreq : ndarray
		          The number of frequencies falling into each bin
		"""
		freq = np.fft.fftfreq(self.length, d=self.dt)

		return len( [f for f in freq if f>=fmin and f<fmax] )

	def __add__(self, other):
		"""
		Overloaded + operator to add two light curves together and return the
		result in a new LightCurve object (e.g. to sum observations from 2
		detectors).

		ERROR columns from the two light curves are summed in quadrature.
		"""
		if isinstance(other, LightCurve):
			if(len(self.rate) != len(other.rate)):
				raise AssertionError("pyLag LightCurve ERROR: Cannot add light curves of different lengths")
				return
			# sum the count rate
			newlc = self.rate + other.rate
			# sum the errors in quadrature
			newerr = np.sqrt( self.error**2 + other.error**2 )
			# construct a new LightCurve with the result
			return LightCurve(t=self.time, r=newlc, e=newerr)

		else:
			return NotImplemented

	def __sub__(self, other):
		"""
		Overloaded + operator to subtract a LightCurve object from this one and
		return the result in a new LightCurve object (e.g. for background
		subtraction).

		ERROR columns from the two light curves are summed in quadrature.
		"""
		if isinstance(other, LightCurve):
			if(len(self.rate) != len(other.rate)):
				raise AssertionError("pyLag LightCurve ERROR: Cannot subtract light curves of different lengths")
				return
			# subtract the count rate
			newlc = self.rate - other.rate
			# sum the errors in quadrature
			newerr = np.sqrt( self.error**2 + other.error**2 )
			# construct a new LightCurve with the result
			return LightCurve(t=self.time, r=newlc, e=newerr)

		else:
			return NotImplemented

	def __eq__(self, other):
		"""
		Overloaded == operator to check light curve lengths and time binning are
		consistent to see if they can be added/subtracted/combined.
		"""
		return (self.length == other.length and self.dt == other.dt)

	def __ne__(self, other):
		"""
		Overloaded == operator to check light curve lengths and time binning are
		inconsistent to see if they can't be added/subtracted/combined.
		"""
		return not (self.length == other.length and self.dt == other.dt)


### --- BINNING UTILITIES ------------------------------------------------------

class Binning(object):
	"""
	pylag.Binning

	Base class to perform binning of data products. Not to be used directly; use
	the specific binning classes derived from this.

	Member Variables
	----------------
	bin_start : ndarray
			    The lower bound of each bin
	bin_end   : ndarray
			    The upper bound of each bin
	bin_cent  : ndarray
			    The central value of each bin
	num       : int
				The number of bins
	"""
	def Bin(self, x, y):
		"""
		binned = pylag.Binning.Bin(x, y)

		Bin (x,y) data by x values into the bins specified by this object and
		return the mean value in each bin.

		Arguments
		---------
		x : ndarray or list
		    The abcissa of input data points that are to be binned
		y : ndarray or list
		    The ordinate/value of input data points

		Return Values
		-------------
		binned : ndarray
		         The mean value in each bin

		"""
		binned = []
		for start,end in zip(self.bin_start, self.bin_end):
			binned.append( np.mean( [b for a,b in zip(x,y) if a>=start and a<end] ) )

		return np.array(binned)


	def BinPoints(self, x, y):
		"""
		bin_points = pylag.Binning.BinPoints(x, y)

		Bin (x,y) data by x values into the bins specified by this object and
		return the list of values that fall in each bin.

		Arguments
		---------
		x : ndarray or list
		    The abcissa of input data points that are to be binned
		y : ndarray or list
		    The ordinate/value of input data points

		Return Values
		-------------
		bin_points : list (2-dimensional)
		             A list containing the list of data points for each bin
		"""
		bin_points = []
		for start,end in zip(self.bin_start, self.bin_end):
			bin_points.append( [b for a,b in zip(x,y) if a>=start and a<end] )

		return bin_points

	def BinNumPoints(self, x):
		"""
		bin_num = pylag.Binning.BinNumPoints(x, y)

		Return the number of data points that fall into each bin.

		Arguments
		---------
		x : ndarray or list
		    The abcissa of input data points that are to be binned

		Return Values
		-------------
		bin_num : ndarray
		          The number of data points that fall into each bin

		"""
		bin_num = []
		for start,end in zip(self.bin_start, self.bin_end):
			bin_num.append( len( [a for a in x if a>=start and a<end] ) )

		return np.array(bin_num)


class LogBinning(Binning):
	"""
	pylag.LogBinning(Binning)

	Class to perform binning of data products into logarithmically-spaced bins.

	Constructor: pylag.LogBinning(minval, maxval, num)

	Constructor Arguments
	---------------------
	minval : float
	         The lowest bound of the bottom bin
	maxval : float
	         The upper bound of the top bin
	num    : int
	         The number of bins
	"""
	def __init__(self, minval, maxval, num):
		ratio = np.exp( np.log(maxval/minval) / num )
		self.bin_start = minval * ratio**np.array(range(num))
		self.bin_end = self.bin_start * ratio
		self.bin_cent = 0.5*(self.bin_end + self.bin_start)
		self.num = num

### --- SPECTRAL TIMING ANALYSIS -----------------------------------------------

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

	Constructor: pylag.Periodogram(lc=None, f=[], per=[], norm=True)

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
	norm : boolean, optional (default=True)
	       If True, the calculated periodogram is normalised to be consistent
		   with the PSD (this only takes effect if the periodogram is calculated
		   from an input light curve)
	"""
	def __init__(self, lc=None, f=[], per=[], norm=True):
		if(lc != None):
			if not isinstance(lc, LightCurve):
				raise ValueError("pyLag CrossSpectrum ERROR: Can only compute cross spectrum between two LightCurve objects")

			self.freq, self.periodogram = self.Calculate(lc, norm)

		else:
			self.freq = np.array(f)
			self.periodogram = np.array(per)

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

		return Periodogram(f=bins.bin_cent, per=bins.Bin(self.freq, self.periodogram))

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
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

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
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

		Return Values
		-------------
		range_points : list
		               List of periodogram points (complex) in the frequency range

		"""
		return [p for f,p in zip(self.freq,self.periodogram) if f>=fmin and f<fmax]


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
	lc1  : LightCurve, optional (default=None)
	       pyLag LightCurve object for the primary or hard band (complex
		   conjugated during cross spectrum calculation)
	lc2  : LightCurve, optional (default=None)
	       pyLag LightCurve object for the reference or soft band
	f    : ndarray or list, optional (default=[])
	       If no light curve is specified, the sample frequency array can be
		   manually initialised using this array (used if storing the result
		   from an external calculation)
	cs   : ndarray or list, optional (default=[])
	       If no light curve is specified, the cross spectrum can be manually
		   initialised using this array (used if storing the result from an
		   external calculation)
	norm : boolean, optional (default=True)
	       If True, the calculated cross spectrum is normalised to be consistent
		   with the PSD normalisation (this only takes effect if the cross
		   spectrum is calculated from input light curves)
	"""
	def __init__(self, lc1=None, lc2=None, f=[], cs=[], norm=True):
		if(lc1 != None and lc2 != None):
			if not (isinstance(lc1, LightCurve) and isinstance(lc2, LightCurve)):
				raise ValueError("pyLag CrossSpectrum ERROR: Can only compute cross spectrum between two LightCurve objects")

			if(lc1 != lc2):
				raise AssertionError("pyLag CrossSpectrum ERROR: Light curves must be the same length and have same time binning to compute cross spectrum")

			self.freq, self.crossft = self.Calculate(lc1, lc2, norm)

		else:
			self.freq = np.array(f)
			self.crossft = np.array(cs)

	def Calculate(self, lc1, lc2, norm=True):
		"""
		f, crossft = pylag.CrossSpectrum.Calculate(lc1, lc2, norm=True)

		Calculate the cross spectrum from a pair of light curves and store it in
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

		Return Values
		-------------
		f       : ndarray
		          numpy array containing the sample frequencies at which the
				  cross spectrum is evaluated
		crossft : ndarray
		          numpy array containing the (complex) cross spectrum at each
				  frequency
		"""
		if norm:
			crossnorm = 2*lc1.dt / (lc1.Mean() * lc2.Mean() * lc1.length)
		else:
			crossnorm = 1

		f1, ft1 = lc1.FT()
		f2, ft2 = lc2.FT()

		crossft = crossnorm * np.conj(ft1) * ft2
		return f1, crossft

	def Bin(self, bins):
		"""
		csbin = pylag.CrossSpectrum.Bin(bins)

		Bin the cross spectrum by frequency using a Binning object then return
		the binned spectrum as a new CrossSpectrum object

		Arguments
		---------
		bins : Binning
			   pyLag Binning object to perform the Binning

		Return Values
		-------------
		csbin : CrossSpectrum
		        pyLag CrossSpectrum object storing the newly binned spectrum

		"""
		if not isinstance(bins, Binning):
			raise ValueError("pyLag CrossSpectrum bin ERROR: Expected a Binning object")

		return CrossSpectrum(f=bins.bin_cent, cs=bins.Bin(self.freq, self.crossft))

	def BinPoints(self, bins):
		"""
		bin_points = pylag.CrossSpectrum.BinPoints(bins)

		Bin the cross spectrum by frequency using a Binning object then return
		the list of data points that fall in each frequency bin.

		Arguments
		---------
		bins : Binning
			   pyLag Binning object to perform the Binning

		Return Values
		-------------
		bin_points : list
		             List of data point values (complex) that fall into each
					 frequency bin

		"""
		if not isinstance(bins, Binning):
			raise ValueError("pyLag CrossSpectrum bin ERROR: Expected a Binning object")

		return bins.BinPoints(self.freq, self.crossft)

	def FreqAverage(self, fmin, fmax):
		"""
		csavg = pylag.CrossSpectrum.FreqAverage(fmin, fmax)

		Calculate the average value of the cross spectrum over a specified
		frequency interval.

		Arguments
		---------
		fmin : float
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

		Return Values
		-------------
		csavg : complex
		        The average value of the cross spectrum over the frequency range

		"""
		return np.mean(  [c for f,c in zip(self.freq,self.crossft) if f>=fmin and f<fmax] )

	def FreqRangePoints(self, fmin, fmax):
		"""
		range_points = pylag.CrossSpectrum.FreqRangePoints(fmin, fmax)

		Return the list of cross spectrum points that fall in a specified
		frequency interval.

		Arguments
		---------
		fmin : float
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

		Return Values
		-------------
		range_points : list
		               List of cross spectrum points (complex) in the frequency
					   range

		"""
		return [c for f,c in zip(self.freq,self.crossft) if f>=fmin and f<fmax]

	def LagSpectrum(self):
		"""
		freq, lag = pylag.CrossSpectrum.LagSpectrum()

		Return the lag/frequency spectrum: The time lag between correlated
		variability in the two light curves as a function of Fourier frequency

		Return Values
		-------------
		freq : ndarray
		       numpy array containing the sample frequencies
		lag  : ndarray
		       numpy array containing the time lag (in seconds) at each sample
			   frequency

		"""
		lag = np.angle(self.crossft) / (2*np.pi*self.freq)
		return self.freq, lag

	def LagAverage(self, fmin, fmax):
		"""
		lagavg = pylag.CrossSpectrum.FreqAverage(fmin, fmax)

		Calculate the average value of the time lag over a specified frequency
		interval.

		Arguments
		---------
		fmin : float
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

		Return Values
		-------------
		lagavg : float
				 The average value of the time lag over the frequency range
				 (in seconds)

		"""
		avgcross = self.FreqAverage(fmin, fmax)
		lag = np.angle(avgcross) / (2*np.pi*np.mean([fmin, fmax]))
		return lag


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

	Constructor: pylag.CrossSpectrum(lc1=None, lc2=None, f=[], cs=[], norm=True)

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
		   Lower bound of freuency range
	fmin : float
		   Upper bound of freuency range
	"""
	def __init__(self, lc1=None, lc2=None, bins=None, fmin=None, fmax=None, bkg1=0., bkg2=0.):
		self.bkg1 = bkg1
		self.bkg2 = bkg2

		self.coh = np.array([])
		self.num_freq = np.array([])

		self.bins = bins

		if(bins != None):
			self.freq = bins.bin_cent
		elif(fmin>0 and fmax>0):
			self.freq = np.mean([fmin, fmax])

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

		# if we're passed lists of light curves, get the stacked cross spectrum
		# and periodograms and count the number of sample frequencies across all
		# the light curves
		elif(isinstance(lc1, list) and isinstance(lc2, list)):
			self.cross_spec = StackedCrossSpectrum(lc1, lc2, bins)
			self.per1 = StackedPeriodogram(lc1).periodogram
			self.per2 = StackedPeriodogram(lc2).periodogram
			if(bins != None):
				self.num_freq = np.zeros(bins.num)

				for lc1 in lc1_list:
					self.num_freq += lc1.BinNumFreq(bins)
			elif(fmin>0 and fmax>0):
				self.num_freq = 0
				for lc1 in lc1_list:
					self.num_freq += lc1.NumFreqInRange(fmin, fmax)

		self.coh = self.Calculate(bins, fmin, fmax)

	def Calculate(self, bins=None, fmin=None, fmax=None):
		"""
		pylag.Coherence.Calculate(bins=None, fmin=None, fmax=None)

		Calculate the coherence either in each bin or over a specified frequency
		range. The result is stored in the coh member variable either as a
		numpy array if calculated over separate bins or as a single float value
		when calculating for a frequency range.

		Arguments
		---------
		bins : Binning, optional (default=None)
			   pyLag Binning object specifying the bins in which coherence is to
			   be calculated
		fmin : float
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

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

		pnoise1 = 2*(lc1.Mean() + self.bkg1) / lc1.Mean()**2
		pnoise2 = 2*(lc2.Mean() + self.bkg2) / lc2.Mean()**2

		nbias = (pnoise2*(per1 - pnoise1) + pnoise1*(per2 - pnoise2) + pnoise1*pnoise2) / self.num_freq

		coh = np.abs(cross_spec)**2 / (per1 * per2)
		return coh

	def PhaseError(self):
		return np.sqrt( (1 - self.coh) / (2 * coh * self.num_freq) )


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
	def __init__(self, lc_list, bins):
		self.periodograms = []
		for lc1 in zip(lc1_list, lc2_list):
			self.periodograms.append( Periodogram(lc1) )

		self.bins = bins

		if(bins != None):
			self.freq = bins.bin_cent
			self.StackBinnedPeriodogram()

	def StackBinnedPeriodogram(self):
		"""
		pylag.StackedPeriodogram.StackBinnedPeriodogram()

		Calculates the average periodogram in each frequency bin. The final
		periodogram in each bin is the average over all of the individual
		frequency points from all of the light curves that fall into that bin.
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
		for freq_points in per_points:
			per.append( np.mean(freq_points) )

		self.periodogram = np.array(per)

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
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

		Return Values
		-------------
		per_avg : complex
		          The average value of the cross spectrum over the frequency range

		"""
		per_points = []
		for per in self.periodograms:
			per_points += per.FreqRangePoints(fmin, fmax)

		return np.mean(per_points)


class StackedCrossSpectrum(CrossSpectrum):
	"""
	pylag.StackedCrossSpectrum(CrossSpectrum)

	Calculate the average cross spectrum from multiple pairs of light curves
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
	def __init__(self, lc1_list, lc2_list, bins):
		self.cross_spectra = []
		for lc1, lc2 in zip(lc1_list, lc2_list):
			self.cross_spectra.append( CrossSpectrum(lc1, lc2) )

		self.bins = bins

		if(bins != None):
			self.freq = bins.bin_cent
			self.StackBinnedCrossSpectrum()

	def StackBinnedCrossSpectrum(self):
		"""
		pylag.StackedCrossSpectrum.StackBinnedCrossSpectrum()

		Calculates the average cross spectrum in each frequency bin. The final
		cross spectrum in each bin is the average over all of the individual
		frequency points from all of the light curves that fall into that bin.
		"""
		cross_spec_points = []
		for b in self.bins.bin_cent:
			cross_spec_points.append([])

		for cs in self.cross_spectra:
			this_cross_spec = cs.BinPoints(self.bins)

			# add the individual frequency points for this cross spectrum (for each
			# frequency bin) into the accumulated lists for each frequency bin
			for i, points in enumerate(this_cross_spec):
				cross_spec_points[i] += points

		# now take the mean of all the points that landed in each bin
		cross_spec = []
		for freq_points in cross_spec_points:
			cross_spec.append( np.mean(freq_points) )

		self.crossft = np.array(cross_spec)

	def FreqAverage(self, fmin, fmax):
		"""
		csavg = pylag.CrossSpectrum.FreqAverage(fmin, fmax)

		Calculate the average value of the cross spectrum over a specified
		frequency interval. The final cross spectrum is the average over all of
		the individual frequency points from all of the light curves that fall
		into the range.

		Arguments
		---------
		fmin : float
			   Lower bound of freuency range
		fmin : float
			   Upper bound of freuency range

		Return Values
		-------------
		csavg : complex
		        The average value of the cross spectrum over the frequency range

		"""
		cross_spec_points = []
		for cs in self.cross_spectra:
			cross_spec_points += cs.FreqRangePoints(fmin, fmax)

		return np.mean(cross_spec_points)







def MultiCrossSpec(lc1_list, lc2_list, bins, weighted_avg=True):
	#
	# Compute the averaged binned cross spectrum across many light curves
	# Returns the result in a CrossSpectrum object
	#
	cross_spec = np.array([])
	freq = bins.bin_cent

	sumweight = 0

	for lc1, lc2 in zip(lc1_list, lc2_list):
		this_cross_spec = CrossSpectrum(lc1, lc2).Bin(bins).crossft

		if weighted_avg:
			weight = lc1.length
		else:
			weight = 1
		sumweight += weight

		if(len(cross_spec)==0):
			cross_spec = weight*this_cross_spec
		else:
			cross_spec = np.vstack([cross_spec, weight*this_cross_spec])

	cross_spec = np.sum(cross_spec, axis=0) / sumweight
	return CrossSpectrum(f=freq, cs=cross_spec)




def AveragePeriodogram(lc_list, bins):
	"""
	avg_per = pylag.AveragePeriodogram(lc_list, bins)

	Calculate the average periodogram over multiple light curves using some
	frequency binning.

	The periodogram is calculated for of the light curves in turn, then	the data
	points are sorted into bins. The final periodogram in each bin is the average
	over all of the individual frequency points from all of the	light curves that
	fall into that bin.

	Arguments
	---------
	lc_list : list (of LightCurve objects)
	          List containing the pyLag LightCurve objects
	bins    : pyLag Binning object specifying the binning
	"""
	per_points = []
	for b in bins.bin_cent:
		per_points.append([])

	for lc in lc_list:
		this_per = Periodogram(lc, norm=True).BinPoints(bins)

		# add the individual frequency points for this cross spectrum (for each
		# frequency bin) into the accumulated lists for each frequency bin
		for i, points in enumerate(this_per):
			per_points[i] += points

	# now take the mean of all the points that landed in each bin
	per = []
	for freq_points in per_points:
		per.append( np.mean(freq_points) )

	return Periodogram(f=bins.bin_cent, per=per)


def AverageCrossSpectrum(lc1_list, lc2_list, bins):
	"""
	avg_cross = pylag.AverageCrossSpectrum(lc1_list, lc2_list, bins)

	Calculate the average cross spectrum from multiple pairs of light curves
	with some frequency binning.

	In the cross spectrum calculation, the complex conjugate is taken of the lc1
	DFT. This results in a positive lag denoting lc1 lagging BEHIND lc2. Thus, in
	order to adopt the common sign convention, lc1 is the HARD band light curve
	and lc2 is the SOFT band or REFERENCE light curve.

	The cross spectrum is calculated for each pair of light curves in turn, then
	the data points are sorted into bins. The final cross spectrum in each bin
	is the average over all of the individual frequency points from all of the
	light curves that fall into that bin.

	Arguments
	---------
	lc1_list : list (of LightCurve objects)
	           List containing the pyLag LightCurve objects for the subject or
			   hard band.
	lc2_list : list (of LightCurve objects)
			   List containing the pyLag LightCurve objects for the reference or
			   soft band.
	bins     : pyLag Binning object specifying the binning
	"""
	cross_spec_points = []
	for b in bins.bin_cent:
		cross_spec_points.append([])

	for lc1, lc2 in zip(lc1_list, lc2_list):
		this_cross_spec = CrossSpectrum(lc1, lc2, norm=True).BinPoints(bins)

		# add the individual frequency points for this cross spectrum (for each
		# frequency bin) into the accumulated lists for each frequency bin
		for i, points in enumerate(this_cross_spec):
			cross_spec_points[i] += points

	# now take the mean of all the points that landed in each bin
	cross_spec = []
	for freq_points in cross_spec_points:
		cross_spec.append( np.mean(freq_points) )

	return CrossSpectrum(f=bins.bin_cent, cs=cross_spec)
