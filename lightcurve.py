"""
pylag.lightcurve

Provides pyLag class for reading and manipulating light curves

Classes
-------
LightCurve : Read, store and manage an X-ray light curve

v1.0 09/03/2017 - D.R. Wilkins
"""
import numpy as np
import scipy.fftpack
import glob
import pyfits

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
	trim        : boolean, optional (default=False)
	              If true, after reading a light curve that begins or ends with
				  a series of zero points, these leading and trailing zeros will
				  be cut off

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
	def __init__(self, filename=None, t=[], r=[], e=[], interp_gaps=False, zero_nan=True, trim=False):
		self.time = np.array(t)
		if(len(r)>0):
			self.rate = np.array(r)
		else:
			self.rate = np.zeros(len(t))
		if(len(e)>0):
			self.error = np.array(e)
		else:
			self.error = np.zeros(len(t))

		if(filename != None):
			self.filename = filename
			self.ReadFITS(filename)

		self.dt = self.time[1] - self.time[0]
		self.length = len(self.rate)

		if interp_gaps:
			self.InterpGaps()
		if zero_nan:
			self.ZeroNaN()
		if trim:
			self.Trim()

		# pointers to the energy and lag arrays to specify the axes for Plotter
		self.xdata = self.time
		self.xerror = None
		self.ydata = self.rate
		self.yerror = self.error
		self.xlabel = 'Time / s'
		self.ylabel = 'Count Rate / ct s$^{-1}$'
		self.xscale = 'linear'
		self.yscale = 'linear'

	def ReadFITS(self, filename, byte_swap=True):
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

			if byte_swap:
				self.ByteSwap()

			self.dt = self.time[1] - self.time[0]

			fitsfile.close()

		except:
			raise AssertionError("pyLag LightCurve ERROR: Could not read light curve from FITS file")

	def ByteSwap(self):
		"""
		pylag.LightCurve.ByteSwap()

		Swap the byte order of the time, rate and error arrays

		This is necessary when reading from a FITS file (big endian) then you
		want to use the FFT functions in scipy.fftpack since these only supported
		little endian (and pyfits preserves the endianness read from the file)
		"""
		self.time = self.time.byteswap().newbyteorder('<')
		self.rate = self.rate.byteswap().newbyteorder('<')
		self.error = self.error.byteswap().newbyteorder('<')

	def Trim(self):
		"""
		pylag.LightCurve.Trim()

		If the light curve begins or ends with a series of zero data points (e.g.
		if it is created with a time filter), cut these points off
		"""
		nonzero = self.rate.nonzero()
		first_index = nonzero[0][0]
		last_index = nonzero[0][-1]
		self.time = self.time[first_index:last_index+1]
		self.rate = self.rate[first_index:last_index+1]
		self.error = self.error[first_index:last_index+1]

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
		# ft = np.fft.fft(self.rate)
		# freq = np.fft.fftfreq(self.length, d=self.dt)

		ft = scipy.fftpack.fft(self.rate)
		freq = scipy.fftpack.fftfreq(self.length, d=self.dt)

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
			   Lower bound of frequency range
		fmin : float
			   Upper bound of frequency range

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

	def __iadd__(self, other):
		"""
		Overloaded += operator to add another light curve to this one, storing
		the result in place

		ERROR columns from the two light curves are summed in quadrature.
		"""
		if isinstance(other, LightCurve):
			if(len(self.rate) != len(other.rate)):
				raise AssertionError("pyLag LightCurve ERROR: Cannot add light curves of different lengths")
				return
			# sum the count rate
			self.rate = self.rate + other.rate
			# sum the errors in quadrature
			self.err = np.sqrt( self.error**2 + other.error**2 )

		else:
			return NotImplemented

	def __sub__(self, other):
		"""
		Overloaded - operator to subtract a LightCurve object from this one and
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

	def __isub__(self, other):
		"""
		Overloaded -= operator to subtract a LightCurve object from this one in
		place (e.g. for background subtraction).

		ERROR columns from the two light curves are summed in quadrature.
		"""
		if isinstance(other, LightCurve):
			if(len(self.rate) != len(other.rate)):
				raise AssertionError("pyLag LightCurve ERROR: Cannot subtract light curves of different lengths")
				return
			# subtract the count rate
			self.rate = self.rate - other.rate
			# sum the errors in quadrature
			self.error = np.sqrt( self.error**2 + other.error**2 )

		else:
			return NotImplemented

	def __eq__(self, other):
		"""
		Overloaded == operator to check light curve lengths and time binning are
		consistent to see if they can be added/subtracted/combined.
		"""
		if isinstance(other, LightCurve):
			return (self.length == other.length and self.dt == other.dt)
		else:
			return NotImplemented

	def __ne__(self, other):
		"""
		Overloaded != operator to check light curve lengths and time binning are
		inconsistent to see if they can't be added/subtracted/combined.
		"""
		if isinstance(other, LightCurve):
			return not (self.length == other.length and self.dt == other.dt)
		elif other==None:
			# need to provide this so that constructors can be passed None in place
			# of a light curve for dummy initialisation
			return True
		else:
			return NotImplemented

	def __len__(self):
		"""
		Overloaded operator to provide len() functionality to get number of bins.
		"""
		return len(self.rate)

	def __getitem__(self, index):
		"""
		Overloaded operator to access count rate via [] operator
		"""
		return self.rate[index]

	def __getslice__(self, start, end):
		"""
		Overloaded operator to extract a portion of the light curve using
		[start:end] operator and return as a new LightCurve object
		"""
		return LightCurve(t=self.time[start:end], r=self.rate[start:end], e=self.error[start:end])


def get_lclist(searchstr, **kwargs):
	"""
	pylag.get_lclist(searchstr, **kwargs)

	Search the filesystem for light curves, read them into LightCurve objects,
	then return them as a list.

	Light curves are sorted alphabetically by filename, thus so long as a common
	prefix convention is adopted between observation segments, lists returned
	with different criteria (e.g. energy band) will line up and can be used
	immediately for stacking analysis.

	Arguments
	---------
	searchstr : string
				Wildcard string (glob) with which to search the filesystem for
	            files
	**kwargs  : Keyword arguments to be passed on to the constructor for each
	            LightCurve object

	Return Values
	-------------
	lclist : list of LightCurve objects
	"""
	lcfiles = sorted( glob.glob(searchstr) )
	lclist = []
	for lc in lcfiles:
		lclist.append( LightCurve(lc, **kwargs) )
	return lclist


def StackedMeanCountRate(lclist):
	"""
	mean_rate = pylag.StackedMeanCountRate(lclist)

	Return the mean count rate over a list of light curves

	Arguments
	---------
	lclist : list of LightCurve objects

	Return Values
	-------------
	mean_rate : float
				The mean count rate
	"""
	rate_points = []
	for lc in lclist:
		rate_points += lc.rate.tolist()
	return np.mean(rate_points)
