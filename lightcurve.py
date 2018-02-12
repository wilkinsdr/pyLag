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
import re
try:
    import astropy.io.fits as pyfits
except:
    import pyfits
from scipy.stats import binned_statistic


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
    max_gap     : int (optional, default=0)
                  If >0, gaps longer than max_gap time bins will not be interpolated
                  over

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

    def __init__(self, filename=None, t=[], r=[], e=[], interp_gaps=False, zero_nan=True, trim=False, max_gap=0):
        self.time = np.array(t)
        if len(r) > 0:
            self.rate = np.array(r)
        else:
            self.rate = np.zeros(len(t))
        if len(e) > 0:
            self.error = np.array(e)
        else:
            self.error = np.zeros(len(t))

        if filename is not None:
            self.filename = filename
            self.read_fits(filename)

        if len(self.time) > 1:
            self.dt = self.time[1] - self.time[0]
        self.length = len(self.rate)

        if interp_gaps:
            self._interp_gaps(max_gap)
        if zero_nan:
            self._zeronan()
        if trim:
            self.trim()

    def read_fits(self, filename, byte_swap=True):
        """
        pylag.LightCurve.read_fits(filename)

        Read the light curve from an OGIP standard FITS file.

        Paremeters
        ----------
        filename : string
                   The path of the FITS file from which the light curve is loaded
        """
        try:
            print("Reading light curve from " + filename)
            fitsfile = pyfits.open(filename)
            tabdata = fitsfile['RATE'].data

            self.time = np.array(tabdata['TIME'])
            self.rate = np.array(tabdata['RATE'])
            self.error = np.array(tabdata['ERROR'])

            if byte_swap:
                self._byteswap()

            self.dt = self.time[1] - self.time[0]

            fitsfile.close()

        except:
            raise AssertionError("pyLag LightCurve ERROR: Could not read light curve from FITS file")

    def _byteswap(self):
        """
        pylag.LightCurve._byteswap()

        Swap the byte order of the time, rate and error arrays

        This is necessary when reading from a FITS file (big endian) then you
        want to use the FFT functions in scipy.fftpack since these only supported
        little endian (and pyfits preserves the endianness read from the file)
        """
        self.time = self.time.byteswap().newbyteorder('<')
        self.rate = self.rate.byteswap().newbyteorder('<')
        self.error = self.error.byteswap().newbyteorder('<')

    def trim(self):
        """
        pylag.LightCurve.trim()

        If the light curve begins or ends with a series of zero data points (e.g.
        if it is created with a time filter), cut these points off
        """
        nonzero = self.rate.nonzero()
        first_index = nonzero[0][0]
        last_index = nonzero[0][-1]
        self.time = self.time[first_index:last_index + 1]
        self.rate = self.rate[first_index:last_index + 1]
        self.error = self.error[first_index:last_index + 1]

    def zero_time(self):
        """
        pylag.LightCurve.zero_time()

        Shift the time axis of the light curve to start at zero. The modified
        time axis is stored	back in the original object.
        """
        self.time = self.time - self.time.min()

    def _interp_gaps(self, max_gap=0):
        """
        pylag.LightCurve._interp_gaps(max_gap=0)

        Interpolate over gaps within the light curve for fixing gaps left by GTI
        filters when performing timing analysis.

        The missing data points are filled in by linear interpolation between the
        start end end points of the gap and the patched light curve is stored
        back in the original object.

        Arguments
        ---------
        max_gap : int (optional, default=0)
                  If >0, gaps longer than max_gap time bins will not be interpolated
                  over
        """
        in_gap = False
        gap_count = 0
        max_gap = 0

        for i in range(len(self.rate)):
            if not in_gap:
                if np.isnan(self.rate[i]):
                    in_gap = True
                    gap_start = i - 1

            elif in_gap:
                if not np.isnan(self.rate[i]):
                    gap_end = i
                    in_gap = False

                    gap_length = gap_end - gap_start

                    if gap_length < max_gap or max_gap==0:
                        gap_count += 1
                        self.rate[gap_start:gap_end] = np.interp(self.time[gap_start:gap_end],
                                                                 [self.time[gap_start], self.time[gap_end]],
                                                                 [self.rate[gap_start], self.rate[gap_end]])

                        if gap_length > max_gap:
                            max_gap = gap_length

        print("Patched %d gaps" % gap_count)
        print("Longest gap was %d bins" % max_gap)

    def _zeronan(self):
        """
        pylag.LightCurve._zeronan()

        Replace all instances of nan with zero in the light curve. The modified
        light curve is stored back in the original object.
        """
        self.rate[np.isnan(self.rate)] = 0
        self.error[np.isnan(self.error)] = 0

    def time_segment(self, start, end):
        """
        lc = pylag.LightCurve.time_segment(start, end)

        Returns the Fourier transform (FFT) of the light curve.
        Only the positive frequencies are returned (N/2 points for light curve length N)

        Arguments
        ---------
        start : float
                The start time from which the extracted light curve begins
        end   : float
                The end time to which the extracted light curve runs

        Returns
        -------
        lc : LightCurve
             The extracted light curve segment as a new LightCurve object
        """
        if start < 0:
            start = self.time.min() + abs(start)
        if end < 0:
            end = self.time.max() - abs(end)
        this_t = np.array([t for t in self.time if start < t <= end])
        this_rate = np.array([r for t, r in zip(self.time, self.rate) if start < t <= end])
        this_error = np.array([e for t, e in zip(self.time, self.error) if start < t <= end])
        segment = LightCurve(t=this_t, r=this_rate, e=this_error)
        segment.__class__ = self.__class__
        return segment

    def split_segments(self, num_segments=1, segment_length=None, use_end=False):
        """
        segments = pylag.LightCurve.split_segments(num_segments=1, segment_length=None)

        Divides the light curve into equal time segments which are returned as a
        list of LightCurve objects. The segments are always of equal legnth. If
        the light curve does not divide into the specified segment length, the
        end of the light curve will not be included.

        Arguments
        ---------
        num_segments   : int, optional (default=1)
                         The number of segments to divide the light curve into
        segment_length : float, optional (default=None)
                         If set, the length of the light curve segments to be
                         created.
        use_end        : boolean, optional (default=False)
                         Sometimes the light curve will not divide exactly into
                         the desired number of segments. If true, in this case,
                         the list of segments will also include the partial
                         segment from the end of the light curve

        Returns
        -------
        segments : list of LightCurves
        """
        if segment_length is None:
            segment_length = (self.time.max() - self.time.min()) / float(num_segments)

        segments = []
        for tstart in np.arange(self.time.min(), self.time.max(), segment_length):
            if ((tstart + segment_length) <= self.time.max()) or use_end:
                segments.append(self.time_segment(tstart, tstart + segment_length))

        return segments

    def split_on_gaps(self, min_segment=0):
        """
        lclist = pylag.LightCurve.split_on_gaps(min_segment=0)

        Split the light curve on gaps into good segments

        Arguments
        ---------
        min_segment : int (optional, default=0)
                      the minimum length of good segment to be included in the output list

        Returns
        -------
        lclist : list of LightCurves
                 the good segments of the light curve
        """
        in_good_segment = False
        good_count = 0
        short_count = 0

        lclist = []

        for i in range(len(self.rate)):
            if not in_good_segment:
                if not np.isnan(self.rate[i]):
                    in_good_segment = True
                    good_start = i

            elif in_good_segment:
                if np.isnan(self.rate[i]):
                    good_end = i
                    in_good_segment = False

                    good_length = good_end - good_start

                    if good_length >= min_segment:
                        lclist.append(self[good_start:good_end])
                        good_count += 1
                    else:
                        short_count += 1

                # make sure we get the end of the light curve if it's good
                elif i==len(self.rate)-1:
                    print("got the end")
                    good_length = good_end - good_start
                    if good_length >= min_segment:
                        lclist.append(self[good_start:good_end])
                        good_count += 1
                    else:
                        short_count += 1

        print("Split light curve into  %d good segments" % good_count)
        if short_count > 0:
            print("%d segments too short" % short_count)
        return lclist

    def rebin(self, tbin):
        """
        rebin_lc = pylag.LightCurve.rebin(tbin)

        Rebin the light curve by summing together counts from the old bins into
        new larger bins and return the rebinned light curve as a new LightCurve
        object.

        Note that the new time bin size should be a multiple of the old for the
        best accuracy

        Arguments
        ---------
        tbin : float
               New time bin size

        Returns
        -------
        rebin_lc : LightCurve
                   The rebinned light curve
        """
        if tbin <= self.dt:
            raise ValueError("pylag LightCurve Rebin ERROR: Must rebin light curve into larger bins")
        if tbin % self.dt != 0:
            print("pylag LightCurve Rebin WARNING: New time binning is not a multiple of the old")

        time = np.arange(min(self.time), max(self.time), tbin)

        counts = []
        for bin_t in time:
            # note we're summing counts, not rate, as if the light curve had been
            # originally made from the event list with a larger time bin
            counts.append(
                np.sum([self.dt * r for t, r in zip(self.time, self.rate) if bin_t <= t < (bin_t + tbin)]))
            # if we have a partial bin, scale the counts to correct the exposure
            time_slice = [t for t in self.time if bin_t <= t < (bin_t + tbin)]
            if (max(time_slice) - min(time_slice)) < tbin:
                counts[-1] *= (float(tbin) / float(max(time_slice) - min(time_slice)))

        counts = np.array(counts)
        rate = counts / tbin
        # calculate the sqrt(N) error from the total counts
        err = rate * np.sqrt(counts) / counts

        binlc = LightCurve(t=time[:-1], r=rate, e=err)
        # make sure the returned object has the right class (if this is called from a derived class)
        binlc.__class__ = self.__class__
        return binlc

    def rebin2(self, tbin):
        """
        rebin_lc = pylag.LightCurve.rebin2(tbin)

        Rebin the light curve by summing together counts from the old bins into
        new larger bins and return the rebinned light curve as a new LightCurve
        object.

        This function should be faster than rebin. It uses the numpy digitize
        function.

        Note that the new time bin size should be a multiple of the old for the
        best accuracy

        Arguments
        ---------
        tbin : float
               New time bin size

        Returns
        -------
        rebin_lc : LightCurve
                   The rebinned light curve
        """
        if tbin <= self.dt:
            raise ValueError("pylag LightCurve Rebin ERROR: Must rebin light curve into larger bins")
        if tbin % self.dt != 0:
            print("pylag LightCurve Rebin WARNING: New time binning is not a multiple of the old")

        time = np.arange(min(self.time), max(self.time), tbin)

        # digitize returns an array stating which bin each element falls into
        digitized = np.digitize(self.time, time)
        counts = np.array([ np.sum(self.dt*self.rate[digitized==i]) for i in range(1,len(time))])
        rate = counts / tbin
        err = rate * np.sqrt(counts) / counts

        binlc = LightCurve(t=time[:-1], r=rate, e=err)
        # make sure the returned object has the right class (if this is called from a derived class)
        binlc.__class__ = self.__class__
        return binlc

    def rebin3(self, tbin):
        """
        rebin_lc = pylag.LightCurve.rebin3(tbin)

        Rebin the light curve by summing together counts from the old bins into
        new larger bins and return the rebinned light curve as a new LightCurve
        object.

        This function should be faster than rebin. It uses the scipy binned_statistic
        function.

        Note that the new time bin size should be a multiple of the old for the
        best accuracy.

        Arguments
        ---------
        tbin : float
               New time bin size

        Returns
        -------
        rebin_lc : LightCurve
                   The rebinned light curve
        """
        if tbin <= self.dt:
            raise ValueError("pylag LightCurve Rebin ERROR: Must rebin light curve into larger bins")
        if tbin % self.dt != 0:
            print("pylag LightCurve Rebin WARNING: New time binning is not a multiple of the old")

        time = np.arange(min(self.time), max(self.time)+tbin, tbin)
        counts = binned_statistic(self.time, self.dt*self.rate, statistic='sum', bins=time)[0]
        rate = counts / tbin
        err = rate * np.sqrt(counts) / counts

        binlc = LightCurve(t=time[:-1], r=rate, e=err)
        # make sure the returned object has the right class (if this is called from a derived class)
        binlc.__class__ = self.__class__
        return binlc

    def mean(self):
        """
        mean = pylag.LightCurve.mean()

        Returns the mean count rate over the light curve

        Return Values
        -------------
        mean : float
               The mean count rate
        """
        return np.mean(self.rate)

    def ft(self):
        """
        freq, ft = pylag.LightCurve.ft()

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

        return freq[:int(self.length / 2)], ft[:int(self.length / 2)]

    def bin_num_freq(self, bins):
        """
        numfreq = pylag.LightCurve.bin_num_freq(bins)

        Returns the number of sample frequencies that fall into bins specified
        in a pyLag Binning object

        Arguments
        ---------
        bins : Binning
               pyLag Binning object defining the bins into which sample frequencies
               are to be counted

        Returns
        -------
        numfreq : ndarray
                  The number of frequencies falling into each bin
        """
        freq = scipy.fftpack.fftfreq(self.length, d=self.dt)

        return bins.num_points_in_bins(freq)

    def num_freq_in_range(self, fmin, fmax):
        """
        num_freq = pylag.LightCurve.num_freq_in_range()

        Returns the number of sample frequencies that fall into bins specified
        in a pyLag Binning object

        Arguments
        ---------
        fmin : float
               Lower bound of frequency range
        fmax : float
               Upper bound of frequency range

        Returns
        -------
        numfreq : ndarray
                  The number of frequencies falling into each bin
        """
        freq = scipy.fftpack.fftfreq(self.length, d=self.dt)

        return len([f for f in freq if fmin <= f < fmax])

    def concatenate(self, other):
        """
        Concatenate the data points from another light curve onto the end of this
        one and return as a new LightCurve object
        """
        if isinstance(other, LightCurve):
            other = [other]
        if not isinstance(other, list):
            raise ValueError("pylag LightCurve Concatenate ERROR: Expected a list of LightCurves to append")

        newtime = np.concatenate([self.time] + [lc.time for lc in other])
        newrate = np.concatenate([self.rate] + [lc.rate for lc in other])
        newerr = np.concatenate([self.error] + [lc.error for lc in other])

        newlc = LightCurve(t=newtime, r=newrate, e=newerr)
        newlc.__class__ = self.__class__
        return newlc

    def __add__(self, other):
        """
        Overloaded + operator to add two light curves together and return the
        result in a new LightCurve object (e.g. to sum observations from 2
        detectors).

        ERROR columns from the two light curves are summed in quadrature.
        """
        if isinstance(other, LightCurve):
            if len(self.rate) != len(other.rate):
                raise AssertionError("pyLag LightCurve ERROR: Cannot add light curves of different lengths, %d and %d" % (len(self.rate), len(other.rate)))
            # sum the count rate
            newrate = self.rate + other.rate
            # sum the errors in quadrature
            newerr = np.sqrt(self.error ** 2 + other.error ** 2)
            # construct a new LightCurve with the result and make sure it has the right class
            # (if calling from a derived class)
            newlc = LightCurve(t=self.time, r=newrate, e=newerr)
            newlc.__class__ = self.__class__
            return newlc

        else:
            return NotImplemented

    def __iadd__(self, other):
        """
        Overloaded += operator to add another light curve to this one, storing
        the result in place

        ERROR columns from the two light curves are summed in quadrature.
        """
        if isinstance(other, LightCurve):
            if len(self.rate) != len(other.rate):
                raise AssertionError("pyLag LightCurve ERROR: Cannot add light curves of different lengths")
            # sum the count rate
            self.rate = self.rate + other.rate
            # sum the errors in quadrature
            self.err = np.sqrt(self.error ** 2 + other.error ** 2)

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
            if len(self.rate) != len(other.rate):
                raise AssertionError("pyLag LightCurve ERROR: Cannot subtract light curves of different lengths")
            # subtract the count rate
            newrate = self.rate - other.rate
            # sum the errors in quadrature
            newerr = np.sqrt(self.error ** 2 + other.error ** 2)
            # construct a new LightCurve with the result and make sure it has the right class
            # (if calling from a derived class)
            newlc = LightCurve(t=self.time, r=newrate, e=newerr)
            newlc.__class__ = self.__class__
            return newlc

        else:
            return NotImplemented

    def __isub__(self, other):
        """
        Overloaded -= operator to subtract a LightCurve object from this one in
        place (e.g. for background subtraction).

        ERROR columns from the two light curves are summed in quadrature.
        """
        if isinstance(other, LightCurve):
            if len(self.rate) != len(other.rate):
                raise AssertionError("pyLag LightCurve ERROR: Cannot subtract light curves of different lengths")
            # subtract the count rate
            self.rate = self.rate - other.rate
            # sum the errors in quadrature
            self.error = np.sqrt(self.error ** 2 + other.error ** 2)

        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Overloaded / operator to divide this LightCurve object by another and
        return the result in a new LightCurve object.

        Fractional errors from the two light curves are summed in quadrature.
        """
        if isinstance(other, (float, int)):
            newrate = self.rate * other
            # add the fractional errors in quadrature
            newerr = newrate * (self.error / self.rate)
            # construct a new LightCurve with the result and make sure it has the right class
            # (if calling from a derived class)
            newlc = LightCurve(t=self.time, r=newrate, e=newerr)
            newlc.__class__ = self.__class__
            return newlc

        else:
            return NotImplemented

    def __div__(self, other):
        """
        Overloaded / operator to divide this LightCurve object by another and
        return the result in a new LightCurve object.

        Fractional errors from the two light curves are summed in quadrature.
        """
        if isinstance(other, LightCurve):
            if len(self.rate) != len(other.rate):
                raise AssertionError("pyLag LightCurve ERROR: Cannot divide light curves of different lengths")
            # subtract the count rate
            newrate = self.rate / other.rate
            # add the fractional errors in quadrature
            newerr = newrate * np.sqrt((self.error / self.rate) ** 2 + (other.error / other.rate) ** 2)
            # construct a new LightCurve with the result and make sure it has the right class
            # (if calling from a derived class)
            newlc = LightCurve(t=self.time, r=newrate, e=newerr)
            newlc.__class__ = self.__class__
            return newlc

        elif isinstance(other, (float, int)):
            newrate = self.rate / other
            # add the fractional errors in quadrature
            newerr = newrate * (self.error / self.rate)
            # construct a new LightCurve with the result and make sure it has the right class
            # (if calling from a derived class)
            newlc = LightCurve(t=self.time, r=newrate, e=newerr)
            newlc.__class__ = self.__class__
            return newlc

        else:
            return NotImplemented

    def __eq__(self, other):
        """
        Overloaded == operator to check light curve lengths and time binning are
        consistent to see if they can be added/subtracted/combined.

        Allow a small fractional error in dt to allow for accuracy issues
        """
        if isinstance(other, LightCurve):
            return self.length == other.length and abs(self.dt - other.dt) < 0.001*self.dt
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Overloaded != operator to check light curve lengths and time binning are
        inconsistent to see if they can't be added/subtracted/combined.
        """
        if isinstance(other, LightCurve):
            return not (self.length == other.length and abs(self.dt - other.dt) < 0.001*self.dt)
        elif other is None:
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
        if isinstance(index, slice):
            lcslice = LightCurve(t=self.time[index], r=self.rate[index], e=self.error[index])
            lcslice.__class__ = self.__class__
            return lcslice
        else:
            return self.rate[index]

    def __getslice__(self, start, end):
        """
        Overloaded operator to extract a portion of the light curve using
        [start:end] operator and return as a new LightCurve object
        """
        slice = LightCurve(t=self.time[start:end], r=self.rate[start:end], e=self.error[start:end])
        slice.__class__ = self.__class__
        return slice

    def _getplotdata(self):
        return self.time, (self.rate, self.error)

    def _getplotaxes(self):
        return 'Time / s', 'linear', 'Count Rate / ct s$^{-1}$', 'linear'


# --- Utility functions --------------------------------------------------------

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

    Returns
    -------
    lclist : list of LightCurve objects
    """
    lcfiles = sorted(glob.glob(searchstr))

    if len(lcfiles) < 1:
        raise AssertionError("pylag get_lclist ERROR: Could not find light curve files")

    lclist = []
    for lc in lcfiles:
        lclist.append(LightCurve(lc, **kwargs))
    return lclist


def stacked_mean_count_rate(lclist):
    """
    mean_rate = pylag.stacked_mean_count_rate(lclist)

    Return the mean count rate over a list of light curves

    Arguments
    ---------
    lclist : list of LightCurve objects

    Returns
    -------
    mean_rate : float
                The mean count rate
    """
    rate_points = []
    for lc in lclist:
        rate_points += lc.rate.tolist()
    return np.mean(rate_points)


def extract_sim_lightcurves(lc1, lc2):
    """
    out_lc1, out_lc2 = pylag.extract_sim_lightcurves(lc1, lc2)

    Returns the simultaneous portions of two light curves; i.e. the intersection
    of the two time series

    Arguments
    ---------
    lc1 : LightCurve
          First input light curve
    lc2 : LightCurve
          Second input light curve

    Returns
    -------
    out_lc1 : LightCurve
              LightCurve object containing the simultaneous portion of the first
              light curve
    out_lc2 : LightCurve
              LightCurve object containing the simultaneous portion of the second
              light curve
    """
    if abs(lc1.dt - lc2.dt) > 1.e-10:
        raise AssertionError('pylag extract_sim_lightcurves ERROR: Light curves must have same time spacing')

    # find the latest of the start times between the two light curves and the
    # earliest end time to get the time series covered by both
    start = max([lc1.time.min(), lc2.time.min()])
    end = min([lc1.time.max(), lc2.time.max()])

    print("Extracting simultaneous light curve portion from t=%g to %g" % (start, end))
    print("Simultaneous portion length = %g" % (end - start))

    # extract the portion of eaach light curve from this range of times
    time1 = np.array([t for t in lc1.time if start <= t <= end])
    rate1 = np.array([r for t, r in zip(lc1.time, lc1.rate) if start <= t <= end])
    err1 = np.array([e for t, e in zip(lc1.time, lc1.error) if start <= t <= end])

    time2 = np.array([t for t in lc2.time if start <= t <= end])
    rate2 = np.array([r for t, r in zip(lc2.time, lc2.rate) if start <= t <= end])
    err2 = np.array([e for t, e in zip(lc2.time, lc2.error) if start <= t <= end])

    # check that we actually have an overlapping section!
    if len(rate1) == 0 or len(rate2) == 0:
        raise AssertionError('pylag extract_sim_lightcurves ERROR: Light curves have no simultaneous part')

    # sometimes rounding causes there to be one more bin in one light curve than
    # the other so take off the extra bin, but make sure it's only 1 bin difference!
    if abs(len(rate1) - len(rate2)) > 1:
        raise AssertionError('pylag extract_sim_lightcurves ERROR: Light curves differ in length by more than one bin')
    if len(rate1) > len(rate2):
        if abs(time1[0] - time2[0]) < abs(time1[-1] - time2[-1]):
            # if the start times match better than the end times, knock off the last bin
            time1 = time1[:-1]
            rate1 = rate1[:-1]
            err1 = err1[:-1]
        else:
            # otherwise knock the first time bin off
            time1 = time1[1:]
            rate1 = rate1[1:]
            err1 = err1[1:]
    if len(rate1) < len(rate2):
        if abs(time1[0] - time2[0]) < abs(time1[-1] - time2[-1]):
            # if the start times match better than the end times, knock off the last bin
            time2 = time2[:-1]
            rate2 = rate2[:-1]
            err2 = err2[:-1]
        else:
            # otherwise knock the first time bin off
            time2 = time2[1:]
            rate2 = rate2[1:]
            err2 = err2[1:]

    out_lc1 = LightCurve(t=time1, r=rate1, e=err1)
    out_lc2 = LightCurve(t=time2, r=rate2, e=err2)

    return out_lc1, out_lc2


def sum_sim_lightcurves(lc1, lc2):
    lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
    return lc1s + lc2s


class EnergyLCList(object):
    def __init__(self, searchstr=None, lcfiles=None, enmin=None, enmax=None, lclist=None, **kwargs):
        if lclist is not None and enmin is not None and enmax is not None:
            self.lclist = lclist
            self.enmin = np.array(enmin)
            self.enmax = np.array(enmax)
        else:
            self.enmin, self.enmax, self.lclist = self.find_light_curves(searchstr, lcfiles)

        self.en = 0.5*(self.enmin + self.enmax)
        self.en_error = self.en - self.enmin

    @staticmethod
    def find_light_curves(searchstr, lcfiles=None, **kwargs):
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
        if lcfiles is None:
            lcfiles = sorted(glob.glob(searchstr))
        enlist = list(set([re.search('(en[0-9]+\-[0-9]+)', lc).group(0) for lc in lcfiles]))

        obsid_list = list(set([re.search('(.*?)_(tbin[0-9]+)_(en[0-9]+\-[0-9]+)', lc).group(1) for lc in lcfiles]))
        for o in obsid_list:
            if len([lc for lc in lcfiles if o in lc]) < len(enlist):
                raise AssertionError('%s does not have a complete set of energy bands' % o)

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

    def time_segment(self, start, end):
        """
        new_lclist = pylag.extract_lclist_time_segment(lclist, tstart, tend)

        Take a list of LightCurve objects or a list of lists of multiple light curve
        segments in each energy band (as used for a lag-energy or covariance spectrum)
        and return only the segment(s) within a	specified time interval
        """
        new_lclist = []

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    lcseg = lc.time_segment(start, end)
                    if len(lcseg) > 0:
                        new_lclist[-1].append(lcseg)

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                lcseg = lc.time_segment(start, end)
                if len(lcseg) > 0:
                    new_lclist.append(lcseg)
                else:
                    print(
                        "pylag extract_lclist_time_segment WARNING: One of the light curves does not cover this time segment. Check consistency!")

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)

    def segment(self, start, end):
        """
        new_lclist = pylag.extract_lclist_time_segment(lclist, tstart, tend)

        Take a list of LightCurve objects or a list of lists of multiple light curve
        segments in each energy band (as used for a lag-energy or covariance spectrum)
        and return only the segment(s) within a	specified time interval
        """
        new_lclist = []

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    lcseg = lc[start:end]
                    if len(lcseg) > 0:
                        new_lclist[-1].append(lcseg)

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                lcseg = lc[start:end]
                if len(lcseg) > 0:
                    new_lclist.append(lcseg)
                else:
                    print(
                        "pylag extract_lclist_time_segment WARNING: One of the light curves does not cover this time segment. Check consistency!")

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)

    def rebin(self, tbin):
        new_lclist = []

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    new_lclist[-1].append(lc.rebin3(tbin))

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                new_lclist.append(lc.rebin3(tbin))

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)

    def __getitem__(self, index):
        return self.lclist[index]

    def __getslice__(self, start, end):
        """
        new_lclist = pylag.extract_lclist_time_segment(lclist, tstart, tend)

        Take a list of LightCurve objects or a list of lists of multiple light curve
        segments in each energy band (as used for a lag-energy or covariance spectrum)
        and return only the segment(s) within a	specified time interval
        """
        new_lclist = []

        # this is a horrible hack and could cause some weirdness but it is
        # to deal with Python adding the length to a negative index
        if end < len(self):
            end -= len(self)

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    lcseg = lc[start:end]
                    if len(lcseg) > 0:
                        new_lclist[-1].append(lcseg)

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                lcseg = lc[start:end]
                if len(lcseg) > 0:
                    new_lclist.append(lcseg)
                else:
                    print(
                        "pylag extract_lclist_time_segment WARNING: One of the light curves does not cover this time segment. Check consistency!")

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)

    def __add__(self, other):
        if not isinstance(other, EnergyLCList):
            return NotImplemented

        if len(self.lclist) != len(other.lclist):
            raise AssertionError("EnergyLCList objects do not have same number of energy bands")

        lclist = []
        if isinstance(self.lclist[0], list):
            if not isinstance(other.lclist[0], list):
                raise AssertionError("EnergyLCList objects do not have the same dimension")
            if len(self.lclist[0]) != len(other.lclist[0]):
                raise AssertionError("EnergyLCList objects do not have the same number of segments in each energy band")

            for en_lclist1, en_lclist2 in zip(self.lclist, other.lclist):
                lclist.append([])
                for lc1, lc2 in zip(en_lclist1, en_lclist2):
                    lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
                    lclist[-1].append(lc1s + lc2s)
        else:
            for lc1, lc2 in zip(self.lclist, other.lclist):
                lc1s, lc2s = extract_lclist_time_segment(lc1, lc2)
                lclist.append(lc1s + lc2s)

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=lclist)

    def __sub__(self, other):
        if not isinstance(other, EnergyLCList):
            return NotImplemented

        if len(self.lclist) != len(other.lclist):
            raise AssertionError("EnergyLCList objects do not have same number of energy bands")

        lclist = []
        if isinstance(self.lclist[0], list):
            if not isinstance(other.lclist[0], list):
                raise AssertionError("EnergyLCList objects do not have the same dimension")
            if len(self.lclist[0]) != len(other.lclist[0]):
                raise AssertionError("EnergyLCList objects do not have the same number of segments in each energy band")

            for en_lclist1, en_lclist2 in zip(self.lclist, other.lclist):
                lclist.append([])
                for lc1, lc2 in zip(en_lclist1, en_lclist2):
                    lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
                    lclist[-1].append(lc1s - lc2s)
        else:
            for lc1, lc2 in zip(self.lclist, other.lclist):
                lc1s, lc2s = extract_lclist_time_segment(lc1, lc2)
                lclist.append(lc1s - lc2s)

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=lclist)

    def __len__(self):
        return len(self.lclist)


def sum_sim_lclists(lclist1, lclist2):
    lclist = []
    if isinstance(lclist1[0], list):
        for en_lclist1, en_lclist2 in zip(lclist1.lclist, lclist2.lclist):
            lclist.append([])
            for lc1, lc2 in zip(en_lclist1, en_lclist2):
                lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
                lclist[-1].append(lc1s + lc2s)
    else:
        for lc1, lc2 in zip(lclist1.lclist, lclist2.lclist):
            lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
            lclist.append(lc1s + lc2s)

    return EnergyLCList(enmin=lclist1.enmin, enmax=lclist2.enmax, lclist=lclist)


def extract_lclist_time_segment(lclist, tstart, tend):
    """
    new_lclist = pylag.extract_lclist_time_segment(lclist, tstart, tend)

    Take a list of LightCurve objects or a list of lists of multiple light curve
    segments in each energy band (as used for a lag-energy or covariance spectrum)
    and return only the segment(s) within a	specified time interval
    """
    new_lclist = []

    if isinstance(lclist[0], list):
        for en_lclist in lclist:
            new_lclist.append([])
            for lc in en_lclist:
                lcseg = lc.time_segment(tstart, tend)
                if len(lcseg) > 0:
                    new_lclist[-1].append(lcseg)

    elif isinstance(lclist[0], LightCurve):
        for lc in lclist:
            lcseg = lc.time_segment(tstart, tend)
            if len(lcseg) > 0:
                new_lclist.append(lcseg)
            else:
                print(
                    "pylag extract_lclist_time_segment WARNING: One of the light curves does not cover this time segment. Check consistency!")

    return new_lclist


def split_lclist_segments(lclist, num_segments=1, segment_length=None):
    """
    new_lclist = pylag.split_lclist_segments(lclist, tstart, tend)

    Take a list of LightCurve objects or a list of lists of multiple light curve
    objects in each energy band (as used for a lag-energy or covariance spectrum)
    and splits them into equal time segments. An extra layer is added to the list
    with each LightCurve being turned into a list of LightCurves.
    """
    new_lclist = []

    if isinstance(lclist[0], list):
        for en_lclist in lclist:
            new_lclist.append([])
            for lc in en_lclist:
                lcseg = lc.split_segments(num_segments, segment_length)
                if len(lcseg) > 0:
                    new_lclist[-1].append(lcseg)

    elif isinstance(lclist[0], LightCurve):
        for lc in lclist:
            lcseg = lc.split_segments(num_segments, segment_length)
            if len(lcseg) > 0:
                new_lclist.append(lcseg)

    return new_lclist


def lclist_separate_segments(lclist):
    """
    """
    new_lclist = []

    if isinstance(lclist[0][0], list):
        for seg_num in range(len(lclist[0][0])):
            new_lclist.append([])
            for en_lclist in lclist:
                new_lclist[-1].append([])
                for seg_lclist in en_lclist:
                    new_lclist[-1][-1].append(seg_lclist[seg_num])

    elif isinstance(lclist[0][0], LightCurve):
        for seg_num in range(len(lclist[0])):
            new_lclist.append([])
            for seg_lclist in lclist:
                new_lclist[-1].append(seg_lclist[seg_num])

    return new_lclist