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
from scipy.interpolate import interp1d

from .plotter import Spectrum
from .util import printmsg

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

    LightCurve / LightCurve : Return the ratio of the count rates between two
                              light curves in a new LightCurve object
                              (e.g. for calculating hardness ratios).
                              Fractional errors from the ERROR columns are combined in
                              quadrature.
                              Light curves must have the same length and time
                              binning.

    """

    def __init__(self, filename=None, t=[], r=[], e=[], interp_gaps=False, zero_nan=True, trim=False, max_gap=0, **kwargs):
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
            self.read_fits(filename, **kwargs)

        if len(self.time) > 1:
            self.dt = self.time[1] - self.time[0]
        self.length = len(self.rate)

        if interp_gaps:
            self._interp_gaps(max_gap)
        if zero_nan:
            self._zeronan()
        if trim:
            self.trim()

        self.filename = filename

    def read_fits(self, filename, byte_swap=True, add_tstart=False, time_col='TIME', rate_col='RATE', error_col='ERROR', hdu='RATE', inst=None, bkg=False):
        """
        pylag.LightCurve.read_fits(filename)

        Read the light curve from an OGIP standard FITS file.

        Paremeters
        ----------
        filename   : string
                    The path of the FITS file from which the light curve is loaded
        byte_swap  : bool, optional (default=True)
                     Swap the byte order of the arrays after reading. This is done to convert the big-endian convention
                     in FITS files to the numpy-standard little-endian. If you don't understand what this means, leave
                     this option set to its default value
        add_tstart : bool, optional (default=False)
                     If True, add the TSTART value from the primary header to every value in the time column. This
                     option is useful for missions that start every light curve at time zero and store the start time
                     of the observation in the header, in order to use the light curve in MET (mission-elapsed time).
        time_col   : str, optional (default='TIME')
                     The name of the time column in the FITS table
        rate_col   : str, optional (default='RATE')
                     The name of the rate column in the FITS table
        error_col  : str, optional (default='ERROR')
                     The name of the error column in the FITS table
        hdu        : str, optional (default='RATE')
                     The name of the table extension in the FITS file that contains the light curve data
        inst       : str, optional (default=None)
                     A shortcut for setting all of the column and HDU names appropriately for light curves from specific
                     missions and instruments. For example, set to 'chandra' to read light curves that were created by
                     CIAO. This can be left set to None for XMM-Newton, NuSTAR and light curves created by xselect,
                     since these will all use the default naming conventions.
        bkg        : bool, optional (default=False)
                     In addition to the source rate and error columns, also read the background rate and corresponding
                     error from light curves that contain these. The background rate and error will be stored in the
                     member arrays bkg_rate and bkg_error.
        """
        # shortcut for loading Chandra light curves which helpfully have different HDU and column names!
        if inst == 'chandra':
            time_col = 'TIME'
            rate_col = 'NET_RATE'
            error_col = 'ERR_RATE'
            hdu = 'LIGHTCURVE'

        try:
            printmsg(1, "Reading light curve from " + filename)
            fitsfile = pyfits.open(filename)
            tabdata = fitsfile[hdu].data

            self.time = np.array(tabdata[time_col])
            self.rate = np.array(tabdata[rate_col])
            self.error = np.array(tabdata[error_col])

            if byte_swap:
                self._byteswap()

            self.dt = self.time[1] - self.time[0]

            if add_tstart:
                try:
                    tstart = fitsfile[0].header['TSTART']
                    self.time += tstart
                except:
                    raise AssertionError("pyLag LightCurve ERROR: Could not read TSTART from FITS header")

            if bkg:
                self.bkg_rate = np.array(tabdata['BACKV'])
                self.bkg_error = np.array(tabdata['BACKE'])

            fitsfile.close()

        except:
            raise AssertionError("pyLag LightCurve ERROR: Could not read light curve from FITS file")

    def write_fits(self, filename, byte_swap=True, time_col='TIME', rate_col='RATE', error_col='ERROR', hdu='RATE'):
        """
        pylag.LightCurve.write_fits(filename, byte_swap=True, time_col='TIME', rate_col='RATE', error_col='ERROR', hdu='RATE')

        Save the light curve to an OGIP standard FITS file.

        Paremeters
        ----------
        filename  : string
                    The path of the FITS file to which the light curve is to be saved
        byte_swap : bool, optional (default=True)
                    Swap the byte order of the arrays before saving. This is done to convert the numpy-standard
                    little-endian to the big-endian convention in FITS files
        time_col  : str, optional (default='TIME')
                    The name of the time column in the FITS table
        rate_col  : str, optional (default='RATE')
                    The name of the rate column in the FITS table
        error_col : str, optional (default='ERROR')
                    The name of the error column in the FITS table
        hdu       : str, optional (default='RATE')
                    The name of the table extension to be created in the FITS file. This extension will be created
                    alongside an empty primary HDU.
        """
        time_arr = self.time.byteswap().newbyteorder('>') if byte_swap else self.time
        rate_arr = self.rate.byteswap().newbyteorder('>') if byte_swap else self.rate
        error_arr = self.error.byteswap().newbyteorder('>') if byte_swap else self.error

        time_c = pyfits.Column(name=time_col, array=time_arr, format='D')
        rate_c = pyfits.Column(name=rate_col, array=rate_arr, format='E')
        error_c = pyfits.Column(name=error_col, array=error_arr, format='E')

        primary_hdu = pyfits.PrimaryHDU()
        table_hdu = pyfits.BinTableHDU.from_columns([time_c, rate_c, error_c], name=hdu)
        hdul = pyfits.HDUList([primary_hdu, table_hdu])
        hdul.writeto(filename, overwrite=True)

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

    def _interp_gaps(self, max_gap=0, min_gap=0, zero_gaps=False):
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
        longest_gap = 0

        for i in range(len(self.rate)):
            gap_cond = np.isnan(self.rate[i]) or (zero_gaps and self.rate[i] == 0)
            if not in_gap:
                if gap_cond:
                    in_gap = True
                    gap_start = i - 1

            elif in_gap:
                if not gap_cond:
                    gap_end = i
                    in_gap = False

                    gap_length = gap_end - gap_start

                    if (gap_length < max_gap or max_gap==0) and gap_length >= min_gap:
                        gap_count += 1
                        self.rate[gap_start:gap_end] = np.interp(self.time[gap_start:gap_end],
                                                                 [self.time[gap_start], self.time[gap_end]],
                                                                 [self.rate[gap_start], self.rate[gap_end]])

                        if gap_length > longest_gap:
                            longest_gap = gap_length

        printmsg(1, "Patched %d gaps" % gap_count)
        printmsg(1, "Longest gap was %d bins" % longest_gap)

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

    def split_segments_time(self, num_segments=1, segment_length=None, use_end=False):
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
                         created (in seconds)

        Returns
        -------
        segments : list of LightCurves
        """
        seg_bins = segment_length // np.min(np.diff(self.time)) if segment_length is not None else len(self.rate) // num_segments

        num_bins = len(self.rate) - (len(self.rate) % seg_bins) # throw away the end of the light curve
        time_arr = self.time[:num_bins].reshape(-1, seg_bins)
        rate_arr = self.rate[:num_bins].reshape(-1, seg_bins)
        error_arr = self.error[:num_bins].reshape(-1, seg_bins)

        return [LightCurve(t=t, r=r, e=e) for t, r, e in zip(time_arr, rate_arr, error_arr)]

    def split_on_gaps(self, min_segment=0):
        gaps = np.concatenate([[-1], np.argwhere(np.diff(s.time) > np.diff(s.time).min()).flatten(), [len(s.time) - 1]])
        segs = [(start + 1, end + 1) for start, end in zip(gaps[:-1], gaps[1:])]

        lc_seg = [LightCurve(t=self.time[start:end], r=self.rate[start:end], e=self.error[start:end]) for start, end in segs]

    def bin_by_gaps(self):
        gaps = np.concatenate([[-1], np.argwhere(np.diff(self.time) > np.diff(self.time).min()).flatten(), [len(self.time) - 1]])
        segs = [(start + 1, end + 1) for start, end in zip(gaps[:-1], gaps[1:])]

        t = np.array([0.5*(self.time[start] + self.time[end-1]) for start, end in segs])
        new_dt = np.array([(self.time[end-1] - self.time[start]) for start, end in segs])

        dt = np.diff(self.time).min()
        cts = np.array([np.sum(self.rate[start:end]) * dt for start, end in segs])
        r = cts / new_dt

        e = np.sqrt(r) / np.sqrt(new_dt)

        lc = LightCurve(t=t, r=r, e=e)
        lc.time_error = new_dt / 2
        return lc

    def split_on_nan(self, min_segment=0):
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
                    good_end = len(self.rate)
                    good_length = good_end - good_start
                    if good_length >= min_segment:
                        lclist.append(self[good_start:good_end])
                        good_count += 1
                    else:
                        short_count += 1

        printmsg(1, "Split light curve into  %d good segments" % good_count)
        if short_count > 0:
            printmsg(1, "%d segments too short" % short_count)
        return lclist

    def find_gaps(self):
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

        good_length = []
        gap_length = []
        good_end = 0

        for i in range(len(self.rate)):
            if not in_good_segment:
                if not np.isnan(self.rate[i]):
                    in_good_segment = True
                    good_start = i
                    gap_length.append(good_start - good_end)

            elif in_good_segment:
                if np.isnan(self.rate[i]):
                    good_end = i
                    in_good_segment = False
                    good_length.append(good_end - good_start)

                # make sure we get the end of the light curve if it's good
                elif i==len(self.rate)-1:
                    printmsg(1, "got the end")
                    good_end = len(self.rate)
                    good_length.append(good_end - good_start)

        printmsg(1, "Good segment lengths: ", good_length)
        printmsg(1, "Gaps: ", gap_length)

    def remove_nan(self, to_self=False):
        t = self.time[np.logical_not(np.isnan(self.rate))]
        r = self.rate[np.logical_not(np.isnan(self.rate))]
        e = self.error[np.logical_not(np.isnan(self.rate))]

        if to_self:
            self.time = t
            self.rate = r
            self.error = e

        else:
            lc = LightCurve(t=t, r=r, e=e)
            lc.__class__ = self.__class__
            return lc

    def remove_gaps(self, to_self=False):
        t = self.time[self.rate>0]
        r = self.rate[self.rate>0]
        e = self.error[self.rate>0]

        if to_self:
            self.time = t
            self.rate = r
            self.error = e

        else:
            lc = LightCurve(t=t,r=r, e=e)
            lc.__class__ = self.__class__
            return lc

    def rebin_slow(self, tbin):
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
            printmsg(1, "pylag LightCurve Rebin WARNING: New time binning is not a multiple of the old")

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
            printmsg(1, "pylag LightCurve Rebin WARNING: New time binning is not a multiple of the old")

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

    def rebin(self, tbin=None, time=None):
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
        # if tbin <= self.dt:
        #     raise ValueError("pylag LightCurve Rebin ERROR: Must rebin light curve into larger bins")
        # if tbin % self.dt != 0:
        #     print("pylag LightCurve Rebin WARNING: New time binning is not a multiple of the old")

        if time is None:
            time = np.arange(min(self.time), max(self.time)+tbin, tbin)
        else:
            tbin = np.diff(time)
        counts, _, binid = binned_statistic(self.time, self.dt*self.rate, statistic='sum', bins=time)

        # count the number of original time bins in each new time bin for the efefctive exposure
        num_in_bin = np.array([np.sum(binid == (i + 1)) for i in range(len(time) - 1)])

        rate = counts / (self.dt*num_in_bin)
        err = rate * np.sqrt(counts) / counts

        binlc = LightCurve(t=time[:-1], r=rate, e=err)
        # make sure the returned object has the right class (if this is called from a derived class)
        binlc.__class__ = self.__class__
        return binlc

    # TODO: Counts binning based on accumulated count over previous time window

    def interpolate(self, tbin=None, time=None, interp_kind='nearest'):
        if time is None:
            time = np.arange(min(self.time), max(self.time)+tbin, tbin)
        rate_interp = interp1d(self.time, self.rate, kind=interp_kind, fill_value='extrapolate')
        tdiff = np.diff(time)
        dt = np.hstack([tdiff, tdiff[-1]])
        rate = rate_interp(time)
        counts = rate * dt
        err = rate * np.sqrt(counts) / counts

        interp_lc = LightCurve(t=time, r=rate, e=err)
        # make sure the returned object has the right class (if this is called from a derived class)
        interp_lc.__class__ = self.__class__
        return interp_lc

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

    def time_at_rate(self, ratemin, ratemax):
        bins = np.array([ratemin, ratemax])
        count, _ = np.histogram(self.rate, bins=bins)
        return count[0] * self.dt

    def ft(self, all_freq=False):
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

        if all_freq:
            return freq, ft
        else:
            return freq[:int(self.length / 2)], ft[:int(self.length / 2)]

    def ftfreq(self, all_freq=False):
        """
        freq = pylag.LightCurve.ftfreq()

        Returns the FFT sample frequencies for this light curve

        Return Values
        -------------
        freq : ndarray
               The sample frequencies
        """
        freq = scipy.fftpack.fftfreq(self.length, d=self.dt)

        if all_freq:
            return freq
        else:
            return freq[:int(self.length / 2)]

    def ft_uneven(self, freq=None, all_freq=False, ft_sign=-1):
        """
        freq, ft = pylag.LightCurve.ft()

        Directly evaluate the discrete Fourier transform of an unevenly sampled light curve
        using the algorithm of Scargle 1989, ApJ 343, 874

        Note that time bins without data should be removed from the light curve before running this

        Arguments
        ---------
        freq     : ndarray, optional (default=None)
                   Array of (linear) frequencies at which the Fourier transform is to be evaluated
                   If None, the Fourier transform will be evaluated at the default frequencies for an FFT
        all_freq : boolean, optional (default=False)
                   If True, include the negative frequencies
        ft_sign  : integer, optional (default=-1)
                   Sign of the exponent in the Fourier transform. -1 to be consistent

        Return Values
        -------------
        freq : ndarray
               The sample frequencies
        ft   : ndarray
               The Fourier transfrom of the light curve
        """
        if freq is None:
            dt = np.min(np.diff(self.time))
            freq = scipy.fftpack.fftfreq(int((self.time.max() - self.time.min()) / dt), d=dt)
            if not all_freq:
                freq = freq[:int(len(freq) / 2)]

        csum = np.array([np.sum(np.cos(2. * 2*np.pi*f * self.time)) for f in freq[1:]])
        ssum = np.array([np.sum(np.sin(2. * 2*np.pi*f * self.time)) for f in freq[1:]])
        ftau = 0.5 * np.arctan2(ssum, csum)

        sumr = np.array([np.sum(self.rate * np.cos(2*np.pi*f * self.time - tau)) for f, tau in zip(freq[1:], ftau)])
        sumi = np.array([np.sum(self.rate * np.sin(2*np.pi*f * self.time - tau)) for f, tau in zip(freq[1:], ftau)])

        scos2 = np.array([np.sum((np.cos(2*np.pi*f * self.time - tau)) ** 2) for f, tau in zip(freq[1:], ftau)])
        ssin2 = np.array([np.sum((np.sin(2*np.pi*f * self.time - tau)) ** 2) for f, tau in zip(freq[1:], ftau)])

        ft_real = (1. / np.sqrt(2.)) * sumr / np.sqrt(scos2)
        ft_imag = ft_sign * (1. / np.sqrt(2.)) * sumi / np.sqrt(ssin2)
        fphase = ftau - freq[1:] * np.min(self.time)

        ft = np.zeros_like(freq, dtype=complex)
        ft[0] = np.sum(self.rate) / np.sqrt(len(self.rate))
        ft[1:] = (ft_real + 1j * ft_imag) * np.exp(1j * fphase)

        return freq, ft

    def autocorr(self):
        """
        freq, ft = pylag.LightCurve.ft()

        Returns the discrete Fourier transform (FFT) of the light curve.
        Only the positive frequencies are returned (N/2 points for light curve length N)

        Return Values
        -------------
        lag  : ndarray
               Lag times
        corr : ndarray
               The autocorrelation function
        """
        t = (self.time - self.time.min())[:self.length//2]
        corr = np.correlate(self.rate, self.rate)[self.length//2:]
        return t, corr


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

    def num_freq_in_range_slow(self, fmin, fmax):
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
        return int((fmax - fmin) / (freq[1] - freq[0])) + 1

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

    def sort_time(self):
        """
        Sort the light curve points in time order
        """
        t, r, e = zip(*sorted(zip(self.time, self.rate, self.error)))
        return LightCurve(t=np.array(t), r=np.array(r), e=np.array(e))

    def log(self):
        """
        Return the logarithm of the count rate
        """
        r = np.log(self.rate)
        e = self.error / self.rate
        return LightCurve(t=self.time, r=r, e=e)

    def rescale_time(self, mult=None, mass=None):
        """
        Rescale the time axis of the light curve, multiplying by a constant
        e.g. for GM/c^3 to s conversion
        """
        if mass is not None:
            mult = 6.67E-11 * mass * 2E30 / (3E8)**3
        t = self.time * mult
        lc = LightCurve(t=t, r=self.rate, e=self.error)
        lc.__class__ = self.__class__
        return lc

    def first_deriv(self):
        """
        Return the first derivative of the light curve
        """
        dt = np.diff(self.time)
        dr = self.rate[1:] - self.rate[:-1]
        drdt = dr / dt
        return LightCurve(t=self.time[:-1], r=drdt, e=np.zeros(dt.shape))

    def resample_noise(self):
        """
        lc = pylag.LightCurve.resample_noise

        Add Poisson noise to the light curve and return the noise light curve as a
        new LightCurve object. For each time bin, the photon counts are drawn from
        a Poisson distribution with mean according to the current count rate in the
        bin.

        Return Values
        -------------
        lc : SimLightCurve
             SimLightCurve object containing the new, noisy light curve
        """
        # sqrt(N) noise applies to the number of counts, not the rate
        counts = self.rate * self.dt
        counts[counts<0] = 0
        # draw the counts in each time bin from a Poisson distribution
        # with the mean set according to the original number of counts in the bin
        rnd_counts = np.random.poisson(counts)
        rate = rnd_counts.astype(float) / self.dt
        # sqrt(N) errors again as if we're making a measurement
        error = np.sqrt(self.rate / self.dt)

        resample_lc = LightCurve(t=self.time, r=rate, e=error)
        resample_lc.__class__ = self.__class__
        return resample_lc

    def moving_average(self, window_size=3):
        window = np.ones(int(window_size)) / float(window_size)
        r_avg = np.convolve(self.rate, window, 'same')
        lc_avg = LightCurve(t=self.time, r=r_avg, e=np.zeros(self.time.shape))
        lc_avg.__class__ = self.__class__
        return lc_avg

    def resample_moving_average(self, window_size=3, num_resamples=10000):
        resamples = []
        for n in range(num_resamples):
            resamples.append(self.resample_noise().moving_average(window_size))
        r = self.moving_average(window_size).rate
        e = np.std(np.array([l.rate for l in resamples]), axis=0)
        resample_lc = LightCurve(t=self.time, r=r, e=e)
        resample_lc.__class__ = self.__class__
        return resample_lc

    def find_nearest(self, other, time_mode='matches'):
        if isinstance(other, LightCurve):
            idx = [np.abs(self.time - t).argmin() for t in other.time]
            if time_mode == 'matches':
                t = self.time[idx]
            elif time_mode == 'orig':
                t = other.time
            match_lc = LightCurve(t=t, r=self.rate[idx], e=self.error[idx])
            match_lc.__class__ = self.__class__
            return match_lc
        else:
            return NotImplemented

    def add_bkg(self, to_self=False):
        rsum = self.rate + self.bkg_rate
        esum = np.sqrt(self.error**2 + self.bkg_error**2)
        if to_self:
            self.rate = rsum
            self.error = esum
        else:
            return LightCurve(t=self.time, r=rsum, e=esum)

    def to_df(self, errors=False):
        import pandas as pd
        cols = {'time':self.time, 'rate':self.rate}
        if errors:
            cols['error'] = self.error
        df = pd.DataFrame(cols).set_index('time')
        return df

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

        elif isinstance(other, (float, int)):
            newrate = self.rate + other
            newerr = np.sqrt(newrate*self.dt) / self.dt
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

    def __truediv__(self, other):
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

    def __str__(self):
        return "<pylag.lightcurve.LightCurve: %d time bins, dt = %g%s>" % (len(self), self.dt, ", loaded from: %s" % self.filename if self.filename is not None else "")

    def __repr__(self):
        return "<pylag.lightcurve.LightCurve: %d time bins, dt = %g%s>" % (len(self), self.dt, ", loaded from: %s" % self.filename if self.filename is not None else "")

    def _getplotdata(self):
        return self.time, (self.rate, self.error)

    def _getplotaxes(self):
        return 'Time / s', 'linear', 'Count Rate / ct s$^{-1}$', 'linear'


class VariableBinLightCurve(LightCurve):
    """
    pylag.VariableBinLightCurve

    LightCurve class for storing and plotting light curves with variable bin sizes (encoded as the central time of each
    bin with symmetric error).

    WARNING: Do not use for computations involving FFTs
    """
    def __init__(self, t=[], te=[], r=[], e=[]):
        self.time = np.array(t)
        if len(r) > 0:
            self.rate = np.array(r)
        else:
            self.rate = np.zeros(len(t))
        if len(e) > 0:
            self.error = np.array(e)
        else:
            self.error = np.zeros(len(t))
        if len(te) > 0:
            self.time_error = np.array(te)
        else:
            self.time_error = np.zeros(len(t))

    def _getplotdata(self):
        return (self.time, self.time_error), (self.rate, self.error)

    def ft(self, all_freq=False):
        raise AssertionError("Ouch! Please don't try to FFT me!!")


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


def stacked_mean_count_rate_slow(lclist):
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
    return np.mean(np.hstack([lc.rate for lc in lclist]))


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

    printmsg(1, "Extracting simultaneous light curve portion from t=%g to %g" % (start, end))
    printmsg(1, "Simultaneous portion length = %g" % (end - start))

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
    out_lc1.__class__ = lc1.__class__
    out_lc2 = LightCurve(t=time2, r=rate2, e=err2)
    out_lc2.__class__ = lc2.__class__

    return out_lc1, out_lc2

def match_lc_timebins(lc1, lc2):
    """
    pylab.match_lc_timebins(a, b)

    extract simultaneous light curve segments by picking out the time bins that match

    Note that this requires the time bins to align exactly between the two light curves

    Arguments
    ---------
    lc1 : LightCurve
          First input light curve
    lc2 : LightCurve
          Second input light curve

    Returns
    -------
    out_lc1 : LightCurve
              LightCurve object containing the matched portion of the first
              light curve
    out_lc2 : LightCurve
              LightCurve object containing the matched portion of the second
              light curve
    """
    # find the time bins that
    tsim = lc1.time[np.isin(lc1.time, lc2.time)]
    # extract the corresponding rate and error bins
    rsim1 = lc1.rate[np.isin(lc1.time, tsim)]
    rsim2 = lc2.rate[np.isin(lc2.time, tsim)]
    esim1 = lc1.error[np.isin(lc1.time, tsim)]
    esim2 = lc2.error[np.isin(lc2.time, tsim)]

    out_lc1 = LightCurve(t=tsim, r=rsim1, e=esim1)
    out_lc1.__class__ = lc1.__class__
    out_lc2 = LightCurve(t=tsim, r=rsim2, e=esim2)
    out_lc2.__class__ = lc2.__class__
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
            self.enmin, self.enmax, self.lclist = self.find_light_curves(searchstr, lcfiles, **kwargs)

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

    def zero_time(self):
        new_lclist = []

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    lc.time -= lc.time.min()

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                lc.time -= lc.time.min()

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

    def split_on_gaps(self, min_segment=0):
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
                    lcseg = lc.split_on_gaps(min_segment)
                    for seg in lcseg:
                        new_lclist[-1].append(seg)

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                lcseg = lc.split_on_gaps(min_segment)
                new_lclist.append(lcseg)

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)

    def concatenate_segments(self):
        new_lclist = []

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append(LightCurve().concatenate(en_lclist).sort_time())

        elif isinstance(self.lclist[0], LightCurve):
            raise AssertionError("EnergyLCList object does not contain multiple segments per energy band")

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

    def remove_zeros(self):
        """
        Remove time bins where any of the light curves go to zero

        :return:
        """
        if isinstance(self.lclist[0], list):
            new_lclist = []
            for en in range(len(self.lclist)):
                new_lclist.append([])
            for seg in range(len(self.lclist[0])):
                filt = np.ones(self.lclist[0][seg].rate.shape).astype(bool)
                for en in range(len(self.lclist)):
                    filt *= (self.lclist[en][seg].rate > 0)
                for en in range(len(self.lclist)):
                    t = self.lclist[en][seg].time[filt]
                    r = self.lclist[en][seg].rate[filt]
                    e = self.lclist[en][seg].error[filt]
                    lc = LightCurve(t=t, r=r, e=e)
                    lc.__class__ = self.lclist[en][seg].__class__
                    new_lclist[en].append(lc)

        elif isinstance(self.lclist[0], LightCurve):
            new_lclist = []
            filt = np.ones(self.lclist[0].rate.shape).astype(bool)
            for en in range(len(self.lclist)):
                filt *= (self.lclist[en].rate > 0)
            for en in range(len(self.lclist)):
                t = self.lclist[en].time[filt]
                r = self.lclist[en].rate[filt]
                e = self.lclist[en].error[filt]
                lc = LightCurve(t=t, r=r, e=e)
                lc.__class__ = self.lclist[en].__class__
                new_lclist.append(lc)

        lclist = EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)
        lclist.__class__ = self.__class__
        return lclist

    def mean_spectrum(self):
        en = 0.5 * (self.enmin + self.enmax)
        enerr = 0.5 * (self.enmax - self.enmin)

        if isinstance(self.lclist[0], LightCurve):
            avg_rate = np.array([l.mean() for l in self.lclist])
            std_rate = np.array([np.std(l.rate) for l in self.lclist])
        elif isinstance(self.lclist[0], list):
            concat_lclist = self.concatenate_segments()
            avg_rate = np.array([l.mean() for l in concat_lclist.lclist])
            std_rate = np.array([np.std(l.rate) for l in concat_lclist.lclist])

        return Spectrum((en, enerr), (avg_rate, std_rate))

    def sum_lightcurve(self):
        if isinstance(self.lclist[0], LightCurve):
            time = self.lclist[0].time
            dt = time[1] - time[0]
            sum_rate = np.sum(np.vstack([lc.rate for lc in self.lclist]), axis=0)
            err = np.sqrt(sum_rate / dt)
            lc = LightCurve(t=time, r=sum_rate, e=err)
            lc.__class__ = self.lclist[0].__class__
            return lc
        elif isinstance(self.lclist[0], list):
            rate_list = []
            time_list = []
            err_list = []
            for seg in range(len(self.lclist[0])):
                time_list.append(self.lclist[0][seg].time)
                dt = time_list[-1][1] - time_list[-1][0]
                rate_list.append(np.sum(np.vstack([self.lclist[en][seg].rate for en in range(len(self.lclist))]), axis=0))
                err_list.append(np.sqrt(rate_list[-1] / dt))
            return [LightCurve(t=t, r=r, e=e) for t, r, e in zip(time_list, rate_list, err_list)]

    def resample_noise(self):
        new_lclist = []

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    new_lclist[-1].append(lc.resample_noise())

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                new_lclist.append(lc.resample_noise())

        return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)

    def to_array(self):
        if isinstance(self.lclist[0], list):
            Nen = len(self.lclist)
            Nseg = len(self.lclist[0])
            return [np.vstack([self.lclist[ien][iseg].rate for ien in range(Nen)]) for iseg in range(Nseg)]

        elif isinstance(self.lclist[0], LightCurve):
            return np.vstack([lc.rate for lc in self.lclist])

    def write_grid(self, filename, fmt='%15.8g', delimiter=' '):
        arr = self.to_array()
        if isinstance(arr, list):
            for n, a in enumerate(arr):
                t = self.lclist[0][n].time
                np.savetxt('%s_%02d.dat' % (filename, n), np.vstack([t, a]).T, fmt=fmt, delimiter=delimiter)
        else:
            t = self.lclist[0].time
            np.savetxt(filenmae, np.vstack(t, arr).T, fmt=fmt, delimiter=delimiter)


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
                    #lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
                    lc1s, lc2s = match_lc_timebins(lc1, lc2)
                    lclist[-1].append(lc1s + lc2s)
        else:
            for lc1, lc2 in zip(self.lclist, other.lclist):
                #lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
                lc1s, lc2s = match_lc_timebins(lc1, lc2)
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
                    #lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
                    lc1s, lc2s = match_lc_timebins(lc1, lc2)
                    lclist[-1].append(lc1s - lc2s)
        else:
            for lc1, lc2 in zip(self.lclist, other.lclist):
                #lc1s, lc2s = extract_sim_lightcurves(lc1, lc2)
                lc1s, lc2s = match_lc_timebins(lc1, lc2)
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


def stack_lclists(lclists):
    """
    Stack a list of EnergyLCLists into a single, EnergyLCList with multiple segments
    """
    enmin = lclists[0].enmin
    enmax = lclists[0].enmax

    stacked_list = [[] for n in range(len(lclists[0].lclist))]    # an empty list for each energy band
    for l in lclists:
        if isinstance(l.lclist[0], list):
            for ien in range(len(l.lclist)):
                stacked_list[ien] += l.lclist[ien]
        elif isinstance(l.lclist[0], LightCurve):
            for ien in range(len(l.lclist)):
                stacked_list[ien].append(l.lclist[ien])

    lcl = EnergyLCList(enmin=enmin, enmax=enmax, lclist=stacked_list)
    lcl.__class__ = lclists[0].__class__
    return lcl


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


def ifft_lightcurve(f, ft):
    r = np.real(scipy.fftpack.ifft(ft))
    nyq = np.abs(np.max(f))
    t = np.arange(0, 1./f[1] + 1./(2.*nyq), 1./(2.*nyq))
    return LightCurve(t=t, r=r, e=[])
