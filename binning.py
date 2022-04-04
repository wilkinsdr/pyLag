"""
pylag.binning

Provides pyLag utility classes for rebinning spectal timing products

Classes
-------
Binning    : Base class for binnng data sets
LogBinning : Derived class for logarithmically-spaced bins

v1.0 09/03/2017 - D.R. Wilkins
"""
import numpy as np
from scipy.stats import binned_statistic


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
    def __init__(self, bin_start=None, bin_end=None, bin_cent=None, bin_edges=None, num=None):
        self.bin_edges = np.array(bin_edges)

        if bin_edges is not None and bin_start is None and bin_end is None and bin_cent is None:
            self.bin_start = np.array(self.bin_edges[:-1])
            self.bin_end = np.array(self.bin_edges[1:])
        elif bin_start is not None and bin_end is not None and bin_cent is not None:
            self.bin_start = np.array(bin_start)
            self.bin_end = np.array(bin_end)
        else:
            raise ArgumentError("pylag Binning ERROR: Bins not specified")
        if bin_cent is not None:
            self.bin_cent = np.array(bin_cent)
        else:
            self.bin_cent = 0.5*(self.bin_start + self.bin_end)

        if num is not None:
            self.num = num
        else:
            self.num = len(self.bin_start)

    def bin_slow(self, x, y):
        """
        binned = pylag.Binning.bin(x, y)

        bin (x,y) data by x values into the bins specified by this object and
        return the mean value in each bin.

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        binned : ndarray
                 The mean value in each bin

        """
        binned = []
        for start, end in zip(self.bin_start, self.bin_end):
            binned.append(np.mean([b for a, b in zip(x, y) if start <= a < end]))

        return np.array(binned)

    def bin_fast(self, x, y, statistic='mean'):
        """
        binned = pylag.Binning.bin_fast(x, y)

        bin (x,y) data by x values into the bins specified by this object and
        return the mean value in each bin.

        Uses scipy binned_statistic for faster binning

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        binned : ndarray
                 The mean value in each bin

        """
        binned,_,_ = binned_statistic(x, y, statistic=statistic, bins=self.bin_edges)
        return binned

    def bin_fast_complex(self, x, y, statistic='mean'):
        """
        binned = pylag.Binning.bin_fast(x, y)

        bin (x,y) data by x values into the bins specified by this object and
        return the mean value in each bin, for complex variable y

        Uses scipy binned_statistic for faster binning

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        binned : ndarray
                 The mean value in each bin

        """
        real = y.real
        imag = y.imag

        real_binned, _, _ = binned_statistic(x, real, statistic=statistic, bins=self.bin_edges)
        imag_binned, _, _ = binned_statistic(x, imag, statistic=statistic, bins=self.bin_edges)
        binned = np.array([np.complex(r, i) for r, i in zip(real_binned, imag_binned)])
        return binned

    def bin(self, x, y, statistic='mean'):
        """
        binned = pylag.Binning.bin(x, y)

        bin (x,y) data by x values into the bins specified by this object and
        return the mean value in each bin.

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        binned : ndarray
                 The mean value in each bin

        """
        if y.dtype == 'complex' or y.dtype == 'complex64' or y.dtype == 'complex128':
            return self.bin_fast_complex(x, y, statistic=statistic)
        else:
            return self.bin_fast(x, y, statistic=statistic)

    def binned_statistic(self, x, y, stat='sum'):
        """
        binned = pylag.Binning.bin_fast(x, y)

        bin (x,y) data by x values into the bins specified by this object and
        return the mean value in each bin.

        Uses scipy binned_statistic for faster binning

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        binned : ndarray
                 The mean value in each bin

        """
        binned,_,_ = binned_statistic(x, y, statistic=stat, bins=self.bin_edges)
        return binned

    def points_in_bins(self, x, y):
        """
        points = pylag.Binning.points_in_bins(x, y)

        bin (x,y) data by x values into the bins specified by this object and
        return the list of values that fall in each bin.

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        points : list (2-dimensional)
                     A list containing the list of data points for each bin
        """
        points = []
        for start, end in zip(self.bin_start, self.bin_end):
            points.append([b for a, b in zip(x, y) if start <= a < end])

        return points

    def num_points_in_bins_slow(self, x):
        """
        bin_num = pylag.Binning.num_points_in_bins(x, y)

        Return the number of data points that fall into each bin.

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned

        Returns
        -------
        bin_num : ndarray
                  The number of data points that fall into each bin

        """
        bin_num = []
        for start, end in zip(self.bin_start, self.bin_end):
            bin_num.append(len([a for a in x if start <= a < end]))

        return np.array(bin_num)

    def num_points_in_bins(self, x):
        """
        bin_num = pylag.Binning.num_points_in_bins(x, y)

        Return the number of data points that fall into each bin.

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned

        Returns
        -------
        bin_num : ndarray
                  The number of data points that fall into each bin

        """
        # num, _, _ = binned_statistic(x, np.ones(x.shape), statistic='sum', bins=self.bin_edges)
        num, _ = np.histogram(x, bins=self.bin_edges)
        return num.astype(int)


    def std_slow(self, x, y):
        """
        stdev = pylag.Binning.std(x, y)

        Return the standard deviation of the data points in each bin

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        stdev : ndarray
                The standard deviation in each bin

        """
        stdev = []
        for start, end in zip(self.bin_start, self.bin_end):
            stdev.append(np.std([b for a, b in zip(x, y) if start <= a < end]))

        return np.array(stdev)

    def std_fast(self, x, y):
        """
        stdev = pylag.Binning.std(x, y)

        Return the standard deviation of the data points in each bin

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        stdev : ndarray
                The standard deviation in each bin

        """
        binned,_,_ = binned_statistic(x, y, statistic='std', bins=self.bin_edges)
        return binned

    def std_fast_complex(self, x, y):
        """
        stdev = pylag.Binning.std(x, y)

        Return the standard deviation of the data points in each bin
        For complex variable y

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        stdev : ndarray
                The standard deviation in each bin

        """
        real = y.real
        imag = y.imag

        # THIS DOES NOT WORK!!! Need to work out how to calculate std from reals

        real_binned, _, _ = binned_statistic(x, real, statistic='std', bins=self.bin_edges)
        imag_binned, _, _ = binned_statistic(x, imag, statistic='std', bins=self.bin_edges)
        binned = np.array([np.complex(r, i) for r, i in zip(real_binned, imag_binned)])
        return binned

    def std(self, x, y):
        """
        stdev = pylag.Binning.std(x, y)

        Return the standard deviation of the data points in each bin

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        stdev : ndarray
                The standard deviation in each bin

        """
        if y.dtype == 'complex':
            return self.std_slow(x, y)
        else:
            return self.std_fast(x, y)

    def std_error_slow(self, x, y):
        """
        stderr = pylag.Binning.std_error(x, y)

        Return the standard error of the data points in each bin

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        binned : ndarray
                 The standard error in each bin

        """
        return self.std(x, y) / np.sqrt(self.num_points_in_bins(x))

    def std_error(self, x, y):
        """
        stderr = pylag.Binning.std_error(x, y)

        Return the standard error of the data points in each bin

        Arguments
        ---------
        x : ndarray or list
            The abcissa of input data points that are to be binned
        y : ndarray or list
            The ordinate/value of input data points

        Returns
        -------
        binned : ndarray
                 The standard error in each bin

        """
        std, _, binnum = binned_statistic(x, y, statistic='std', bins=self.bin_edges)
        num_points = np.array([len(binnum[binnum==i+1]) for i in range(len(self))])

        return std / np.sqrt(num_points)

    def x_error(self):
        """
        xerr = pylag.Binning.x_error()

        Returns the x error bar for each bin (central value minus minimum)
        """
        return self.bin_cent - self.bin_start

    def x_width(self):
        """
        xw = pylag.Binning.x_width()

        Returns the width of each bin
        """
        return self.bin_end - self.bin_start

    def delta_x(self):
        """
        delta_x = pylag.Binning.delta_x()

        Returns the range spanned by each bin (bin max minus min)
        """
        return self.bin_end - self.bin_start

    def min(self):
        return self.bin_start.min()

    def max(self):
        return self.bin_end.max()

    def __len__(self):
        return len(self.bin_start)

    def __mul__(self, other):
        bin_start = self.bin_start * other
        bin_end = self.bin_end * other
        bin_cent = self.bin_cent * other
        bin_edges = self.bin_edges * other
        bin_obj = Binning(bin_start, bin_end, bin_cent, bin_edges, self.num)
        bin_obj.__class__ = self.__class__
        return bin_obj

    def __imul__(self, other):
        self.bin_start *= other
        self.bin_end *= other
        self.bin_cent *= other
        self.bin_edges *= other
        return self

    def __truediv__(self, other):
        bin_start = self.bin_start / other
        bin_end = self.bin_end / other
        bin_cent = self.bin_cent / other
        bin_edges = self.bin_edges / other
        bin_obj = Binning(bin_start, bin_end, bin_cent, bin_edges, self.num)
        bin_obj.__class__ = self.__class__
        return bin_obj

    def __itruediv__(self, other):
        self.bin_start /= other
        self.bin_end /= other
        self.bin_cent /= other
        self.bin_edges /= other
        return self

    def __str__(self):
        return "<pylag.binning.Binning: %g to %g in %d bins>" % (self.bin_start[0], self.bin_end[-1], self.num)

    def __repr__(self):
        return "<pylag.binning.Binning: %g to %g in %d bins>" % (self.bin_start[0], self.bin_end[-1], self.num)



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
        self.ratio = np.exp(np.log(maxval / minval) / num)
        self.bin_start = minval * self.ratio ** np.array(range(num))
        self.bin_end = self.bin_start * self.ratio
        self.bin_cent = 0.5 * (self.bin_end + self.bin_start)
        self.num = num
        self.bin_edges = np.concatenate((self.bin_start,[self.bin_end[-1]]))

    def bin_index(self, x):
        """
        bin = pylag.LogBinning.bin_index(x)

        Returns the index of the bin for value x

        Parameters
        ----------
        x   : float
            : Value to be placed into a bin

        Returns
        -------
        bin : int
              The bin number
        """
        return int((np.log(x / self.bin_start[0]) / np.log(self.ratio)))

    def __str__(self):
        return "<pylag.binning.LogBinning: %g to %g in %d log-spaced bins>" % (self.bin_start[0], self.bin_end[-1], self.num)

    def __repr__(self):
        return "<pylag.binning.LogBinning: %g to %g in %d log-spaced bins>" % (self.bin_start[0], self.bin_end[-1], self.num)


class LinearBinning(Binning):
    """
    pylag.LinearBinning(Binning)

    Class to perform binning of data products into linearly-spaced bins.

    Constructor: pylag.LinearBinning(minval, maxval, num)

    Constructor Arguments
    ---------------------
    minval : float
             The lowest bound of the bottom bin
    maxval : float
             The upper bound of the top bin
    num    : int
             The number of bins
    """

    def __init__(self, minval, maxval, num=None, step=None):
        if num is not None:
            self.step = (maxval - minval) / float(num)
        elif step is not None:
            self.step = step
            num = int((maxval - minval) / step + 1)
        else:
            raise ValueError("pylag LinearBinning ERROR: Must specify either number of bins or bin width")
        self.bin_start = minval + self.step * np.array(range(num))
        self.bin_end = self.bin_start + self.step
        self.bin_cent = 0.5 * (self.bin_end + self.bin_start)
        self.bin_edges = np.concatenate((self.bin_start, [self.bin_end[-1]]))
        self.num = num

    def bin_index(self, x):
        """
        bin = pylag.LinearBinning.bin_index(x)

        Returns the index of the bin for value x

        Parameters
        ----------
        x   : float
            : Value to be placed into a bin

        Returns
        -------
        bin : int
              The bin number
        """
        return int((x - self.bin_start[0]) / self.step)

    def __str__(self):
        return "<pylag.binning.LinearBinning: %g to %g in %d linear bins (delta = %g)>" % (self.bin_start[0], self.bin_end[-1], self.num, self.step)

    def __repr__(self):
        return "<pylag.binning.LinearBinning: %g to %g in %d linear bins (delta = %g)>" % (self.bin_start[0], self.bin_end[-1], self.num, self.step)
