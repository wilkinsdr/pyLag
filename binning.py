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

    def bin(self, x, y):
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
        bin_num = []
        for start, end in zip(self.bin_start, self.bin_end):
            bin_num.append(len([a for a in x if start <= a < end]))

        return np.array(bin_num)

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
        stdev = []
        for start, end in zip(self.bin_start, self.bin_end):
            stdev.append(np.std([b for a, b in zip(x, y) if start <= a < end]))

        return np.array(stdev)

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
        return self.std(x, y) / np.sqrt(self.num_points_in_bins(x))

    def x_error(self):
        """
        xerr = pylag.Binning.x_error()

        Returns the x error bar for each bin (central value minus minimum)
        """
        return self.bin_cent - self.bin_start

    def delta_x(self):
        """
        delta_x = pylag.Binning.delta_x()

        Returns the range spanned by each bin (bin max minus min)
        """
        return self.bin_end - self.bin_start


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
        ratio = np.exp(np.log(maxval / minval) / num)
        self.bin_start = minval * ratio ** np.array(range(num))
        self.bin_end = self.bin_start * ratio
        self.bin_cent = 0.5 * (self.bin_end + self.bin_start)
        self.num = num
