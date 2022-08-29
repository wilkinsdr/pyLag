"""
pylag.response

Classes to handle instrument/telescope response files to produce realistic simulations
of spectral-timing products.

Classes
-------
Arf        : Class to handle ARF effective area curves

v1.0 29/08/2022 - D.R. Wilkins
"""
import numpy as np
from astropy.io import fits
from scipy.integrate import simpson

from .binning import *

class Arf(object):
    """
    pylag.response.Arf

    Class to handle effective area curves in the standard OGIP ancillary response format (ARF) for folding
    energy-dependent effective areas into simulation of timing products.

    Constructor arguments:
    filename: str, optional (default=None): FITS file to load the response from
    enbins: Binning, optional (default=None): if not laoding from a FITS file, the energy bins for the response
    arf: ndarray, optional (default=None): if not loading from a FITS file, the effective area in each bin
    """
    def __init__(self, filename=None, enbins=None, arf=None):
        if filename is not None:
            with fits.open(filename) as f:
                elow = f['SPECRESP'].data['ENERG_LO']
                ehigh = f['SPECRESP'].data['ENERG_HI']
                self.arf = f['SPECRESP'].data['SPECRESP']

            self.enbins = Binning(bin_start=elow, bin_end=ehigh, bin_cent=0.5*(elow + ehigh))
        else:
            self.enbins = enbins
            self.arf = arf

    def integrate(self, enrange=None):
        """
        integral = pylag.response.Arf.integrate(enrange=None)

        Integrate the efefctive area curve between the desired energy limits

        :param enrange: tuple, optional (default=None): lower and upper energy bounds for the integration in a tuple.
        If none, the response will be integrated over the entire energy range.
        :return: integral: float: the integral of the effective area curve
        """
        if isinstance(enrange, tuple):
                x = self.enbins.bin_cent[np.logical_and(self.enbins.bin_start>=enrange[0], self.enbins.bin_end<enrange[1])]
                y = self.arf[np.logical_and(self.enbins.bin_start >= enrange[0], self.enbins.bin_end < enrange[1])]
        else:
            x = self.enbins.bin_cent
            y = self.arf

        if len(x) < 1:
            # make sure we actually have points to integrate, otherwise simpson() isn't happy!
            return 0;

        return simpson(y, x)

    def bin(self, bins):
        """
        arf_bin = pylag.response.Arf.bin(bins)

        Rebin the effective area curve onto a new set of energy points. The original curve is intagrated over each
        bin to account for variations in the effective area.

        :param bins: Binning: Binning object sepcifying the new bins
        :return: arf_bin: Arf: the binned effective area curve
        """
        arf_bin = np.array([self.integrate((enmin, enmax)) for enmin, enmax in zip(bins.bin_start, bins.bin_end)]) / bins.x_width()
        return Arf(enbins=bins, arf=arf_bin)

    def bin_fraction(self, bins, enrange=None):
        """
        bin_frac = pylag.response.Arf.bin_fraction(bins)

        Calculate the fraction of the total effective area in each energy bin

        :param bins: Binning: Binning object sepcifying the bins for which the area fractions are to be calculated
        :param enrange: tuple, optional (default=None): lower and upper energy bounds for denominator of the area
        fraction. If None, the fraction will be calculated with respect to the entire effective area
        :return: bin_frac: ndarray: array of the effective area fractions in each bin
        """
        integral = self.integrate(enrange)
        bin_frac = np.array([self.integrate((enmin, enmax)) for enmin, enmax in zip(bins.bin_start, bins.bin_end)]) / integral
        return bin_frac

    def _getplotdata(self):
        return (self.enbins.bin_cent, self.enbins.x_error()), self.arf

    def _getplotaxes(self):
        return 'Energy / keV', 'log', 'Effective Area / cm$^2$', 'linear'