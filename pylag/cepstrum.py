from .lightcurve import *
from .binning import *

import numpy as np


class Cepstrum(object):
    """
    pylag.Cepstrum

    Class to calculate the cepstrum from a light curve.

    Member Variables
    ----------------
    quef        : ndarray
                  numpy array storing the sample quefrencies at which the
                  cepstrum is evaluated
    cepstrum    : ndarray
                  numpy array storing the calculated cepstrum

    """

    def __init__(self, lc=None, q=[], cep=[], err=None, qerr=None):
        if lc is not None:
            if not isinstance(lc, LightCurve):
                raise ValueError(
                    "pyLag CrossSpectrum ERROR: Can only compute cross spectrum between two LightCurve objects")

            self.quef, self.cepstrum = self.calculate(lc)

        else:
            self.quef = np.array(q)
            self.cepstrum = np.array(cep)

        # these will only be set once the cepstrum is binned
        self.quef_error = qerr
        self.error = err

    def calculate(self, lc):
        """
        pylag.Periodogram.calculate(lc, norm=True)

        calculate the periodogram from a light curve and store it in the member
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

        Returns
        -------
        f   : ndarray
              numpy array containing the sample frequencies at which the
              periodogram is evaluated
        per : ndarray
              numpy array containing the periodogram at each frequency
        """
        f, ft = lc.ft(all_freq=True)
        lft = np.log(np.abs(ft)**2)
        ift = scipy.fftpack.ifft(lft)
        cep = np.abs(ift)**2
        return lc.time[0:int(len(f)/2)] - lc.time[0], cep[0:int(len(f)/2)]

    def bin(self, bins, calc_error=True):
        """
        perbin = pylag.Periodogram.bin(bins)

        bin the periodogram using a Binning object then return the binned spectrum
        as a new Periodogram object

        Arguments
        ---------
        bins       : Binning
                     pyLag Binning object to perform the Binning
        calc_error : bool, optional (default=True)
                     Whether the error on each bin is required in returned periodogram

        Returns
        -------
        perbin : Periodogram
                 pyLag Periodogram object storing the newly binned periodogram

        """
        if not isinstance(bins, Binning):
            raise ValueError("pyLag Periodogram bin ERROR: Expected a Binning object")

        if calc_error:
            binned_error = bins.std_error(self.quef, self.cepstrum)
        else:
            binned_error = None

        return Cepstrum(q=bins.bin_cent, cep=bins.bin(self.quef, self.cepstrum),
                           err=binned_error, qerr=bins.x_error())

    def _getplotdata(self):
        return (self.quef, self.quef_error), (self.cepstrum, self.error)

    def _getplotaxes(self):
        return 'Quefrency / s', 'log', 'Cepstrum', 'log'


class StackedCepstrum(Cepstrum):
    def __init__(self, lc_list, bins=None, calc_error=True):
        self.cepstra = []
        for lc in lc_list:
            self.cepstra.append(Cepstrum(lc))

        self.bins = bins
        quef = []
        cepstrum = []
        err = []
        qerr = []

        if bins is not None:
            quef = bins.bin_cent
            qerr = bins.x_error()
            cepstrum, err = self.calculate(calc_error)

        Cepstrum.__init__(self, q=quef, cep=cepstrum, err=err, qerr=qerr)

    def calculate(self, calc_error=True):
        quef_list = np.hstack([c.quef for c in self.cepstra])
        cep_list = np.hstack([c.cepstrum for c in self.cepstra])

        if calc_error:
            error = self.bins.std_error(quef_list, cep_list)
        else:
            error = None

        return self.bins.bin(quef_list, cep_list), error