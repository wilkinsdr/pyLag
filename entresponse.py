"""
pylag.entresponse

Classes for handling energy-time response functions from ray tracing simulations

v1.0 - 05/09/2017 - D.R. Wilkins
"""
from .simulator import *
from .lightcurve import *
from .lag_frequency_spectrum import *
from .lag_energy_spectrum import *

import numpy as np

try:
    import astropy.io.fits as pyfits
except ModuleNotFoundError:
    import pyfits

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import binned_statistic

from .binning import *
from .plotter import Spectrum

from .math_functions import *


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)


class ENTResponse(object):
    """
    pylag.ENTResponse

    Classes to store the energy-time response function from a ray tracing simulation. ENTResponse
    objects are used as the basis for modelling X-ray reverberation and store the count rate as a
    function of energy and time after a single flare of emission from the corona (i.e. they are
    the Green's function for the disc response).

    Member Variables
    ----------------
    time    : ndarray
              The time axis points at which the response function is sampled
    en_bins : Binning
              pyLag Binning object describing the energy binning
    ent     : ndarray
              2-dimensional array storing the count rate at each (energy, time)
    tstart  : float
              The arrival time of the continuum (total light travel time from
              source to image plane). This is the zero-point in time and the time
              axis is relative to tstart.

    Constructor: pylag.ENTResponse(filename=None, hduname=None, hdunum=0, en_bins=None, t=None, ent=None, logbin_en=None, tstart=0.)

    Constructor Arguments
    ---------------------
    filename  : string, optional (default=None)
                Path to the FITS file from which to load the response function
    hduname   : string, optional (default=None)
                Name of the HDU/extension within the FITS file. If None, the first extension will be loaded.
    hdu       : string or int, optional (default='RESPONSE')
                Name or number of the HDU/extension within the FITS file
    en_bins   : Binning, optional (default=None)
                If not loading a response, a pyLag binning object describing the energy bins
    t         : ndarray, optional (default=None)
                If not loading a response, an array of the time axis points
    ent       : ndarray, optional (default=None)
                If not loading a response, a 2-dimensional array storing the count rate at each (energy, time)
    logbin_en : bool, optional (default=None)
                If not loading a response, flag showing energy bins are logarithmic
    tstart    : float, optional (default=None)
                If not loading a response, the continuum arrival time

    Overloaded Operators
    --------------------
    ENTResponse + ENTResponse  : add two response functions together and return the result in a
                                 new ENTResponse object. Responses must have the same energy and
                                 time bins.

    ENTResponse - ENTResponse  : subtract one response from another and return the result in a
                                 new ENTResponse object. Responses must have the same energy and
                                 time bins.

    ENTResponse * ENTResponse  : multiply one response function by another (the count rate in each
                                 ach bin is multiplied by that in the corresponding bin in the
                                 other response). Useful for applying absoprtion.

    ENTResponse * float        : multiply the count rate in each bin by a constant

    ENTResponse / ENTResponse  : divide one response function by another (the count rate ine each
                                 in each bin is divided by that in the corresponding bin in
                                 the other response).

    ENTResponse / float        : divide the count rate in each bin by a constant

    Arithmetic operators can be used inline (e.g. +=, -=, *=, /=)
    """
    def __init__(self, filename=None, hdu='RESPONSE', en_bins=None, t=None, ent=None, logbin_en=None, tstart=0.):
        self.en = np.array([])
        self.time = np.array([])
        self.ent = np.array([])
        self.tstart = tstart

        if filename is not None:
            self.en_bins, self.time, self.ent, self.logbin_en, self.tstart = self.read_fits(filename, hdu)
        else:
            if en_bins is not None:
                self.en_bins = en_bins
            if t is not None:
                self.time = t
            if ent is not None:
                self.ent = ent
            elif en_bins is not None and t is not None:
                self.ent = np.zeros((len(en_bins), len(t)))
            if logbin_en is not None:
                self.logbin_en = logbin_en
            else:
                self.logbin_en = False

        self.t0 = min(self.time)
        self.dt = self.time[1] - self.time[0]

    @staticmethod
    def read_fits(filename, hduname='RESPONSE', byte_swap=True):
        """
        pylag.ENTResponse.read_fits(filename, hduname=None, byte_swap=True)

        Read the energy/time response from a FITS file

        Arguments
        ---------
        filename  : string
                    The path of the FITS file from which the response is to be loaded
        hduname   : string or int, optional (default='RESPONSE')
                    Name or number of HDU or extension within the FITS file
        byte_swap : bool, optional (default=True)
                    Swap the byte order of the array.  This is necessary when reading
                    from a FITS file (big endian) if you want to use the FFT functions
                    in scipy.fftpack since these only supported little endian
                    (and pyfits preserves the endianness read from the file)

        Returns
        -------
        en_bins   : Binning, optional (default=None)
                    A pyLag binning object describing the energy bins
        t         : ndarray, optional (default=None)
                    Aan array of the time axis points
        ent       : ndarray, optional (default=None)
                    A 2-dimensional array storing the count rate at each (energy, time)
        logbin_en : bool, optional (default=None)
                    Flag showing energy bins are logarithmic
        tstart    : float, optional (default=None)
                    The continuum arrival time
        """
        try:
            fitsfile = pyfits.open(filename)
        except:
            raise AssertionError("pyLag ENTResponse ERROR: Could not open FITS file " + filename)

        try:
            if hduname is not None:
                hdu = fitsfile[hduname]
        except:
            raise AssertionError("pyLag ENTResponse ERROR: Could not open HDU")

        ent = np.array(hdu.data)
        if byte_swap:
            ent = ent.byteswap().newbyteorder('<')

        try:
            en0 = hdu.header['EN0']
            Nen = hdu.header['NEN']
            try:
                logbin_en = hdu.header['ENLOG']
            except:
                logbin_en = False

            try:
                enmax = hdu.header['ENMAX']
            except:
                # fallback for old format FITS header
                den = hdu.header['DEN']
                if logbin_en:
                    enmax = en0 * den**Nen
                else:
                    enmax = en0 + den*Nen

            if logbin_en:
                en_bins = LogBinning(en0, enmax, Nen)
            else:
                en_bins = LinearBinning(en0, enmax, Nen)

            t0 = hdu.header['T0']
            dt = hdu.header['DT']
            Nt = hdu.header['NT']
            tstart = hdu.header['TSTART']

            t = t0 + dt * np.arange(0, Nt, 1)
        except:
            raise AssertionError("pyLag ENTResponse ERROR: Could not read axis information from FITS header")

        fitsfile.close()

        return en_bins, t, ent, logbin_en, tstart

    def t_index(self, time, from_start=False):
        if from_start:
            return int((time - (self.t0 + self.tstart)) / self.dt)
        else:
            return int((time - self.t0) / self.dt)

    def en_index(self, energy):
        return self.en_bins.bin_index(energy)

    def add_continuum(self, tcont, gamma, ref_frac, from_start=True):
        """
        entc = pylag.ENTResponse.add_continuum(tcont, gamma, ref_frac, from_start=True)

        Returns a new ENTResponse object with the primary continuum emission added.
        The continuum has a power law spectrum and arrives in a single time bin,
        appropriate for a point source (if more complex or extended continuum response
        functions are needed, see the classes in pylag.continuum).

        Arguments
        ---------
        tcont      : float
                     Arrival time of the continuum emission. If set to -1, will use the continuum
                     time pre-defined for this response function.
        gamma      : float
                     Photon index of the continuum spectrum
        ref_frac   : float
                     The reflection fraction, defined as the ratio of counts in the reflection
                     spectrum (i.e. the counts already in the response function) to the counts
                     in the continuum, over the whole energy band of this response.
        from_start : bool, optional (default=True)
                     Whether the continuum arrival is defined as the total light travel time from
                     the source to the image plane (default yes) or, if false, relative to the
                     zero time on the time axis.

        Returns
        --------
        entc : ENTResponse
               ENTResponse object with the continuum added
        """
        if tcont == -1:
            tcont = self.tstart

        ti = self.t_index(tcont, from_start)

        # the power law continuum, factoring in variable-width energy bins (i.e. logarithmically spaced)
        cont = self.en_bins.bin_cent ** -gamma * (self.en_bins.bin_end - self.en_bins.bin_start)

        tot_ent = np.array(self.ent)
        tot_ent[:, ti] = self.ent[:, ti] + np.sum(self.ent) * cont / (ref_frac * np.sum(cont))

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=tot_ent, tstart=self.tstart)

    def continuum_ent(self, tcont, gamma, ref_frac, from_start=True):
        """
        cont = pylag.ENTResponse.add_continuum(tcont, gamma, ref_frac, from_start=True)

        Returns a new ENTResponse object with just the primary continuum emission added
        that would be added by the add_continuum method. Useful for plotting the continuum.
        The continuum has a power law spectrum and arrives in a single time bin,
        appropriate for a point source (if more complex or extended continuum response
        functions are needed, see the classes in pylag.continuum).

        Arguments
        ---------
        tcont      : float
                     Arrival time of the continuum emission. If set to -1, will use the continuum
                     time pre-defined for this response function.
        gamma      : float
                     Photon index of the continuum spectrum
        ref_frac   : float
                     The reflection fraction, defined as the ratio of counts in the reflection
                     spectrum (i.e. the counts already in the response function) to the counts
                     in the continuum, over the whole energy band of this response.
        from_start : bool, optional (default=True)
                     Whether the continuum arrival is defined as the total light travel time from
                     the source to the image plane (default yes) or, if false, relative to the
                     zero time on the time axis.

        Returns
        --------
        cont : ENTResponse
               ENTResponse object containing only the continuum
        """
        ti = self.t_index(tcont, from_start)

        cont = self.en_bins.bin_cent ** -gamma

        cont_ent = np.zeros(self.ent.shape)
        cont_ent[:, ti] = np.sum(self.ent) * cont / (ref_frac * np.sum(cont))

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=cont_ent, tstart=self.tstart)

    def energy_range(self, enmin, enmax):
        """
        ent = pylag.ENTResponse.energy_range(enmin, enmax)

        Returns just the portion of the response in the specified energy range.

        Arguments
        ---------
        enmin : float
                Lower energy bound.
        enmax : float
                Upper energy bound.

        Returns
        --------
        ent : ENTResponse
              ENTResponse object containing just the specified range.
        """
        enstarti = self.en_index(enmin)
        enendi = self.en_index(enmax)
        return ENTResponse(en_bins=self.en_bins[enstarti:enendi], t=self.time, ent=self.ent[enstarti:enendi, :], tstart=self.tstart)

    def time_range(self, tmin, tmax):
        """
        ent = pylag.ENTResponse.time_range(tmin, tmax)

        Returns just the portion of the response in the specified time range.

        Arguments
        ---------
        tmin : float
               Lower time bound.
        tmax : float
               Upper time bound.

        Returns
        --------
        ent : ENTResponse
              ENTResponse object containing just the specified range.
        """
        tstarti = self.t_index(tmin)
        tendi = self.t_index(tmax)
        return ENTResponse(en_bins=self.en_bins, t=self.time[tstarti:tendi], ent=self.ent[:, tstarti:tendi], tstart=self.tstart)

    def rebin_energy(self, bins=None, Nen=None):
        """
        ent = pylag.ENTResponse.rebin_energy(bins=None, Nen=None)

        Returns the response function, rebinned on the energy axis.

        Arguments
        ---------
        bins : Binning, optional (default=None)
               pyLag Binning object defining the desired binning.
        Nen  : int, optional (default=None)
               If bins=None, rebin into this many logarithmically-spaced bins.

        Returns
        --------
        ent : ENTResponse
              Rebinned ENTResponse object.
        """
        if bins is None:
            bins = LogBinning(self.en_bins.min(), self.en_bins.max(), Nen)

        ent = []
        for bin_start, bin_end in zip(bins.bin_start, bins.bin_end):
            enstarti = self.en_index(bin_start)
            enendi = self.en_index(bin_end)
            ent.append(np.sum(self.ent[enstarti:enendi,:], axis=0))
        return ENTResponse(en_bins=bins, t=self.time, ent=np.array(ent), tstart=self.tstart)

    def rebin_time(self, bins=None, dt=None, Nt=None, statistic='mean'):
        """
        ent = pylag.ENTResponse.rebin_time(bins=None, dt=None, Nt=None)

        Returns the response function, rebinned on the time axis.

        Arguments
        ---------
        bins : Binning, optional (default=None)
               pyLag Binning object defining the desired binning.
        dt   : float, optional (default=None)
               If bins=None, the desired time step.
        Nt   : int, optional (degault=None)
               If bins=None and dt=None, rebin into this many time bins

        Returns
        --------
        ent : ENTResponse
              Rebinned ENTResponse object.
        """
        if bins is None:
            if dt is not None:
                bins = LinearBinning(self.time.min(), self.time.max(), step=dt)
            if Nt is not None:
                bins = LinearBinning(self.time.min(), self.time.max(), num=Nt)

        ent = []
        for ien in range(self.ent.shape[0]):
            ent.append(bins.bin(self.time, self.ent[ien,:], statistic=statistic))
        return ENTResponse(en_bins=self.en_bins, t=bins.bin_start, ent=np.array(ent), tstart=self.tstart)

    def rescale_time(self, mult=None, mass=None):
        """
        ent = pylag.ENTResponse.rescale_time(mult=None, mass=None)

        Rescale the time axis of the response function, multiplying by a constant
        e.g. for GM/c^3 to s conversion

        Arguments
        ---------
        mult : float, optional (default=None)
               Factor by which to multiply the time axis.
        mass : float, optional (default=None)
               Calculate the multipl;icative factor automatically for this black hole mass
               (in units of Solar masses)

        Returns
        --------
        ent : ENTResponse
              ENTResponse object with rescaled time axis.
        """
        if mass is not None:
            mult = 6.67E-11 * mass * 2E30 / (3E8)**3
        t = self.time * mult
        return ENTResponse(en_bins=self.en_bins, t=t, ent=self.ent, tstart=self.tstart*mult)

    def moving_average_energy(self, window_size=3):
        """
        ent = pylag.ENTResponse.moving_average_energy(window_size=3)

        Apply a moving average filter along the energy axis, in each time bin, to reduce
        numerical noise caused by binning of spectrum.

        Arguments
        ---------
        window_size : int, optional (default=3)
                      Total window size of the moving average filter. 3 is 1 point either side.

        Returns
        --------
        ent : ENTResponse
              ENTResponse object with moving average filter applied.
        """
        window = np.ones(int(window_size)) / float(window_size)
        ent_avg = np.zeros(self.ent.shape)
        for it in range(self.ent.shape[1]):
            ent_avg[...,it] = np.convolve(self.ent[...,it], window, 'same')
        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=ent_avg, tstart=self.tstart)

    def norm(self):
        """
        ent = pylag.ENTResponse.norm()

        Normalises the response function such that the summed count rate across all
        energy and time bins is unity. Useful when using the response as a convolutional
        line response.

        Returns
        --------
        ent : ENTResponse
              Normalised ENTResponse object.
        """
        norm_ent = self.ent / self.ent.sum()

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=norm_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def plot_image(self, vmin=None, vmax=None, mult_scale=True, cmap='gray_r', log_scale=True):
        """
        p = pylag.ENTResponse.plot_image(vmin=None, vmax=None, mult_scale=True, cmap='gray_r', log_scale=True)

        Plots the energy/time response function as an image in which the shading represents
        the count rate at each time and energy

        Arguments
        ---------
        vmin       : float, optional (default=None)
                     Minimum value, corresponding to lower end of colour map. If None, calculated automatically.
        vmax       : float, optional (default=None)
                     Maximum value, corresponding to upper end of colour map. If None, calculated automatically.
        mult_scale : bool, optional (default=True)
                     If True, vmin and vmax are multiples of the maximum count rate in the response
        cmap       : string, optional (default='gray_r')
                     Name of the matplotlib colour map to use
        log_scale  : bool, optional (default=True)
                     Whether the count rate scale (shading) should be logarithmic

        Returns
        --------
        p : ImagePlot
            pyLag ImagePlot object
        """
        return ImagePlot(self.time, self.en_bins.bin_cent, self.ent, cmap=cmap, log_scale=log_scale, vmin=vmin, vmax=vmax, mult_scale=mult_scale, xlabel='Time / GM c$^{-3}$', ylabel='Energy / keV')

    def spectrum(self, time=None, index=False, from_start=True, perkev=True):
        """
        spec = pylag.ENTResponse.spectrum(time=None, index=False, from_start=True)

        Compute the time-averaged spectrum (by summing the response acros all time bins)
        or return the spectrum at a specific time, or averaged over a range of times

        Arguments
        ---------
        time       : tuple of floats, float or int, optional (default=None)
                     If None, return the time-averaged spectrum. If a tuple (tstart, tend), sum the
                     spectrum over a specific time range, or return the spectrum in the specified time bin.
        index      : bool, optional (default=False)
                     If true, interpret time as an array index instead of a time value
        from_start : bool, optional (default=True)
                     Whether the specified time is defined as the total light travel time from
                     the source to disc to image plane (default yes) or, if false, relative to the
                     zero time on the time axis.
        perkev     : bool, optional (default=True)
                     Divide the spectrum by the energy bin width for counts/sec/keV

        Returns
        --------
        spec : Spectrum
               pyLag Spectrum object containing the spectrum
        """
        if isinstance(time, tuple):
            tstarti = self.t_index(time[0], from_start)
            tendi = self.t_index(time[1], from_start)
            spec = np.sum(self.ent[:, tstarti:tendi], axis=1)
        elif time is None:
            spec = np.sum(self.ent, axis=1)
        elif index:
            spec = np.array(self.ent[:, time])
        else:
            ti = self.t_index(time, from_start)
            spec = np.array(self.ent[:, ti])

        if perkev:
            spec /= self.en_bins.x_width()

        return Spectrum(self.en_bins.bin_cent, spec)

    def time_response(self, energy=None, index=None):
        """
        resp = pylag.ENTResponse.time_response(energy=None, index=False)

        Compute the impulse response function (count rate as a function of time,
        either summed over all energies, or in a specifid energy range

        Arguments
        ---------
        energy     : tuple of floats, float or int, optional (default=None)
                     If None, sum the response over all energies. If a tuple (enstart, enend), sum the
                     response over a specific energy range, or return the response in the specified energy bin.
        index      : bool, optional (default=False)
                     If true, interpret energy as an array index instead of an energy value

        Returns
        --------
        resp : ImpulseResponse
               pyLag ImpulseResponse object containing the response function
        """
        if isinstance(energy, tuple):
            enstarti = self.en_index(energy[0])
            enendi = self.en_index(energy[1])
            resp = np.sum(self.ent[enstarti:enendi, :], axis=0)
        elif energy is None and index is None:
            resp = np.sum(self.ent, axis=0)
        elif index is not None:
            resp = np.array(self.ent[index, :])
        else:
            eni = self.en_index(energy)
            resp = np.array(self.ent[eni, :])

        return ImpulseResponse(t=self.time, r=resp)

    def avg_arrival(self):
        """
        avg_arr = pylag.ENTResponse.avg_arrival()

        Compute the average arrival time or average response time of photons in each energy bin
        (this is the simplest representation of the lag-energy spectrum).

        Returns
        --------
        avg_arr : Spectrum
                  pyLag Spectrum object containing the arrival/response time spectrum
        """
        lag = []
        for ien in range(self.ent.shape[0]):
            if np.sum(self.ent[ien, :] > 0):
                lag.append(np.average(self.time, weights=self.ent[ien, :]))
            else:
                lag.append(np.nan)
        lag = np.array(lag)

        return Spectrum(self.en_bins.bin_cent, lag, ylabel='Lag', yscale='linear')

    def std_arrival(self):
        """
        avg_arr = pylag.ENTResponse.std_arrival()

        Compute the standard deviation of the arrival times in each energy bin.

        Returns
        --------
        avg_arr : Spectrum
                  pyLag Spectrum object containing the standard deviation response time in each energy bin
        """
        lag = []
        for ien in range(self.ent.shape[0]):
            if np.sum(self.ent[ien, :] > 0):
                lag.append(weighted_std(self.time, weights=self.ent[ien, :]))
            else:
                lag.append(np.nan)
        lag = np.array(lag)

        return Spectrum(self.en_bins.bin_cent, lag, ylabel='Lag', yscale='linear')

    def lag_frequency_spectrum(self, enband1, enband2, fbins=None, Nf=None, tmax=None):
        """
        lf = pylag.ENTResponse.lag_frequency_spectrum(enband1, enband2, fbins=None, Nf=None, tmax=None)

        Compute the Fourier lag-frequency spectrum between two energy bands.

        Arguments
        ---------
        enband1  : tuple of floats
                   The energy range (enstart, enend) of the first energy band (conventionally the hard band)
        enband2  : tuple of floats
                   The energy range (enstart, enend) of the second energy band (conventionally the soft/reference band)
        fbins    : Binning, optional (default=None)
                   pyLag Binning object defining the desired frequency bins for the spectrum. If None, will be
                   calculated automatically based on the time-binning of the response function
        Nf       : int, optional (default=None)
                   If fbins=None, the desired number of (logarithmically-spaced) frequency bins
        tmax     : float, optional (default=None)
                   If not None, zero-pad the end of the response function (up to time tmax) to compute
                   the Fourier transforms (and hence lags) at lower frequencies.

        Returns
        --------
        lf : LagFrequencySpectrum
             pyLag LagFrequencySpectrum object containing the spectrum
        """
        if fbins is None:
            if Nf is None:
                raise ValueError("pylag ENTResponse lag_frequency_spectrum ERROR: Either frequency binning object or number of frequency bins required")
            minfreq = 1./(2.*(self.time.max() - self.time.min()))
            maxfreq = 1./(2.*(self.time[1]-self.time[0]))
            fbins = LogBinning(minfreq, maxfreq, Nf)

        if tmax is None:
            tmax = 1./fbins.bin_start.min()

        resp1 = self.time_response(enband1).pad(tmax)
        resp2 = self.time_response(enband2).pad(tmax)
        return LagFrequencySpectrum(fbins, lc1=resp1, lc2=resp2, calc_error=False)

    def cross_spectrum(self, enband1, enband2, fbins=None, tmax=None):
        """
        cross = pylag.ENTResponse.cross_spectrum(enband1, enband2, fbins=None, tmax=None)

        Compute the Fourier cross spectrum between two energy bands.

        Arguments
        ---------
        enband1  : tuple of floats
                   The energy range (enstart, enend) of the first energy band (conventionally the hard band)
        enband2  : tuple of floats
                   The energy range (enstart, enend) of the second energy band (conventionally the soft/reference band)
        fbins    : Binning, optional (default=None)
                   pyLag Binning object defining the desired frequency bins for the spectrum. If None, the cross
                   spectrum will be returned unbinned, using the raw frequency points in the FFT of the response.
        tmax     : float, optional (default=None)
                   If not None, zero-pad the end of the response function (up to time tmax) to compute
                   the Fourier transforms and cross spectrum at lower frequencies.

        Returns
        --------
        cross : CrossSpectrum
                pyLag CrossSpectrum object containing the spectrum
        """
        if tmax is None and fbins is not None:
            tmax = 1./fbins.bin_start.min()

        resp1 = self.time_response(enband1).pad(tmax)
        resp2 = self.time_response(enband2).pad(tmax)
        c = CrossSpectrum(lc1=resp1, lc2=resp2)

        if fbins is not None:
            return c.bin(fbins)
        else:
            return c

    def energy_lc_list(self, pad=None):
        """
        lclist = pylag.ENTResponse.energy_lc_list(pad=None)

        Returns the energy and time dependent response function as a list of response functions
        in each energy band (in the form of an EnergyLCList for a list of light curves in energy
        bands). Useful for applying light curve analysis methods to the response.

        Arguments
        ---------
        pad : float, optional (default=None)
              If not None, zero-pad the end of the response function (up to time tmax) to compute
              the Fourier transforms at lower frequencies.

        Returns
        --------
        lclist : SimEnergyLCList
                 pyLag SimEnergyLCList object, containing the response functions
        """
        lclist = []
        for ien in range(len(self.en_bins)):
            if pad is not None:
                lclist.append(ImpulseResponse(t=self.time, r=self.ent[ien, :]).pad(pad))
            else:
                lclist.append(ImpulseResponse(t=self.time, r=self.ent[ien,:]))
        return SimEnergyLCList(enmin=self.en_bins.bin_start, enmax=self.en_bins.bin_end, lclist=lclist)

    def lag_energy_spectrum(self, fmin=None, fmax=None, tmax=None):
        """
        le = pylag.ENTResponse.lag_energy_spectrum(fmin=None, fmax=None, **kwargs)

        Compute the Fourier lag-energy spectrum in a specified frequency range.

        Arguments
        ---------
        fmin  : float, optional (default=None)
                Lower bound of frequency range. If None, calculated automatically from time binning of response.
        fmax  : float, optional (default=None)
                Upper bound of frequency range. If None, calculated automatically from time binning of response.
        tmax  : float, optional (default=None)
                If not None, zero-pad the end of the response function (up to time tmax) to compute
                the Fourier transforms and cross spectrum at lower frequencies.

        Returns
        --------
        le : LagEnergySpectrum
             pyLag LagEnergySpectrum object containing the spectrum
        """
        if fmin is None:
            fmin = 1./(2.*(self.time.max() - self.time.min()))
            fmax = 1./(2.*(self.time[1]-self.time[0]))
        return LagEnergySpectrum(fmin, fmax, lclist=self.energy_lc_list(pad=tmax), calc_error=False)

    def lag_energy_frequency(self, fbins=None, Nf=100, tmax=1E6):
        """
        p = pylag.ENTResponse.lag_energy_frequency( fbins=None, Nf=100, tmax=1E6)

        Create an image plot of the lag-energy-frequency spectrum. Will use the energy bins already
        defined in the response.

        Arguments
        ---------
        fbins    : Binning, optional (default=None)
                   pyLag Binning object defining the desired frequency bins for the spectrum. If None, will be
                   calculated automatically based on the time-binning of the response function
        Nf       : int, optional (default=None)
                   If fbins=None, the desired number of (logarithmically-spaced) frequency bins
        tmax     : float, optional (default=None)
                   If not None, zero-pad the end of the response function (up to time tmax) to compute
                   the Fourier transforms (and hence lags) at lower frequencies.

        Returns
        --------
        p : ImagePlot
            pyLag ImagePlot object containing the spectrum
        """
        if Nf is None:
            raise ValueError(
                "pylag ENTResponse lag_frequency_spectrum ERROR: Either frequency binning object or number of frequency bins required")
        if fbins is None:
            minfreq = 1. / (2. * (tmax - self.time.min()))
            maxfreq = 1. / (2. * (self.time[1] - self.time[0]))
            fbins = LogBinning(minfreq, maxfreq, Nf)
        lagfreq = np.zeros((len(fbins), len(self.en_bins)))
        for i in range(len(self.en_bins)):
            _, lagfreq[:,i] = self.time_response(index=i).pad(tmax).lagfreq(fbins)
        #return fbins.bin_cent, self.en_bins.bin_cent, lagfreq

        lagfreq[np.isnan(lagfreq)] = 0

        return ImagePlot(fbins.bin_cent, self.en_bins.bin_cent, lagfreq.T, log_scale=False, vmin=lagfreq.min(), vmax=lagfreq.max(), mult_scale=False, xscale='log', yscale='log')

    def weight_arf(self, arf):
        """
        ent_weight = pylag.entresponse.ENTResponse.weight_arf(arf)

        Weight the count rate in each energy band by an effective area curve (Arf) to simulate count rates
        seen in telescopes with realistic effective area curves. Note that this will only scale the count rate
        in each band by the fraction of the effective area in those bands, it will not apply the absolute count
        rate scaling. To apply this, you should normalise the whole response for the desired count rate for whatever
        luminosity you require.

        :param arf: Arf object contraining the effective area curve to be applied
        :return: ent_weight: ENTResponse with effective area weighting applied
        """
        ent_weight = np.array(self.ent)
        bin_frac = arf.bin_fraction(self.en_bins, enrange=(self.en_bins.bin_start.min(), self.en_bins.bin_end.max()))
        for ien in range(len(self.en_bins)):
            ent_weight[ien, :] *= bin_frac[ien]
        return ENTResponse(t=self.time, en_bins=self.en_bins, ent=ent_weight, tstart=self.tstart)

    def simulate_lc_list(self, tmax, plslope, std, lcmean, add_noise=False, rebin_time=None, lc=None, arf=None):
        """
        lclist = pylag.ENTResponse.simulate_lc_list(tmax, plslope, std, lcmean, add_noise=False, rebin_time=None, lc=None)

        Simulate a set of light curves that would be observed based upon this response function.
        The light curves are generated based on a random driving light curve, following Timmer & Konig (1995),
        which is then convolved by the response function in each energy band. Light curves will use the energy
        bins defined in the response.

        Arguments
        ---------
        tmax       : float
                     Length of light curve (i.e. maximum time) to generate
        plslope    : float or tuple of floats
                     The slope of the power spectrum of the driving light curve, or if a tuple,
                     the broken power law to use (slope1, fbreak, slope2)
        std        : float
                     The standard deviation count rate of the driving light curve to be generated
        lcmean     : float
                     The mean count rate of the driving light curve to be generated
        add_noise  : bool, optional (default=False)
                     If True, add Poisson noise to the generated light curves
        rebin_time : float, optional (default=None)
                     Initial light curves will be generated using the time binning in the response.
                     If rebin_time is set, the light curves will be rebinned to this time bin width,
                     This is useful for accurately modelling time lags shorter than or comparable to
                     the time bin size of an observation.
        lc         : LightCurve, optional (default=None)
                     If set, use this as the driving light curve, instead of generating a random one.
        arf        : Arf, optional (default-None)
                     If set, weight the count rate in each band by the effective area curve

        Returns
        --------
        lclist : SimEnergyLCList
                 pyLag SimEnergyLCList object, containing the simulated light curves
        """
        lclist = []
        if lc is None:
            lc = SimLightCurve(self.time[1] - self.time[0], tmax, plslope, std, lcmean)

        # weight each energy band by the ARF if we're using one, otherwise just use self
        ent = self.weight_arf(arf) if arf is not None else self
        # and make sure the response is normalised so lcmean will actually set the summed mean count rate
        ent = ent.norm()

        for ien in range(len(self.en_bins)):
            enlc = ent.time_response(index=ien).convolve(lc)
            if rebin_time is not None:
                enlc = enlc.rebin(rebin_time)
            if add_noise:
                enlc = enlc.add_noise()
            lclist.append(enlc)
        return SimEnergyLCList(enmin=self.en_bins.bin_start, enmax=self.en_bins.bin_end, lclist=lclist, base_lc=lc)

    def convolve_spectrum(self, spectrum, enbins, binspec=None, line_en=6.4, **kwargs):
        """
        ents = pylag.ENTResponse.convolve_spectrum(spectrum, enbins, binspec=None, line_en=6.4, **kwargs)

        Convolve the response function in each time bin with a spectrum. This is used to transform
        a single line response function to the full spectral response function, by convolving with
        the rest frame reflection spectrum (e.g. Xillver or REFLIONX).

        Arguments
        ---------
        spectrum   : FITSSpectralModel
                     pyLag FITSSpectralModel object, containing the spectrum to convolve with the response.
        enbins     : Binning
                     pyLag Binning object describing the energy bins to be used in the final convolved
                     response function.
        binspec    : Binning, optional (default=True)
                     If set, the pyLag Binning object to be used to rebin the input spectrum before running
                     the convolution (to speed up the calculation). These bins should be finer than enbins.
        line_en    : float, optional (default=6.4)
                     The rest-frame energy of the line described in the original response. This is usually
                     6.4 if the response function is calculated for a 6.4keV iron K line.
        **kwargs   : passed to spectrum.spectrum(). These arguments are usually the parameters of the spectrum
                     to be used here (e.g. A_Fe=1., Gamma=2., logXi=1.5).

        Returns
        --------
        ents : ENTResponse
               pyLag SimEnergyLCList object, containing the full spectral (i.e. the convolved) response.
        """
        spec = spectrum.spectrum(**kwargs)

        if binspec is not None:
            spec = Spectrum(binspec.bin_cent, binspec.bin(spec.en, spec.spec))

        ent_conv = np.zeros((len(enbins), self.ent.shape[1]))
        for i in range(self.ent.shape[1]):
            ent_conv[:,i] = convolve_spectrum(spec.en, spec.spec, self.en_bins.bin_cent/line_en, self.ent[:,i], enbins)
        return ENTResponse(en_bins=enbins, t=self.time, ent=ent_conv, tstart=self.tstart)

    def write_fits(self, filename):
        """
        pylag.ENTResponse.write_fits(filename)

        Write the response function to a FITS file in a format that can be re-loaded by this class.
        The response is simply stored in the primary HDU.
        """
        hdu = pyfits.PrimaryHDU()

        hdu.header['EN0'] = self.en_bins.bin_start[0]
        hdu.header['ENMAX'] = self.en_bins.bin_end[-1]
        hdu.header['NEN'] = len(self.en_bins)
        hdu.header['ENLOG'] = self.logbin_en

        hdu.header['T0'] = self.t0
        hdu.header['DT'] = self.dt
        hdu.header['NT'] = len(self.time)
        hdu.header['TSTART'] = self.tstart

        hdu.data = self.ent

        hdu.writeto(filename)

    def __getitem__(self, index):
        return ENTResponse(en_bins=self.en_bins, t=self.time[index], ent=self.ent[:, index], tstart=self.tstart)

    def __add__(self, other):
        if not isinstance(other, ENTResponse):
            return NotImplemented

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be added!")

        sum_ent = self.ent + other.ent

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=sum_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __iadd__(self, other):
        if not isinstance(other, ENTResponse):
            raise AssertionError("Can only add ENTResponse objects!")

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be added!")

        self.ent += other.ent
        return self

    def __sub__(self, other):
        if not isinstance(other, ENTResponse):
            return NotImplemented

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be subtracted!")

        sub_ent = self.ent - other.ent

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=sub_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __isub__(self, other):
        if not isinstance(other, ENTResponse):
            return NotImplemented

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be subtracted!")

        self.ent -= other.ent
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            mul_ent = self.ent * other
        elif isinstance(other, ENTResponse):
            mul_ent = self.ent * other.ent
        else:
            return NotImplemented

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=mul_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.ent *= other
        elif isinstance(other, ENTResponse):
            self.ent *= other.ent
        else:
            return NotImplemented

        return self

    def __div__(self, other):
        if isinstance(other, (int, float)):
            div_ent = self.ent / other
        elif isinstance(other, ENTResponse):
            div_ent = self.ent / other.ent
        else:
            return NotImplemented

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=div_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __idiv__(self, other):
        if isinstance(other, (int, float)):
            self.ent /= other
        elif isinstance(other, ENTResponse):
            self.ent /= other.ent
        else:
            return NotImplemented

        return self

    def __str__(self):
        return "<pylag.entresponse.ENTResponse: (%d, %d) energy, time channels>" % (self.ent.shape[0], self.ent.shape[1])

    def __repr__(self):
        return "<pylag.entresponse.ENTResponse: (%d, %d) energy, time channels>" % (self.ent.shape[0], self.ent.shape[1])


class ENTResponseSet(object):
    def __init__(self, response_file):
        try:
            import h5py
        except ModuleNotFoundError:
            raise ModuleNotFoundError('ENTResponseSet requires h5py to be installed')

        with h5py.File(response_file) as hdf:
            en0 = hdf['responses'].attrs['en0']
            enmax = hdf['responses'].attrs['enmax']
            Nen = hdf['responses'].attrs['Nen']
            self.logbin_en = bool(hdf['responses'].attrs['logbin_en'])
            self.en_bins = LogBinning(en0, enmax, Nen) if self.logbin_en else LinearBinning(en0, enmax, Nen)

            t0 = hdf['responses'].attrs['t0']
            dt = hdf['responses'].attrs['dt']
            Nt = hdf['responses'].attrs['Nt']
            self.time = t0 + dt * np.arange(0, Nt, 1)

            self.heights = np.array(hdf['heights'])
            self.incl = np.array(hdf['incl'])
            self.tstart = np.array(hdf['tstart'])
            self.responses = np.array(hdf['responses'])

    def get_response(self, incl, h):
        i_num = np.argmin(np.abs(self.incl - incl))
        h_num = np.argmin(np.abs(self.heights - h))

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=self.responses[i_num, h_num],
                           logbin_en=self.logbin_en, tstart=self.tstart[i_num, h_num])


class RadiusENTResponse(object):
    def __init__(self, response_file):
        try:
            import h5py
        except ModuleNotFoundError:
            raise ModuleNotFoundError('RadiusENTResponse requires h5py to be installed')

        hdf = h5py.File(response_file)
        en0 = hdf.attrs['en0']
        enmax = hdf.attrs['enmax']
        Nen = hdf.attrs['Nen']
        self.logbin_en = bool(hdf.attrs['logbin_en'])
        self.en_bins = LogBinning(en0, enmax, Nen) if self.logbin_en else LinearBinning(en0, enmax, Nen)

        t0 = hdf.attrs['t0']
        dt = hdf.attrs['dt']
        Nt = hdf.attrs['Nt']
        self.time = t0 + dt * np.arange(0, Nt, 1)

        self.tstart = hdf.attrs['tstart']
        self.line_en = hdf.attrs['line_en']

        r0 = hdf.attrs['r0']
        r_max = hdf.attrs['r_max']
        Nr = hdf.attrs['Nr']
        logbin_r = hdf.attrs['logbin_r']
        self.r_bins = LogBinning(r0, r_max, Nr) if logbin_r else LinearBinning(r0, r_max, Nr)

        self.responses = hdf['radius_response']

    def get_response(self, r):
        r_num = self.r_bins.bin_index(r)

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=self.responses[r_num],
                           logbin_en=self.logbin_en, tstart=self.tstart)

    def response_list(self):
        return [ENTResponse(en_bins=self.en_bins, t=self.time, ent=resp,
                           logbin_en=self.logbin_en, tstart=self.tstart) for resp in self.responses]

    def sum_radii(self, r_min, r_max):
        r_min_num = self.r_bins.bin_index(r_min)
        r_max_num = self.r_bins.bin_index(r_max)

        ent_sum = np.sum(self.responses[r_min_num:r_max_num+1], axis=0)

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=ent_sum,
                           logbin_en=self.logbin_en, tstart=self.tstart)

    def rebin_time(self, bins=None, dt=None, Nt=None):
        if bins is None:
            if dt is not None:
                bins = LinearBinning(self.time.min(), self.time.max(), step=dt)
            if Nt is not None:
                bins = LinearBinning(self.time.min(), self.time.max(), num=Nt)

        rebin_responses = np.zeros((self.responses.shape[0], self.responses.shape[1], len(bins)))
        for i in range(self.responses.shape[0]):
            for j in range(self.responses.shape[1]):
                rebin_responses[i,j,:] = bins.bin(self.time, self.responses[i,j,:])
        self.responses = rebin_responses

        self.dt = bins.bin_end[0] = bins.bin_start[0]
        self.t0 = bins.bin_start[0]
        self.Nt = len(bins)
        self.time = bins.bin_start

    def rebin_energy(self, bins=None, Nen=None):
        if bins is None:
            bins = LogBinning(self.en_bins.min(), self.en_bins.max(), Nen)

        rebin_responses = np.zeros((self.responses.shape[0], len(bins), self.responses.shape[2]))
        for i in range(self.responses.shape[0]):
            for k in range(self.responses.shape[2]):
                rebin_responses[i,:,k] = bins.bin(self.en_bins.bin_cent, self.responses[i,:,k], statistic='sum')
        self.responses = rebin_responses

        self.en_bins = bins
        self.logbin_en = isinstance(bins, LogBinning)

    def convolve_iongrad_spectra(self, spectrum, enbins, binspec=None, ion_func=powerlaw, ion_args=None, **kwargs):
        r = self.r_bins.bin_cent
        xi = np.log10(ion_func(r, **ion_args))

        try:
            xi_vals = np.trim_zeros(spectrum.param_tab_vals[spectrum.params.index('logXi')], 'b')
            xi[xi < xi_vals.min()] = xi_vals.min()
            xi[xi > xi_vals.max()] = xi_vals.max()
        except:
            raise ValueError("Could not find logXi parameter in table model. Are you using xillver?")

        # note we use the normalised rest-frame reflection spectra and assume the photon count per
        # radial bin comes from the response functions

        r_ent = [ENTResponse(en_bins=self.en_bins, t=self.time, ent=self.responses[r_num],
                           logbin_en=self.logbin_en, tstart=self.tstart).convolve_spectrum(spectrum,
                               enbins, binspec, self.line_en, norm_spec=True, logXi=x, **kwargs) for r_num, x in enumerate(xi)]

        return np.sum(r_ent)

    def convolve_densitygrad_spectra(self, spectrum, enbins, binspec=None, dens_func=powerlaw, dens_args=None,
                                     ion_func=powerlaw, ion_args=None, **kwargs):
        r = self.r_bins.bin_cent
        xi = np.log10(ion_func(r, **ion_args))
        dens = np.log10(dens_func(r, **dens_args))

        try:
            xi_vals = np.trim_zeros(spectrum.param_tab_vals[spectrum.params.index('logXi')], 'b')
            xi[xi < xi_vals.min()] = xi_vals.min()
            xi[xi > xi_vals.max()] = xi_vals.max()
        except:
            raise ValueError("Could not find logXi parameter in table model. Are you using xillver?")

        try:
            dens_vals = np.trim_zeros(spectrum.param_tab_vals[spectrum.params.index('Dens')], 'b')
            dens[dens < dens_vals.min()] = dens_vals.min()
            dens[dens > dens_vals.max()] = dens_vals.max()
        except:
            raise ValueError("Could not find Dens parameter in table model. Are you using xillverD?")

        # note we use the normalised rest-frame reflection spectra and assume the photon count per
        # radial bin comes from the response functions

        r_ent = [ENTResponse(en_bins=self.en_bins, t=self.time, ent=self.responses[r_num],
                           logbin_en=self.logbin_en, tstart=self.tstart).convolve_spectrum(spectrum,
                               enbins, binspec, self.line_en, norm_spec=True, Dens=d, logXi=x, **kwargs) for r_num, (d, x) in enumerate(zip(dens, xi))]

        return np.sum(r_ent)
