"""
pylag.simulator

Tools for simulating light curves and X-ray timing measurements

v1.0 - 27/03/2017 - D.R. Wilkins
"""
from .lightcurve import *

import numpy as np
import scipy.fftpack
import scipy.signal


class SimLightCurve(LightCurve):
    """
    pylag.SimLightCurve

    Class fcor simulating the observation of an X-ray light curve. A random time
    series is generated with a specified power spectrum following the method of
    Timmer and Konig; the amplitude of the Fourier transorm of the time series at
    each sample frequency is set according to the power spectrum and the phase of
    each component is drawn at random from a uniform distribution. The time series
    is computed from the inverse FFT of this.

    Once a random time series is generated, it is possible to add random noise by
    drawing the measured photon count in each time bin from a Poisson distribution
    using the add_noise() method.

    Constructor: pylag.SimLightCurve(dt=10., tmax=1000., plslope=2.0, std=0.5, 
                                     lcmean=1.0, t=None, r=None, e=None, gtzero=True)

    Constructor Arguments
    ---------------------
    dt      : float, optional (default=10.)
              The time bin size, in seconds, in the generated light curve
    tmax	: float, optional (default=1000.)
              The length, in seconds, of the light curve
    plslope : float, optional (default=2.)
              Slope of the power law (P = f^-a) power spectral density
    std		: float, optional (default=1.)
              Standard deviation of the computed light curve. After the random
              time series is generated with zero mean it is rescaled to the
              specified standard deviation
    lcmean	: float, optional (default=1.)
              Mean count rate of the computed light curve. After the random time
              series is generated and scaled with zero mean, it is shifted to the
              specified mean count rate
    t	    : ndarray, optional (default=None)
              If set, a light curve is not calculated but created from an existing
              time series. This is the time axis.
    r	    : ndarray, optional (default=None)
              If set, a light curve is not calculated but created from an existing
              time series. This is the count rate.
    e	    : ndarray, optional (default=None)
              If set, the error in the count rate. If not set, the error is
              calculated as sqrt(N)
    gtzero  : Boolean, optional (default=True)
              Force all points in the light curve to have count rate greater
              than zero. Set all points below zero to zero
    """
    def __init__(self, dt=10., tmax=1000., plslope=2.0, std=0.5, lcmean=1.0, t=None, r=None, e=None, gtzero=True, exp=True):
        if t is None and r is None:
            t = np.arange(0, tmax, dt)
            r = self.calculate(t, plslope, std, lcmean, gtzero=gtzero, exp=exp)
        if e is None:
            e = np.sqrt(r)
            #e = np.sqrt(r / dt)
        LightCurve.__init__(self, t=t, r=r, e=e)

    def calculate(self, t, plslope, std, lcmean, plnorm=1., gtzero=True, exp=False):
        """
        pylag.SimLightCurve.calculate(t, plslope, lcmean, std, plnorm=1.)

        Simulate a random light curve with a specified power spectrum using the
        Timmer & Konig method; the amplitude of the Fourier transorm of the time
        series at each sample frequency is set according to the power spectrum
        and the phase of each component is drawn at random from a uniform
        distribution. The time series is computed from the inverse FFT of this.

        Arguments
        ---------
        t	  	: ndarray
                  Time axis upon which the light curve will be calculated
        plslope : float,
                  Slope of the power law (P = f^-a) power spectral density
        std		: float
                  Standard deviation of the computed light curve. After the random
                  time series is generated with zero mean it is rescaled to the
                  specified standard deviation
        lcmean	: float
                  Mean count rate of the computed light curve. After the random time
                  series is generated and scaled with zero mean, it is shifted to the
                  specified mean count rate
        plnorm  : float, optional (default=1.)
                  Normalisation of the power law power spectral density at zero
                  frequency; thsi can usually be set to 1 as the resulting light
                  curve is rescaled to produce the desired standard deviation
        gtzero  : Boolean, optional (default=True)
                  Force all points in the light curve to have count rate greater
                  than zero. Set all points below zero to zero
        """
        # sample frequencies
        freq = scipy.fftpack.fftfreq(len(t), d=t[1] - t[0])
        # normalise the power law PSD
        plnorm = plnorm / ((2 * np.pi * freq[1]) ** (-plslope))
        # build the Fourier transform of the light curve
        # amplitude at each frequency is according to a power law, phase is random
        # note we use abs(freq) to populate the negative and positive frequencies
        # since a real light curve has a symmetric FFT. Also skip f=0
        ampl = np.sqrt(0.5 * plnorm * (2 * np.pi * np.abs(freq[1:])) ** (-plslope))
        ampl = np.insert(ampl, 0, plnorm)  # add the zero frequency element
        phase = 2 * np.pi * np.random.rand(len(freq))

        # put the Fourier transform together then invert it to get the light curbe
        ft = ampl * np.exp(1j * phase)
        r = np.real(scipy.fftpack.ifft(ft))

        if exp:
            r = np.exp(r)

        # normalise and shift the light curve to get the desired mean and stdev
        r = std * r / np.std(r)
        r = r - np.mean(r) + lcmean
        # don't let any points drop below zero (the detector will see nothing here)
        if gtzero:
            r[r < 0] = 0

        return r

    def add_noise(self):
        """
        lc = pylag.SimLightCurve.add_noise

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
        error = np.sqrt(self.rate)

        return SimLightCurve(t=self.time, r=rate, e=error)


def resample_light_curves(lclist, resamples=1):
    """
    new_lclist = pylag.resample_light_curves(lclist, resamples=1)

    Take a list of LightCurve objects or a list of lists of multiple light curve
    segments in each energy band (as used for a lag-energy or covariance spectrum)
    and resample them by replacing each data point with one drawn from a Poisson
    distribution using the orignal number of counts as the mean value.

    The returned list of light curves retains the original structure of the list
    passed in.

    Can produce multiple resamplings. If more than one resampling is produced,
    a list is returned, each of which is a list bearing the original structure.

    Arguments
    ---------
    lclist    : list or list-of-lists of LightCurves
                The list of light curves to be resampled
    resamples : int, optional (default=1)
                The number of resampled light curve sets to produce

    Returns
    -------
    new_lclist : list, list-of-lists or list-of-lists-of-lists of LightCurves
                 If one resampling is selected, this will be the list of
                 resampled light curves. retaining the structure of the input
                 list. If multiple resamplings are selected, this will be a list
                 of the resampled lists (the first index is the resampling)
    """
    if resamples > 1:
        lclist_set = []

    for i in range(resamples):
        new_lclist = []

        if isinstance(lclist[0], list):
            for en_lclist in lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    temp_lc = SimLightCurve(t=lc.time, r=lc.rate, e=lc.error)
                    temp_lc.rate[temp_lc.rate < 0] = 0
                    new_lclist[-1].append(temp_lc.add_noise())

        elif isinstance(lclist[0], LightCurve):
            for lc in lclist:
                temp_lc = SimLightCurve(t=lc.time, r=lc.rate, e=lc.error)
                temp_lc.rate[temp_lc.rate < 0] = 0
                new_lclist.append(temp_lc.add_noise())

        if resamples > 1:
            lclist_set.append(new_lclist)

    if resamples > 1:
        return lclist_set
    else:
        return new_lclist

def resample_enlclist(lclist, resamples=1):
    """
    new_lclist = pylag.resample_light_curves(lclist, resamples=1)

    Take a list of LightCurve objects or a list of lists of multiple light curve
    segments in each energy band (as used for a lag-energy or covariance spectrum)
    and resample them by replacing each data point with one drawn from a Poisson
    distribution using the orignal number of counts as the mean value.

    The returned list of light curves retains the original structure of the list
    passed in.

    Can produce multiple resamplings. If more than one resampling is produced,
    a list is returned, each of which is a list bearing the original structure.

    Arguments
    ---------
    lclist    : list or list-of-lists of LightCurves
                The list of light curves to be resampled
    resamples : int, optional (default=1)
                The number of resampled light curve sets to produce

    Returns
    -------
    new_lclist : list, list-of-lists or list-of-lists-of-lists of LightCurves
                 If one resampling is selected, this will be the list of
                 resampled light curves. retaining the structure of the input
                 list. If multiple resamplings are selected, this will be a list
                 of the resampled lists (the first index is the resampling)
    """
    if resamples > 1:
        lclist_set = []

    for i in range(resamples):
        new_lclist = []

        if isinstance(lclist[0], list):
            for en_lclist in lclist.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    temp_lc = SimLightCurve(t=lc.time, r=lc.rate, e=lc.error)
                    temp_lc.rate[temp_lc.rate < 0] = 0
                    new_lclist[-1].append(temp_lc.add_noise())

        elif isinstance(lclist[0], LightCurve):
            for lc in lclist.lclist:
                temp_lc = SimLightCurve(t=lc.time, r=lc.rate, e=lc.error)
                temp_lc.rate[temp_lc.rate < 0] = 0
                new_lclist.append(temp_lc.add_noise())

        if resamples > 1:
            lclist_set.append( EnergyLCList(enmin=lclist.enmin, enmax=lclist.enmax, lclist=new_lclist) )

    if resamples > 1:
        return lclist_set
    else:
        return EnergyLCList(enmin=lclist.enmin, enmax=lclist.enmax, lclist=new_lclist)


class ImpulseResponse(LightCurve):
    """
    pylag.ImpulseResponse

    Class for storing an impulse response function with which a light curve can
    be convolved to e.g calculate the reverberation response.

    This class is derived from LightCurve and all operations that can be performed
    on a LightCurve can be performed on the response function.

    Note that the response function does not have error bars.

    Member Variables
    ----------------
    time : ndarray
           The time axis of the response function
    rate : ndarray
           The response count rate as a function of time

    Constructor: pylag.ImpulseResponse(dt=10., tmax=1000., t=None, r=None)

    Constructor Arguments
    ---------------------
    dt      : float, optional (default=10.)
              The time bin size, in seconds, of the response function
    tmax	: float, optional (default=1000.)
              The length, in seconds, of the full response function
    t	    : ndarray, optional (default=None)
              If set, a response functionis created from an existing
              time series. This is the time axis.
    r	    : ndarray, optional (default=None)
              If set, a response function created from an existing
              time series. This is the count rate.
    """
    def __init__(self, dt=10., tmax=1000., t=None, r=None):
        if t is None and r is None:
            t = np.arange(0, tmax, dt)
            r = np.zeros(t.shape)
        e = np.zeros(t.shape)   # the response function doesn't have error bars
        LightCurve.__init__(self, t=t, r=r, e=e)

    def norm(self):
        """
        pylag.ImpulseResponse.norm()

        Normalises the response function such that the sum of all points is unity
        """
        self.rate = self.rate / self.rate.sum()

    def convolve(self, lc):
        """
        clc = pylag.ImpulseResponse.convolve(lc)

        Convolves a light curve with this response.

        The returned light curve contains only the 'valid' part of the convolution
        (i.e. the part that requires no zero padding, starting the length of the
         response function after te start of the original light curve)

        Time axis alignes with input light curve.

        Parameters
        ----------
        lc : LightCurve
             LightCurve object which si to be convolved with this response

        Returns
        -------
        clc : SimLightCurve
              SimLightCurve object that contains the new light curve, convolved
              with the response
        """
        #t = np.arange(lc.time.min() + len(self)*self.dt, lc.time.max()+self.dt, self.dt)
        t = lc.time[len(self)-1:]
        r = scipy.signal.convolve(lc.rate, self.rate, mode='valid')
        return SimLightCurve(t=t, r=r)


class GaussianResponse(ImpulseResponse):
    """
    pylag.GaussianResponse

    Gaussian impulse response function (derived from ImpulseResponse)

    Member Variables
    ----------------
    time : ndarray
           The time axis of the response function
    rate : ndarray
           The response count rate as a function of time

    Constructor: pylag.GaussianResponse(mu, sigma, dt=10., tmax=None)

    Constructor Arguments
    ---------------------
    mu      : float
              Mean arrival time of the Gaussian impulse response profile
    sigma   : float
              Standard deviation of the Gaussian profile
    dt      : float, optional (default=10.)
              The time bin size, in seconds, of the response function
    tmax	: float, optional (default=None)
              The length, in seconds, of the full response function.
              If None, will be set to 3*sigma after the mean time
    """
    def __init__(self, mu, sigma, dt=10., tmax=None):
        if tmax is None:
            tmax = mu + 3*sigma
            tmax -= tmax % dt
        t = np.arange(0, tmax, dt)
        r = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(t - mu) ** 2 / (2. * sigma ** 2))
        r = r / r.sum()
        ImpulseResponse.__init__(self, t=t, r=r)


class DeltaResponse(ImpulseResponse):
    """
    pylag.DeltaResponse

    Delta function impulse response function, i.e. a time shift
    (derived from ImpulseResponse)

    Member Variables
    ----------------
    time : ndarray
           The time axis of the response function
    rate : ndarray
           The response count rate as a function of time

    Constructor: pylag.DeltaResponse(t0, dt=10., tmax=None)

    Constructor Arguments
    ---------------------
    t0      : float
              Arrival time of the delta function response
    dt      : float, optional (default=10.)
              The time bin size, in seconds, of the response function
    tmax	: float, optional (default=None)
              The length, in seconds, of the full response function.
              If None, will be set to one time bin after t0
    """
    def __init__(self, t0, dt=10., tmax=None):
        if tmax is None:
            tmax = t0 + 2*dt
        t = np.arange(0, tmax, dt)
        tix = int(t0 / dt)
        r = np.zeros(t.shape)
        r[tix] = 1
        ImpulseResponse.__init__(self, t=t, r=r)


class TopHatResponse(ImpulseResponse):
    """
    pylag.TopHatResponse

    Top hat impulse response function (derived from ImpulseResponse)

    Member Variables
    ----------------
    time : ndarray
           The time axis of the response function
    rate : ndarray
           The response count rate as a function of time

    Constructor: pylag.TopHatResponse(mu, sigma, dt=10., tmax=None)

    Constructor Arguments
    ---------------------
    tstart  : float
              Start time of the top hat profile
    tend    : float
              End time of the top hat profile
    dt      : float, optional (default=10.)
              The time bin size, in seconds, of the response function
    tmax	: float, optional (default=None)
              The length, in seconds, of the full response function.
              If None, set to one time bin after tend
    """
    def __init__(self, tstart, tend, dt=10., tmax=None):
        if tmax is None:
            tmax = tend + dt
        t = np.arange(0, tmax, dt)
        r = np.zeros(t.shape)
        r[(t >= tstart) & (t < tend)] = 1
        r = r / r.sum()
        ImpulseResponse.__init__(self, t=t, r=r)
