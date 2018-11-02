"""
pylag.simulator

Tools for simulating light curves and X-ray timing measurements

v1.0 - 27/03/2017 - D.R. Wilkins
"""
from .lightcurve import *
from .cross_spectrum import *
from .lag_frequency_spectrum import LagFrequencySpectrum
from .lag_energy_spectrum import LagEnergySpectrum

import numpy as np
import scipy.fftpack
import scipy.signal


def psd_powerlaw(freq, slope1, fbreak=None, slope2=None):
    psd = np.zeros(freq.shape)
    if fbreak is None and slope2 is None:
        psd = (2 * np.pi * np.abs(freq)) ** (-slope1 / 2.)
    else:
        psd[np.abs(freq) <= fbreak] = (2 * np.pi * np.abs(freq[np.abs(freq) <= fbreak])) ** (-slope1 / 2.)
        psd[np.abs(freq) > fbreak] = (2 * np.pi * fbreak) ** ((slope2 - slope1) / 2.) * (
                2 * np.pi * np.abs(freq[np.abs(freq) > fbreak])) ** (-slope2 / 2.)

    return psd


def psd_sho(freq, S0, f0, Q):
    return np.sqrt(2./np.pi) * (S0 * f0**4) / ((freq**2 - f0**2)**2 + freq**2 * (f0/Q)**2)


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
    def __init__(self, dt=10., tmax=1000., psd_param=(2), std=0.5, lcmean=1.0, t=None, r=None, e=None, gtzero=True, lognorm=False, psdfn=psd_powerlaw, oversample=1.):
        if t is None and r is None:
            t = np.arange(0, oversample*tmax, dt)
            r = self.calculate(t, psd_param, std, lcmean, gtzero=gtzero, lognorm=lognorm, psdfn=psdfn)
            if oversample > 1:
                t = np.arange(0, tmax, dt)
                itstart = int((oversample - 1.) * tmax / (2*dt))
                itend = itstart + len(t)
                r = r[itstart:itend]
        if e is None:
            #e = np.sqrt(r)
            e = np.sqrt(r / dt)
        LightCurve.__init__(self, t=t, r=r, e=e, zero_nan=False)

    def calculate(self, t, psd_param, std, lcmean, plnorm=1., gtzero=True, lognorm=False, psdfn=psd_powerlaw):
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
        Nf = len(freq)

        if not isinstance(psd_param, tuple):
            psd_param = tuple([psd_param])
        psd = psdfn(freq, *psd_param)

        re = np.zeros(freq.shape)
        imag = np.zeros(freq.shape)

        # randomly draw the real part of the Fourier transform from a normal distribution
        # total magnitude of the FT is set by the desired power spectrum
        re[1:int(Nf/2)] = np.random.randn(len(freq[1:int(Nf/2)])) * psd[1:int(Nf/2)]
        if Nf % 2 == 0:
            # if total number of frequency bins is even, we have the Nyquist frequency with no positive counterpart
            # FT at the Nyquist frequency is real
            re[int(Nf / 2)] = np.random.randn() * psd[int(Nf / 2)]
            re[int(Nf/2)+1:] = np.flip(re[1:int(Nf/2)], axis=0)
        else:
            re[int(Nf / 2)] = np.random.randn() * psd[int(Nf / 2)]
            re[int(Nf / 2)+1:] = np.flip(re[1:int(Nf / 2)+1], axis=0)
        re[0] = plnorm * np.random.randn()
        # and the imaginary part
        imag[1:int(Nf/2)] = np.random.randn(len(freq[1:int(Nf/2)])) * psd[1:int(Nf/2)]
        if Nf % 2 == 0:
            # the FT at negative frequencies is the complex conjugate of that at positive frequencies
            # if total number of frequency bins is even, we have the Nyquist frequency with no positive counterpart
            # FT at the Nyquist frequency is real
            imag[int(Nf/2)+1:] = -1.*np.flip(imag[1:int(Nf/2)], axis=0)
        else:
            imag[int(Nf / 2)] = np.random.randn() * psd[int(Nf / 2)]
            imag[int(Nf / 2)+1:] = -1.*np.flip(imag[1:int(Nf / 2)+1], axis=0)

        # put the Fourier transform together then invert it to get the light curve
        ft = re + (1j * imag)
        r = np.real(scipy.fftpack.ifft(ft))

        # normalise and shift the light curve to get the desired mean and stdev
        # for lognorm light curves, this is mean and std of log(count rate)
        r = std * r / np.std(r)
        r = r - np.mean(r) + lcmean

        if lognorm:
            r = np.exp(r)

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

    def add_gaps(self, period, length, gap_value=0):
        freq = 1. / period
        duty = (period - length) / period

        window = scipy.signal.square(2*np.pi*freq*self.time, duty=duty)
        #window[window < 0] = gap_value

        rate = np.array(self.rate)
        error = np.array(self.error)

        rate[window < 0] = gap_value
        error[window < 0] = gap_value

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


class SimEnergyLCList(EnergyLCList):
    def __init__(self, enmin=None, enmax=None, lclist=None, base_lc=None, **kwargs):
        if lclist is not None and enmin is not None and enmax is not None:
            self.lclist = lclist
            self.enmin = np.array(enmin)
            self.enmax = np.array(enmax)

        self.base_lc = base_lc

        self.en = 0.5*(self.enmin + self.enmax)
        self.en_error = self.en - self.enmin

    def add_noise(self):
        new_lclist = []

        if isinstance(self.lclist[0], list):
            for en_lclist in self.lclist:
                new_lclist.append([])
                for lc in en_lclist:
                    new_lclist[-1].append(lc.add_noise())

        elif isinstance(self.lclist[0], LightCurve):
            for lc in self.lclist:
                new_lclist.append(lc.add_noise())

        return SimEnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=new_lclist)


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

    def avg_arrival(self):
        """
        avg_arr = pylag.ImpulseResponse.avg_arrival()

        Returns the average arrival time of this response function
        (i.e. the weighted average of the time according to count rate)

        Returns
        -------
        avg_arr : float
                  Average arrival time
        """
        return np.average(self.time, weights=self.rate)

    def lagfreq(self, fbins=None):
        """
        freq, lag = pylag.ImpulseResponse.lagfreq()

        Returns the time lag associated with Fourier frequencies defined by fbins

        Parameters
        ----------
        fbins : Binning, optional (default=None)
                If a Binning object is passed, the Fourier transform will be binned
                before computing the phase

        Returns
        -------
        freq : ndarray
               Sample frequencies or the central frequency of each bin
        lag  : ndarray
               The time lag associated with each frequency or bin
        """
        f, ft = self.ft()
        if fbins is not None:
            ft_bin = fbins.bin(f, ft)
            freq = fbins.bin_cent
            lag = np.angle(ft_bin) / (2*np.pi*fbins.bin_cent)
        else:
            freq = f
            lag = np.angle(ft) / (2*np.pi*f)
        return freq, lag

    def pad(self, new_tmax):
        """
        padded_resp = pylag.ImpulseResponse.pad()

        Pads the end of the response function, linearly ramping to zero, to extend the
        time axis, for getting to lower frequencies in FFT

        Parameters
        ----------
        new_tmax : float
                   New end to time axis to pad to

        Returns
        -------
        padded_resp : ndarray
                      Padded impulse response function
        """
        pad_t = np.arange(self.time.min(), new_tmax, self.time[1]-self.time[0])
        pad_r = np.pad(self.rate, (0, len(pad_t) - len(self.time)), 'linear_ramp')
        resp = ImpulseResponse(t=pad_t, r=pad_r)
        resp.__class__ = self.__class__
        return resp


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


class SimLagFrequencySpectrum(LagFrequencySpectrum):
    def __init__(self, bins, ent, enband1, enband2, rate, tbin=10, tmax=1E5, add_noise=True, sample_errors=True, nsamples=100, plslope=2,
                 std=1., lc=None, oversample=1.):
        self.freq = bins.bin_cent
        self.freq_error = bins.x_error()

        if lc is None:
            lc = SimLightCurve(ent.time[1] - ent.time[0], tmax, plslope, std, lcmean=1., oversample=oversample)

        self.base_lc = lc

        fullrate = np.sum(ent.time_response())

        resp1 = ent.time_response(energy=enband1)
        norm1 = rate * np.sum(resp1) / fullrate
        lc1 = resp1.convolve(lc)
        lc1 = lc1.rebin3(tbin)
        lc1 = lc1 * (norm1 / lc1.mean())

        resp2 = ent.time_response(energy=enband2)
        norm2 = rate * np.sum(resp2) / fullrate
        lc2 = resp2.convolve(lc)
        lc2 = lc2.rebin3(tbin)
        lc2 = lc2 * (norm2 / lc2.mean())


        print("Count rate per energy band: %g, %g" % (lc1.mean(), lc2.mean()))

        model_cross_spec = CrossSpectrum(lc1, lc2).bin(bins)
        _, self.model_lag = model_cross_spec.lag_spectrum()

        if add_noise and sample_errors:
            lags = []
            for i in range(nsamples):
                cross_spec = CrossSpectrum(lc1.add_noise(), lc2.add_noise()).bin(bins)
                _, sample_lag = cross_spec.lag_spectrum()
                lags.append(sample_lag)
            lags = np.vstack(lags)
            self.lag = np.mean(lags, axis=0)
            self.error = np.std(lags, axis=0)
        elif add_noise:
            raise AssertionError("I don't know how to do that yet!")
        else:
            self.lag = self.model_lag
            self.error = np.zeros(self.lag.shape)


class SimLagEnergySpectrum(LagEnergySpectrum):
    def __init__(self, fmin, fmax, ent, rate, tbin=10, tmax=1E5, add_noise=True, sample_errors=True, nsamples=100, plslope=2,
                 std=1., refband=None, bias=True, lc=None):
        self.en = np.array(ent.en_bins.bin_cent)
        self.en_error = ent.en_bins.x_error()

        lclist = ent.norm().simulate_lc_list(tmax, plslope, std*rate, rate, add_noise=False, rebin_time=tbin, lc=lc)

        self.base_lc = lclist.base_lc

        self.model_lag, _, _ = self.calculate(lclist.lclist, fmin, fmax, refband=refband, energies=self.en,
                                        bias=False, calc_error=False)

        if add_noise and sample_errors:
            lags = []
            for i in range(nsamples):
                sample_lag, _, _ = self.calculate(lclist.add_noise().lclist, fmin, fmax, refband=refband, energies=self.en, bias=False, calc_error=False)
                lags.append(sample_lag)
            lags = np.vstack(lags)
            self.lag = np.mean(lags, axis=0)
            self.error = np.std(lags, axis=0)
            self.coh = None
        elif add_noise:
            self.lag, self.error, self.coh = self.calculate(lclist.lclist, fmin, fmax, refband=refband, energies=self.en,
                                       bias=bias, calc_error=True)
        else:
            self.lag, self.model_lag
            self.error = np.zeros(self.lag.shape)
            self.coh = None