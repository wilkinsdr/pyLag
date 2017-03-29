"""
pylag.simulator

Tools for simulating light curves and X-ray timing measurements

v1.0 - 27/03/2017 - D.R. Wilkins
"""
from pylag.lightcurve import *

import numpy as np
import scipy.fftpack

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
	using the AddNoise() method.

	Constructor: pylag.SimLightCurve(dt=10., tmax=1000., plslope=2.0, std=0.5, lcmean=1.0, t=None, r=None, e=None, gtzero=True)

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
	def __init__(self, dt=10., tmax=1000., plslope=2.0, std=0.5, lcmean=1.0, t=None, r=None, e=None, gtzero=True):
		if t is None and r is None:
			t = np.arange(0, tmax, dt)
			r = self.Calculate(t, plslope, std, lcmean, gtzero=gtzero)
		if e is None:
			e = np.sqrt(r/dt)
		LightCurve.__init__(self, t=t, r=r, e=e)

	def Calculate(self, t, plslope, std, lcmean, plnorm=1., gtzero=True):
		"""
		pylag.SimLightCurve.Calculate(t, plslope, lcmean, std, plnorm=1.)

		Simulate a random light curve with a specified power spectrum using the
		Timmer & Konig method; the amplitude of the Fourier transorm of the time
		series at each sample frequency is set according to the power spectrum
		and the phase of each component is drawn at random from a uniform
		distribution. The time series is computed from the inverse FFT of this.

		Constructor Arguments
		---------------------
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
		freq = scipy.fftpack.fftfreq(len(t), d=t[1]-t[0])
		# normalise the power law PSD
		plnorm = plnorm / ( (2*np.pi*freq[1])**(-plslope) )
		# build the Fourier transform of the light curve
		# amplitude at each frequency is according to a power law, phase is random
		# note we use abs(freq) to populate the negative and positive frequencies
		# since a real light curve has a symmetric FFT. Also skip f=0
		ampl = np.sqrt( 0.5 * plnorm * (2*np.pi*np.abs(freq[1:]))**(-plslope) )
		ampl = np.insert(ampl, 0, plnorm) # add the zero frequency element
		phase = 2 * np.pi * np.random.rand(len(freq))

		# put the Fourier transform together then invert it to get the light curbe
		ft = ampl * np.exp(1j*phase)
		r = np.real( scipy.fftpack.ifft(ft) )

		# normalise and shift the light curve to get the desired mean and stdev
		r = std * r/np.std(r)
		r = r - np.mean(r) + lcmean
		# don't let any points drop below zero (the detector will see nothing here)
		if gtzero:
			r[r<0] = 0

		return r

	def AddNoise(self):
		"""
		lc = pylag.SimLightCurve.AddNoise

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
		# draw the counts in each time bin from a Poisson distribution
		# with the mean set according to the original number of counts in the bin
		rnd_counts = np.random.poisson(counts)
		rate = rnd_counts.astype(float) / self.dt
		# sqrt(N) errors again as if we're making a measurement
		error = np.sqrt(self.rate)

		return SimLightCurve(t=self.time, r=rate, e=error)
