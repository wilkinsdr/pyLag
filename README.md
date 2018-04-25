# pyLag
pyLag is an object-oriented and (hopefully) easy to use X-ray timing analysis package implemented in Python by Dan Wilkins.

Functionality is provided for common timing analyses conducted on X-ray observations of accreting black holes, in particular the calculation of periodograms, power spectra and covariance spectra and the measurement of the reverberation of X-rays from the accretion disc. Reverberation is detected through time lags between correlated variability between two time series, either as a function of Fourier frequency (lag-frequency spectra) or X-ray energy (lag-energy spectra).

See the wiki for documentation

Dependencies
------------
- astropy.io.fits or pyfits
- numpy
- scipy (specifically fftpack and stats)
- matplotlib (for plotting functionality)

pyLag runs in python 2 or 3.

Getting Started
---------------
1) Put the pylag directory somewhere in your PYTHONPATH
2) import pylag
3) Make some cool discoveries!

D.R. Wilkins 16/03/2017
