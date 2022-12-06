# pyLag
pyLag is an object-oriented and (hopefully) easy to use X-ray timing analysis package implemented in Python by Dan Wilkins.

Functionality is provided for common timing analyses conducted on X-ray observations of accreting black holes, in particular the calculation of periodograms, power spectra and covariance spectra and the measurement of the reverberation of X-rays from the accretion disc. Reverberation is detected through time lags between correlated variability between two time series, either as a function of Fourier frequency (lag-frequency spectra) or X-ray energy (lag-energy spectra).

See the wiki for documentation

Dependencies
------------
- `astropy.io.fits` or, alternatively, the now-deprecated `pyfits`
- `numpy`
- `scipy` (specifically `scipy.fftpack` and `scipy.stats`)
- `matplotlib` (for plotting functionality)

pyLag runs in python 2 or 3.

Getting Started
---------------
1) pyLag can be installed from PyPI using pip: `pip install pylag` (or `pip install --user pylag` if you do not have write access to system-level directories and/or would like to install pyLag only for the current user)
2) `import pylag`
3) Make some cool discoveries!

Introductory information and tutorials demonstrating the use of pyLag's functionality are available in the GitHub wiki.

Keeping Up to Date
------------------
pyLag gets updated regularly to provide new functionality (usually by adding new classes to do new types of calculation, or adding functions to existing classes). There are also occasional bug-fixes to existing functions, though only if absolutely necessary will an update change a way a function fundamentally operates. All changes are documented in the git log.

If you find any bugs or errors (either computational or scientific in nature), do please open an issue or send me a message!

If you cloned pyLag from this repository, you can automatically update it to the latest version by running `git pull` in the pylag directory.

D.R. Wilkins 21/04/2021
