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
1) Clone the pylag repository somewhere in your `PYTHONPATH`
    - One way to do this is to download or clone the `pylag` directory to a convenient location `/path/to/pylag` and then add this to your `PYTHONPATH` environment variable by typing `export PYTHONPATH=$PYTHONPATH:/path/to/pylag` (in Bash or other Borne-like shells, use `setenv` in C-shells)
    - If you are using Python through Anaconda, instead of adding the directory to `PYTHONPATH`, you need to add the pyLag directory to a file called `local.pth` in your Anaconda installation's `site-packages` directory. This directory is usually `~/opt/anaconda3/lib/pythonX.Y/site-packages` (type `which python` to find out where it is). `~` denotes your home directory and `X.Y` should be substituted for the version of Python you are running. If `local.pth` does not exist in this directory, create it.
2) `import pylag`
3) Make some cool discoveries!

Introductory information and tutorials demonstrating the use of pyLag's functionality are available in the GitHub wiki.

Keeping Up to Date
------------------
pyLag gets updated regularly to provide new functionality (usually by adding new classes to do new types of calculation, or adding functions to existing classes). There are also occasional bug-fixes to existing functions, though only if absolutely necessary will an update change a way a function fundamentally operates. All changes are documented in the git log.

If you find any bugs or errors (either computational or scientific in nature), do please open an issue or send me a message!

If you cloned pyLag from this repository, you can automatically update it to the latest version by running `git pull` in the pylag directory.

D.R. Wilkins 21/04/2021
