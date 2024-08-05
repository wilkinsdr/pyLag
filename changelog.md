## Version 2.3 (August 2024)
- XspecModel class for incorporating Xspec spectral models into timing simulations
- Functionality to check and extract simultaneous light curve segments in EnergyLCList
- Enhancements to Gaussian process classes

## Version 2.2 (May 2024)
- MultiPlot class to create figures with multiple panels
- Reverberation model to compute and fit lag spectra and other timing products from realistic GR ray tracing simulations
- Enhanced model fitting functionality, including the ability to exclude data points
- Whittle statistic to fit unbinned periodograms
- Reimplementation of LagEnergySpectrum and CovarianceSpectrum to improve performance and enable rapid rebinning in frequency, to faciliate easy exploration of how the lags vary as a function of frequency
- Interpolation between parameter values for FITS spectral models
- Various bug fixes and enhancements

## Version 2.1 (November 2023)
- Maximum likelihood fitting of power spectrum and cross spectrum across multiple (stacked) light curves
- Include absorption and instrument ARFs for more realistic spectral timing simulations
- Convert time axes between MET for different X-ray missions
- Include background in light curves from XMM-Newton
- Compatibility with Python 3.12 and Numpy 1.24
- Various bug fixes and enhancements

## Version 2.0
- First public release