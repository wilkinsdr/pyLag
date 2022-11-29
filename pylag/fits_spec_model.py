import numpy as np
import astropy.io.fits as pyfits
from .plotter import Spectrum
from .util import printmsg

class FITSSpecModel(object):
    def __init__(self, filename, bins=None, interp=False):
        self.fits_file = pyfits.open(filename)

        self.en_low = np.array(self.fits_file['ENERGIES'].data['ENERG_LO'])
        self.en_high = np.array(self.fits_file['ENERGIES'].data['ENERG_HI'])
        self.energy = 0.5*(self.en_low + self.en_high)

        self.params = tuple(self.fits_file['PARAMETERS'].data['NAME'])
        self.param_num_vals = tuple(self.fits_file['PARAMETERS'].data['NUMBVALS'])
        self.param_tab_vals = tuple(self.fits_file['PARAMETERS'].data['VALUE'])

        # initially, spectra is a pointer to the table column in the FITS file
        # but we can replace it with a numpy array, e.g. if we rebin
        self.spectra = self.fits_file['SPECTRA'].data['INTPSPEC']

        param_initial = tuple(self.fits_file['PARAMETERS'].data['INITIAL'])
        self.values = {}
        for p,v in zip(self.params, param_initial):
            self.values[p] = v

        if bins is not None:
            self.rebin(bins)

        self.interpolator = None
        if interp:
            self.init_interpolator()

        self.filename = filename

    def __del__(self):
        self.fits_file.close()

    def find_energy(self, en):
        return np.argwhere(np.logical_and(self.en_low<=en, self.en_high>en))[0,0]

    def find_spec_num(self, **kwargs):
        values = dict(self.values)
        for p, v in kwargs.items():
            values[p] = v

        printmsg(1, values)

        # find the nearest tabulated value for each parameter
        param_num = [np.argmin(np.abs(tabvals - values[p])) for p, tabvals in zip(self.params, self.param_tab_vals)]

        spec_num = 0
        for n, pnum in enumerate(param_num):
            block_size = np.prod(self.param_num_vals[n+1:]) if n < (len(param_num) - 1) else 1
            spec_num += pnum * block_size

        return spec_num

    def spectrum(self, energy=None, norm_spec=False, **kwargs):
        if self.interpolator is not None:
            values = dict(self.values)
            for p, v in kwargs.items():
                values[p] = v

            spec = self.interpolator([values[k] for k in values])[0]
        else:
            spec_num = self.find_spec_num(**kwargs)
            spec = np.array(self.spectra[spec_num])

        if norm_spec:
            spec /= np.sum(spec)

        en = self.energy
        if energy is not None:
            imin = self.find_energy(energy[0])
            imax = self.find_energy(energy[1])
            en = en[imin:imax]
            spec = spec[imin:imax]

        return Spectrum(en, spec, xscale='log', xlabel='Energy / keV', yscale='log', ylabel='Count Rate')

    def rebin(self, en_bins):
        binspec = np.zeros((self.spectra.shape[0], len(en_bins)))
        for i in range(self.spectra.shape[0]):
            binspec[i, :] = en_bins.bin(self.energy, self.spectra[i])
        self.spectra = binspec
        self.energy = en_bins.bin_cent
        self.en_low = en_bins.bin_start
        self.en_high = en_bins.bin_end

    def init_interpolator(self):
        from scipy.interpolate import RegularGridInterpolator

        spec_array = np.array(self.spectra)
        spec_array = spec_array.reshape(13, 4, 15, 11, 10, 2999)

        vals = [np.trim_zeros(a, 'b') for a in self.param_tab_vals]
        self.interpolator = RegularGridInterpolator(tuple(vals), spec_array)

    def __str__(self):
        return "<pylag.fits_spec_model.FITSSpecModel: %s, parameters: %s%s>" % (self.filename, str(self.params), ", interpolation enabled" if self.interpolator is not None else "")

    def __repr__(self):
        return "<pylag.fits_spec_model.FITSSpecModel: %s, parameters: %s%s>" % (self.filename, str(self.params), ", interpolation enabled" if self.interpolator is not None else "")
