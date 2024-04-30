"""
pylag.reverb_model

Provides pyLag model to fit to reverberation measurements

Classes
-------


v1.0 02/05/2021 - D.R. Wilkins
"""
import lmfit
import re
import os
import numpy as np

from .fit import *
from .entresponse import *
from .fits_spec_model import FITSSpecModel

class ReverbModel(Model):
    def __init__(self, en_bins, resp_file, refl_spec, conv_bins=None, spec_bin=None, interp_resp=True, interp_spec=True):
        # working bins to calculate spectral response function
        self.conv_bins = LogBinning(0.1,100,250) if conv_bins is None else conv_bins
        # working bins to rebin reflection spectrum before convolution (to speed it up)
        self.spec_bin = LogBinning(0.1,100,500) if spec_bin is None else spec_bin

        self.en_bins = en_bins # we need the bin edges, not just the x values

        # load the response functions from specified directory
        self.response = ENTResponseSet(resp_file, interp=interp_resp)
        # and load the rest-frame Xillver reflection spectrum to convolve with the response
        self.spec = FITSSpecModel(refl_spec, interp=interp_spec)

        if self.spec_bin is not None and self.spec_bin != False:
            self.spec.rebin(self.spec_bin)

    def get_params(self, mass=7.3, source_h=5, le_fmin=1e-4, le_fmax=1e-3):
        params = lmfit.Parameters()

        params.add('mass', value=mass, min=6., max=8., vary=False)
        params.add('spin', value=mass, min=self.response.spin.min(), max=self.response.spin.max(), vary=False)
        params.add('h', value=source_h, min=self.response.heights.min(), max=self.response.heights.max())
        params.add('reffrac', value=1.0, min=0.1, max=10., vary=False)
        params.add('gamma', value=2.0, min=1.7, max=3.5, vary=False)
        params.add('logxi', value=2, min=0, max=4.7, vary=False)
        params.add('A_Fe', value=2, min=0, max=10, vary=False)
        params.add('incl', value=30, min=self.response.incl.min(), max=self.response.incl.max(), vary=False)

        params.add('offset', value=0, min=-1000, max=1000)

        params.add('fmin', value=le_fmin, min=1e-5, max=1e-3, vary=False)
        params.add('fmax', value=le_fmax, min=1e-5, max=1e-3, vary=False)

        return params

    def eval(self, params, x):
        source_h = params['h'].value
        spin = params['spin'].value
        lgmass = params['mass'].value
        reffrac = params['reffrac'].value
        gamma = params['gamma'].value
        logxi = params['logxi'].value
        A_Fe = params['A_Fe'].value
        incl = params['incl'].value
        offset = params['offset'].value
        fmin = params['fmin'].value
        fmax = params['fmax'].value

        # convolve with the reflection spectrum
        ent = self.response.get_response(spin, incl, source_h).convolve_spectrum(self.spec, self.conv_bins, binspec=None, Gamma=gamma, A_Fe=A_Fe, logXi=logxi, Incl=incl)
        # add the continuum emission (delta function at t=0)
        ent = ent.add_continuum(ent.tstart, gamma, reffrac)
        # rebin onto the required energy grid
        ent = ent.rebin_energy(self.en_bins)
        # scale the response
        ent = ent.rescale_time(mass=10. ** lgmass)
        # and calculate the lag-energy spectrum in the required frequency bin
        if fmin>0:
            lag = ent.lag_energy_spectrum(fmin, fmax, tmax=100000).lag
        else:
            lag = ent.avg_arrival().spec

        return lag

    def spectrum(self, params, x=0):
        lag = self.eval(params, x)
        en = self.en_bins.bin_cent
        en_error = self.en_bins.x_error
        return Spectrum(en=(en, en_error), spec=lag, ylabel='Lag', yscale='linear')