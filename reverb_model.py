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

from .fit import *
from .entresponse import *
from .fits_spec_model import FITSSpecModel

class ReverbModel(Model):
    def __init__(self, en_bins, resp_dir, refl_spec, conv_bins=None, spec_bin=None):
        # working bins to calculate spectral response function
        self.conv_bins = LogBinning(0.1,100,250) if conv_bins is None else conv_bins
        # working bins to rebin reflection spectrum before convolution (to speed it up)
        self.spec_bin = LogBinning(0.1,100,500) if spec_bin is None else spec_bin

        self.en_bins = en_bins # we need the bin edges, not just the x values

        # load the response functions from specified directory
        self.h, self.line_resp = self.read_response(resp_dir)
        # and load the rest-frame Xillver reflection spectrum to convolve with the response
        self.spec = FITSSpecModel(refl_spec)

    def read_response(self, resp_dir):
        # find the response files
        resp_files = sorted(glob.glob(resp_dir + '/*.fits'))
        # extract the source height from the filename
        h_re = re.compile('.*?_h([0-9\.]+)[_\.].*?')
        h_labels = ['h = ' + h_re.match(f).group(1) for f in [os.path.basename(e) for e in resp_files]]
        h = [float(h_re.match(f).group(1)) for f in [os.path.basename(e) for e in resp_files]]
        # sort the files in order of source height
        resp_files = [f for _, f in sorted(zip(h, resp_files))]
        h = np.array(sorted(h))
        # and load the responses
        resp = [ENTResponse(f, 'RESPONSE') for f in resp_files]

        return h, resp

    def get_params(self, mass=7.3, source_h=5, le_fmin=1e-4, le_fmax=1e-3):
        params = lmfit.Parameters()

        params.add('mass', value=mass, min=6, max=8, vary=False)
        params.add('h', value=source_h, min=self.h.min(), max=self.h.max())
        params.add('reffrac', value=0.5, min=0.1, max=5., vary=False)
        params.add('gamma', value=2.15, min=2.0, max=3.0, vary=False)
        params.add('logxi', value=2, min=0, max=3.5, vary=False)
        params.add('A_Fe', value=2, min=0, max=10, vary=False)
        params.add('incl', value=2, min=0, max=90, vary=False)

        params.add('offset', value=0, min=-1000, max=1000)

        params.add('fmin', value=le_fmin, min=1e-5, max=1e-3, vary=False)
        params.add('fmax', value=le_fmax, min=1e-5, max=1e-3, vary=False)

        return params

    def eval(self, params, x):
        source_h = params['h'].value
        lgmass = params['mass'].value
        reffrac = params['reffrac'].value
        gamma = params['gamma'].value
        logxi = params['logxi'].value
        A_Fe = params['A_Fe'].value
        incl = params['incl'].value
        offset = params['offset'].value
        fmin = params['fmin'].value
        fmax = params['fmax'].value

        if source_h in self.h:
            #
            # if the response for this height is available, just use it
            #

            # find the repsonse we need
            ix = np.argwhere(self.h == source_h)[0][0]
            print(ix)
            # convolve with the reflection spectrum
            ent = self.line_resp[ix].convolve_spectrum(self.spec, self.conv_bins, binspec=self.spec_bin, Gamma=gamma, A_Fe=A_Fe, logXi=logxi, Incl=incl)
            # add the continuum emission (delta function at t=0)
            ent = ent.add_continuum(ent.tstart, gamma, reffrac)
            # rebin onto the required energy grid
            ent = ent.rebin_energy(self.en_bins)
            # scale the response
            ent = ent.rescale_time(mass=10. ** lgmass)
            # and calculate the lag-energy spectrum in the required frequency bin
            if fmin>0:
                lag = ent.lag_energy_spectrum(fmin, fmax, pad=100000).lag
            else:
                lag = ent.avg_arrival().spec
        else:
            #
            # if not, interpolate between the heights we have either side of the requested value
            #

            # find the low and high end response indices
            ix_low = np.max(np.argwhere(self.h < source_h))
            ix_high = np.min(np.argwhere(self.h > source_h))

            # linear interpolation coefficients
            frac_low = (self.h[ix_high] - source_h) / (self.h[ix_high] - self.h[ix_low])
            frac_high = (source_h - self.h[ix_low]) / (self.h[ix_high] - self.h[ix_low])

            # get the lag-energy spectrum for the low end of the interpolation
            ent = self.line_resp[ix_low].convolve_spectrum(self.spec, self.conv_bins, binspec=self.spec_bin, Gamma=gamma, A_Fe=A_Fe, logXi=logxi, Incl=incl)
            ent = ent.add_continuum(ent.tstart, gamma, reffrac)
            ent = ent.rebin_energy(self.en_bins)
            ent = ent.rescale_time(mass=10. ** lgmass)
            if fmin>0:
                lag_low = ent.lag_energy_spectrum(fmin, fmax, pad=100000).lag
            else:
                lag_low = ent.avg_arrival().spec

            # and the high end
            ent = self.line_resp[ix_high].convolve_spectrum(self.spec, self.conv_bins, binspec=self.spec_bin, Gamma=gamma, A_Fe=A_Fe, logXi=logxi, Incl=incl)
            ent = ent.add_continuum(ent.tstart, gamma, reffrac)
            ent = ent.rebin_energy(self.en_bins)
            ent = ent.rescale_time(mass=10. ** lgmass)
            if fmin > 0:
                lag_high = ent.lag_energy_spectrum(fmin, fmax, pad=100000).lag
            else:
                lag_high = ent.avg_arrival().spec

            lag = frac_low * lag_low + frac_high * lag_high - offset

        return lag

    def spectrum(self, params, x=0):
        lag = self.eval(params, x)
        en = self.en_bins.bin_cent
        en_error = self.en_bins.x_error
        return Spectrum(en=(en, en_error), spec=lag, ylabel='Lag', yscale='linear')