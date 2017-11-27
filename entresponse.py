"""
pylag.entresponse

Class for handling energy-time response functions from ray tracing simulations

v1.0 - 05/09/2017 - D.R. Wilkins
"""
from .simulator import *
from .lightcurve import *
from .lag_frequency_spectrum import *
from .lag_energy_spectrum import *

import numpy as np

try:
    import astropy.io.fits as pyfits
except:
    import pyfits

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import binned_statistic

from .binning import *


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)


class ENTResponse(object):
    def __init__(self, filename=None, hduname=None, hdunum=0, en_bins=None, t=None, ent=None, tstart=0.):
        self.en = np.array([])
        self.time = np.array([])
        self.ent = np.array([])
        self.tstart = tstart

        if filename is not None:
            self.en_bins, self.time, self.ent, self.logbin_en, self.tstart = self.read_fits(filename, hduname, hdunum)
        else:
            if en_bins is not None:
                self.en_bins = en_bins
            if t is not None:
                self.time = t
            if ent is not None:
                self.ent = ent

        self.t0 = min(self.time)
        self.dt = self.time[1] - self.time[0]

    @staticmethod
    def read_fits(filename, hduname=None, hdunum=0, byte_swap=True):
        try:
            fitsfile = pyfits.open(filename)
        except:
            raise AssertionError("pyLag ENTResponse ERROR: Could not open FITS file " + filename)

        try:
            if hduname is not None:
                hdu = fitsfile[hduname]
            else:
                hdu = fitsfile[hdunum]
        except:
            raise AssertionError("pyLag ENTResponse ERROR: Could not open HDU")

        ent = np.array(hdu.data)
        if byte_swap:
            ent = ent.byteswap().newbyteorder('<')

        try:
            en0 = hdu.header['EN0']
            enmax = hdu.header['ENMAX']
            Nen = hdu.header['NEN']
            logbin_en = False  # hdu.header['ENLOG']

            if logbin_en:
                en_bins = LogBinning(en0, enmax, Nen)
            else:
                en_bins = LinearBinning(en0, enmax, Nen)

            t0 = hdu.header['T0']
            dt = hdu.header['DT']
            Nt = hdu.header['NT']
            tstart = hdu.header['TSTART']

            t = t0 + dt * np.arange(0, Nt, 1)
        except:
            raise AssertionError("pyLag ENTResponse ERROR: Could not read axis information from FITS header")

        fitsfile.close()

        return en_bins, t, ent, logbin_en, tstart

    def t_index(self, time, from_start=False):
        if from_start:
            return int((time - (self.t0 + self.tstart)) / self.dt)
        else:
            return int((time - self.t0) / self.dt)

    def en_index(self, energy):
        return self.en_bins.bin_index(energy)

    def add_continuum(self, tcont, gamma, ref_frac, from_start=True):
        ti = self.t_index(tcont, from_start)

        cont = self.en_bins.bin_cent ** -gamma

        tot_ent = np.array(self.ent)
        tot_ent[:, ti] = self.ent[:, ti] + np.sum(self.ent) * cont / (ref_frac * np.sum(cont))

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=tot_ent, tstart=self.tstart)

    def energy_range(self, enmin, enmax):
        enstarti = self.en_index(enmin)
        enendi = self.en_index(enmax)
        return ENTResponse(en_bins=self.en_bins[enstarti:enendi], t=self.time, ent=self.ent[enstarti:enendi, :], tstart=self.tstart)

    def time_range(self, tmin, tmax):
        tstarti = self.t_index(tmin)
        tendi = self.t_index(tmax)
        return ENTResponse(en_bins=self.en_bins, t=self.time[tstarti:tendi], ent=self.ent[:, tstarti:tendi], tstart=self.tstart)

    def __getitem__(self, index):
        return ENTResponse(en_bins=self.en_bins, t=self.time[index], ent=self.ent[:, index], tstart=self.tstart)

    def rebin_energy(self, bins=None, Nen=None):
        if bins is None:
            bins = LogBinning(self.en_bins.min(), self.en_bins.max(), Nen)

        ent = []
        for bin_start, bin_end in zip(bins.bin_start, bins.bin_end):
            enstarti = self.en_index(bin_start)
            enendi = self.en_index(bin_end)
            ent.append(np.sum(self.ent[enstarti:enendi,:], axis=0))
        return ENTResponse(en_bins=bins, t=self.time, ent=np.array(ent), tstart=self.tstart)

    def rebin_time(self, bins=None, dt=None, Nt=None):
        if bins is None:
            if dt is not None:
                bins = LinearBinning(self.time.min(), self.time.max(), step=dt)
            if Nt is not None:
                bins = LinearBinning(self.time.min(), self.time.max(), num=Nt)

        ent = []
        for ien in range(self.ent.shape[0]):
            ent.append(bins.bin_fast(self.time, self.ent[ien,:]))
        return ENTResponse(en=self.en, t=bins.bin_start, ent=np.array(ent), logbin_en=self.logbin_en, tstart=self.tstart)

    def plot_image(self, vmin=None, vmax=None, mult_scale=True, cmap='hot'):
        fig, ax = plt.subplots()

        if vmin is None:
            vmin = self.ent.min()
        if vmax is None:
            vmax = self.ent.max()
        if mult_scale:
            vmin *= self.ent.max()
            vmax *= self.ent.max()
        if vmin == 0:
            vmin = 1.e-3
        ax.pcolormesh(self.time, self.en_bins.bin_cent, self.ent, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
        plt.xlabel('Time')
        plt.ylabel('Energy / keV')

        return fig, ax

    def spectrum(self, time=None, index=False, from_start=True):
        if isinstance(time, tuple):
            tstarti = self.t_index(time[0], from_start)
            tendi = self.t_index(time[1], from_start)
            spec = np.sum(self.ent[:, tstarti:tendi], axis=1)
        elif time is None:
            spec = np.sum(self.ent, axis=1)
        elif index:
            spec = np.array(self.ent[:, time])
        else:
            ti = self.t_tindex(time, from_start)
            spec = np.array(self.ent[:, ti])

        return Spectrum(self.en_bins.bin_cent, spec)

    def time_response(self, energy=None, index=False):
        if isinstance(energy, tuple):
            enstarti = self.en_index(energy[0])
            enendi = self.en_index(energy[1])
            resp = np.sum(self.ent[enstarti:enendi, :], axis=0)
        elif energy is None:
            resp = np.sum(self.ent, axis=0)
        elif index:
            resp = np.array(self.ent[:, energy])
        else:
            eni = self.en_tindex(energy)
            resp = np.array(self.ent[eni, :])

        return ImpulseResponse(t=self.time, r=resp)

    def avg_arrival(self):
        lag = []
        for ien in range(self.ent.shape[0]):
            if np.sum(self.ent[ien, :] > 0):
                lag.append(np.average(self.time, weights=self.ent[ien, :]))
            else:
                lag.append(np.nan)
        lag = np.array(lag)

        return Spectrum(self.en_bins.bin_cent, lag, ylabel='Lag', yscale='linear')

    def std_arrival(self):
        lag = []
        for ien in range(self.ent.shape[0]):
            if np.sum(self.ent[ien, :] > 0):
                lag.append(weighted_std(self.time, weights=self.ent[ien, :]))
            else:
                lag.append(np.nan)
        lag = np.array(lag)

        return Spectrum(self.en_bins.bin_cent, lag, ylabel='Lag', yscale='linear')

    def lag_frequency_spectrum(self, enband1, enband2, fbins=None, Nf=None):
        if fbins is None:
            if Nf is None:
                raise ValueError("pylag ENTResponse lag_frequency_spectrum ERROR: Either frequency binning object or number of frequency bins required")
            minfreq = 1./(2.*(self.time.max() - self.time.min()))
            maxfreq = 1./(2.*(self.time[1]-self.time[0]))
            fbins = LogBinning(minfreq, maxfreq, Nf)

        resp1 = self.time_response(enband1)
        resp2 = self.time_response(enband2)
        return LagFrequencySpectrum(fbins, lc1=resp1, lc2=resp2)

    def energy_lc_list(self):
        lclist = []
        for ien in range(len(self.en_bins)):
            lclist.append(LightCurve(t=self.time, r=self.ent[ien,:]))
        return EnergyLCList(enmin=self.en_bins.bin_start, enmax=self.en_bins.bin_end, lclist=lclist)

    def lag_energy_spectrum(self, fmin=None, fmax=None):
        if fmin is None:
            fmin = 1./(2.*(self.time.max() - self.time.min()))
            fmax = 1./(2.*(self.time[1]-self.time[0]))
        return LagEnergySpectrum(fmin, fmax, lclist=self.energy_lc_list())


class Spectrum(object):
    def __init__(self, en, spec, xlabel='Energy / keV', xscale='log', ylabel='Count Rate', yscale='log'):
        self.en = en
        self.spec = spec

        self.xlabel = xlabel
        self.xscale = xscale
        self.ylabel = ylabel
        self.yscale = yscale

    def _getplotdata(self):
        return self.en, self.spec

    def _getplotaxes(self):
        return self.xlabel, self.xscale, self.ylabel, self.yscale

    def rebin2(self, Nen=None, logbin=True, den=None):
        if logbin:
            if den is None:
                den = np.exp(np.log(self.en.max() / self.en.min()) / (float(Nen) - 1.))
            en_bin = self.en.min() * den ** np.arange(0, Nen+1, 1)
        else:
            if den is None:
                den = (self.en.max() - self.en.min()) / (float(Nen) - 1.)
            en_bin = np.arange(self.en.min(), self.en.max() + 2*den, den)

        spec_bin,_,_ = binned_statistic(self.en, self.spec, statistic='mean', bins=en_bin)
        return Spectrum(en=en_bin, spec=spec_bin, xlabel=self.xlabel, xscale=self.xscale, ylabel=self.ylabel,
                        yscale=self.yscale)

    def rebin(self, bins=None, Nen=None):
        if bins is None:
            bins = LogBinning(self.en.min(), self.en.max(), Nen)
        spec_bin = bins.bin(self.en, self.spec)
        return Spectrum(en=bins.bin_cent, spec=spec_bin, xlabel=self.xlabel, xscale=self.xscale, ylabel=self.ylabel,
                        yscale=self.yscale)
