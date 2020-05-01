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
from .plotter import Spectrum


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)


class ENTResponse(object):
    def __init__(self, filename=None, hduname=None, hdunum=0, en_bins=None, t=None, ent=None, logbin_en=None, tstart=0.):
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
            elif en_bins is not None and t is not None:
                self.ent = np.zeros((len(en_bins), len(t)))
            if logbin_en is not None:
                self.logbin_en = logbin_en
            else:
                self.logbin_en = False

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
            Nen = hdu.header['NEN']
            try:
                logbin_en = hdu.header['ENLOG']
            except:
                logbin_en = False

            try:
                enmax = hdu.header['ENMAX']
            except:
                # fallback for old format FITS header
                den = hdu.header['DEN']
                if logbin_en:
                    enmax = en0 * den**Nen
                else:
                    enmax = en0 + den*Nen

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

    def continuum_ent(self, tcont, gamma, ref_frac, from_start=True):
        ti = self.t_index(tcont, from_start)

        cont = self.en_bins.bin_cent ** -gamma

        cont_ent = np.zeros(self.ent.shape)
        cont_ent[:, ti] = np.sum(self.ent) * cont / (ref_frac * np.sum(cont))

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=cont_ent, tstart=self.tstart)

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
            ent.append(bins.bin(self.time, self.ent[ien,:]))
        return ENTResponse(en_bins=self.en_bins, t=bins.bin_start, ent=np.array(ent), tstart=self.tstart)

    def rescale_time(self, mult=None, mass=None):
        """
        Rescale the time axis of the light curve, multiplying by a constant
        e.g. for GM/c^3 to s conversion
        """
        if mass is not None:
            mult = 6.67E-11 * mass * 2E30 / (3E8)**3
        t = self.time * mult
        return ENTResponse(en_bins=self.en_bins, t=t, ent=self.ent, tstart=self.tstart)

    def moving_average_energy(self, window_size=3):
        window = np.ones(int(window_size)) / float(window_size)
        ent_avg = np.zeros(self.ent.shape)
        for it in range(self.ent.shape[1]):
            ent_avg[...,it] = np.convolve(self.ent[...,it], window, 'same')
        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=ent_avg, tstart=self.tstart)

    def norm(self):
        norm_ent = self.ent / self.ent.sum()

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=norm_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def plot_image(self, vmin=None, vmax=None, mult_scale=True, cmap='gray_r', log_scale=True):
        return ImagePlot(self.time, self.en_bins.bin_cent, self.ent, cmap=cmap, log_scale=log_scale, vmin=vmin, vmax=vmax, mult_scale=mult_scale, xlabel='Time / GM c$^{-3}$', ylabel='Energy / keV')

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
            ti = self.t_index(time, from_start)
            spec = np.array(self.ent[:, ti])

        return Spectrum(self.en_bins.bin_cent, spec)

    def time_response(self, energy=None, index=None):
        if isinstance(energy, tuple):
            enstarti = self.en_index(energy[0])
            enendi = self.en_index(energy[1])
            resp = np.sum(self.ent[enstarti:enendi, :], axis=0)
        elif energy is None and index is None:
            resp = np.sum(self.ent, axis=0)
        elif index is not None:
            resp = np.array(self.ent[index, :])
        else:
            eni = self.en_index(energy)
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

    def lag_frequency_spectrum(self, enband1, enband2, fbins=None, Nf=None, tmax=None):
        if fbins is None:
            if Nf is None:
                raise ValueError("pylag ENTResponse lag_frequency_spectrum ERROR: Either frequency binning object or number of frequency bins required")
            minfreq = 1./(2.*(self.time.max() - self.time.min()))
            maxfreq = 1./(2.*(self.time[1]-self.time[0]))
            fbins = LogBinning(minfreq, maxfreq, Nf)

        if tmax is None:
            tmax = 1./fbins.bin_start.min()

        resp1 = self.time_response(enband1).pad(tmax)
        resp2 = self.time_response(enband2).pad(tmax)
        return LagFrequencySpectrum(fbins, lc1=resp1, lc2=resp2, calc_error=False)

    def cross_spectrum(self, enband1, enband2, fbins=None, tmax=None):
        if tmax is None and fbins is not None:
            tmax = 1./fbins.bin_start.min()

        resp1 = self.time_response(enband1).pad(tmax)
        resp2 = self.time_response(enband2).pad(tmax)
        c = CrossSpectrum(lc1=resp1, lc2=resp2)

        if fbins is not None:
            return c.bin(fbins)
        else:
            return c

    def lag_energy_frequency(self, fbins=None, Nf=100, pad=1E6):
        if Nf is None:
            raise ValueError(
                "pylag ENTResponse lag_frequency_spectrum ERROR: Either frequency binning object or number of frequency bins required")
        if fbins is None:
            minfreq = 1. / (2. * (pad - self.time.min()))
            maxfreq = 1. / (2. * (self.time[1] - self.time[0]))
            fbins = LogBinning(minfreq, maxfreq, Nf)
        lagfreq = np.zeros((len(fbins), len(self.en_bins)))
        for i in range(len(self.en_bins)):
            _, lagfreq[:,i] = self.time_response(index=i).pad(pad).lagfreq(fbins)
        #return fbins.bin_cent, self.en_bins.bin_cent, lagfreq

        lagfreq[np.isnan(lagfreq)] = 0

        return ImagePlot(fbins.bin_cent, self.en_bins.bin_cent, lagfreq.T, log_scale=False, vmin=lagfreq.min(), vmax=lagfreq.max(), mult_scale=False, xscale='log', yscale='log')

    def energy_lc_list(self):
        lclist = []
        for ien in range(len(self.en_bins)):
            lclist.append(ImpulseResponse(t=self.time, r=self.ent[ien,:]))
        return SimEnergyLCList(enmin=self.en_bins.bin_start, enmax=self.en_bins.bin_end, lclist=lclist)

    def lag_energy_spectrum(self, fmin=None, fmax=None):
        if fmin is None:
            fmin = 1./(2.*(self.time.max() - self.time.min()))
            fmax = 1./(2.*(self.time[1]-self.time[0]))
        return LagEnergySpectrum(fmin, fmax, lclist=self.energy_lc_list(), calc_error=False)

    def simulate_lc_list(self, tmax, plslope, std, lcmean, add_noise=False, rebin_time=None, lc=None):
        lclist = []
        if lc is None:
            lc = SimLightCurve(self.time[1] - self.time[0], tmax, plslope, std, lcmean)
        for ien in range(len(self.en_bins)):
            enlc = self.time_response(index=ien).convolve(lc)
            if rebin_time is not None:
                enlc = enlc.rebin3(rebin_time)
            if add_noise:
                enlc = enlc.add_noise()
            lclist.append(enlc)
        return SimEnergyLCList(enmin=self.en_bins.bin_start, enmax=self.en_bins.bin_end, lclist=lclist, base_lc=lc)

    def write_fits(self, filename):
        hdu = pyfits.PrimaryHDU()

        hdu.header['EN0'] = self.en_bins.bin_start[0]
        hdu.header['ENMAX'] = self.en_bins.bin_end[-1]
        hdu.header['NEN'] = len(self.en_bins)
        hdu.header['ENLOG'] = self.logbin_en

        hdu.header['T0'] = self.t0
        hdu.header['DT'] = self.dt
        hdu.header['NT'] = len(self.time)
        hdu.header['TSTART'] = self.tstart

        hdu.data = self.ent

        hdu.writeto(filename)

    def __add__(self, other):
        if not isinstance(other, ENTResponse):
            return NotImplemented

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be added!")

        sum_ent = self.ent + other.ent

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=sum_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __iadd__(self, other):
        if not isinstance(other, ENTResponse):
            raise AssertionError("Can only add ENTResponse objects!")

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be added!")

        self.ent += other.ent
        return self

    def __sub__(self, other):
        if not isinstance(other, ENTResponse):
            return NotImplemented

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be subtracted!")

        sub_ent = self.ent - other.ent

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=sub_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __isub__(self, other):
        if not isinstance(other, ENTResponse):
            return NotImplemented

        if self.ent.shape != other.ent.shape:
            raise AssertionError("Response matrices must have the same dimensions to be subtracted!")

        self.ent -= other.ent
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            mul_ent = self.ent * other
        elif isinstance(other, ENTResponse):
            mul_ent = self.ent * other.ent
        else:
            return NotImplemented

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=mul_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.ent *= other
        elif isinstance(other, ENTResponse):
            self.ent *= other.ent
        else:
            return NotImplemented

        return self

    def __div__(self, other):
        if isinstance(other, (int, float)):
            div_ent = self.ent / other
        elif isinstance(other, ENTResponse):
            div_ent = self.ent / other.ent
        else:
            return NotImplemented

        return ENTResponse(en_bins=self.en_bins, t=self.time, ent=div_ent, logbin_en=self.logbin_en, tstart=self.tstart)

    def __idiv__(self, other):
        if isinstance(other, (int, float)):
            self.ent /= other
        elif isinstance(other, ENTResponse):
            self.ent /= other.ent
        else:
            return NotImplemented

        return self

