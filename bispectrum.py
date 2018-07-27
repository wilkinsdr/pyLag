from .lightcurve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import binned_statistic_2d
from .plotter import DataSeries

class Bispectrum(object):
    def __init__(self, lc):
        self.freq1, self.freq2 = self.bispec_freq(lc[0])
        self.bispec = self.calculate_bispectrum(lc)
        self.bicoherence = self.calculate_bicoherence(lc)
        self.biphase = np.angle(self.bispec)

    @staticmethod
    def bispec_segment(lc):
        f, ft = lc.ft()
        ft_conj = np.conj(ft)

        f1, f2 = np.meshgrid(range(int(len(f)/4)), range(int(len(f)/4)))

        bispec = ft[f1] * ft[f2] * ft_conj[f1 + f2]

        return bispec

    @staticmethod
    def bispec_segment_norm(lc):
        f, ft = lc.ft()

        f1, f2 = np.meshgrid(range(int(len(f) / 4)), range(int(len(f) / 4)))

        norm1 = np.abs(ft[f1] * ft[f2])**2
        norm2 = np.abs(ft[f1 + f2])**2

        return norm1, norm2

    @staticmethod
    def bispec_freq(lc):
        f = lc.ftfreq()
        f1, f2 = np.meshgrid(range(int(len(f) / 4)), range(int(len(f) / 4)))
        return f[f1], f[f2]

    def calculate_bispectrum(self, lclist):
        lc_len = len(lclist[0])
        bispec = []
        for lc in lclist:
            if len(lc) != lc_len:
                raise AssertionError("pylag Bispectrum ERROR: Light curve segments must be the same length")
            bispec.append(self.bispec_segment(lc))

        return (1./len(lclist)) * np.sum(bispec, axis=0)

    def calculate_bicoherence(self, lclist):
        lc_len = len(lclist[0])
        bispec = []
        norm1 = []
        norm2 = []
        for lc in lclist:
            if len(lc) != lc_len:
                raise AssertionError("pylag Bispectrum ERROR: Light curve segments must be the same length")
            bispec.append(self.bispec_segment(lc))
            n1, n2 = self.bispec_segment_norm(lc)
            norm1.append(n1)
            norm2.append(n2)

        return np.abs(np.sum(bispec, axis=0))**2 / (np.sum(norm1, axis=0) * np.sum(norm2, axis=0))

    def plot_bicoherence(self):
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(self.freq1, self.freq2, self.bicoherence)
        cb = fig.colorbar(mesh)
        ax.set_xlabel('Frequency 1 / Hz')
        ax.set_ylabel('Frequency 2 / Hz')
        cb.set_label('Bicoherence')

    def plot_biphase(self):
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(self.freq1, self.freq2, self.biphase, norm=colors.Normalize(-np.pi, np.pi))
        cb = fig.colorbar(mesh)
        ax.set_xlabel('Frequency 1 / Hz')
        ax.set_ylabel('Frequency 2 / Hz')
        cb.set_label('Biphase / rad')


class BinnedBispectrum(Bispectrum):
    def __init__(self, lc, bins):
        self.freq1, self.freq2 = np.meshgrid(bins.bin_cent, bins.bin_cent)
        self.bispec = self.calculate_bispectrum(lc, bins)
        self.bicoherence = self.calculate_bicoherence(lc, bins)
        self.biphase = np.angle(self.bispec)

    def calculate_bispectrum(self, lclist, bins):
        freq1 = []
        freq2 = []
        bispec = []
        for lc in lclist:
            f1, f2 = self.bispec_freq(lc)
            freq1.append(f1.flatten())
            freq2.append(f2.flatten())
            bispec.append(self.bispec_segment(lc).flatten())

        freq1 = np.hstack(freq1)
        freq2 = np.hstack(freq2)
        bispec = np.hstack(bispec)
        real = np.real(bispec)
        imag = np.imag(bispec)

        real_binned, _, _, _ = binned_statistic_2d(freq1, freq2, real, statistic='mean', bins=bins.bin_edges)
        imag_binned, _, _, _ = binned_statistic_2d(freq1, freq2, imag, statistic='mean', bins=bins.bin_edges)
        bispec_binned = real_binned + 1j * imag_binned

        return bispec_binned

    def calculate_bicoherence(self, lclist, bins):
        freq1 = []
        freq2 = []
        norm1 = []
        norm2 = []
        bispec = []
        for lc in lclist:
            f1, f2 = self.bispec_freq(lc)
            n1, n2 = self.bispec_segment_norm(lc)
            freq1.append(f1.flatten())
            freq2.append(f2.flatten())
            norm1.append(n1.flatten())
            norm2.append(n2.flatten())
            bispec.append(self.bispec_segment(lc).flatten())

        freq1 = np.hstack(freq1)
        freq2 = np.hstack(freq2)
        norm1 = np.hstack(norm1)
        norm2 = np.hstack(norm2)
        bispec = np.hstack(bispec)
        real = np.real(bispec)
        imag = np.imag(bispec)

        real_binned, _, _, _ = binned_statistic_2d(freq1, freq2, real, statistic='sum', bins=bins.bin_edges)
        imag_binned, _, _ , _= binned_statistic_2d(freq1, freq2, imag, statistic='sum', bins=bins.bin_edges)
        bispec_binned = real_binned + 1j * imag_binned

        norm1_binned, _, _, _ = binned_statistic_2d(freq1, freq2, norm1, statistic='sum', bins=bins.bin_edges)
        norm2_binned, _, _, _ = binned_statistic_2d(freq1, freq2, norm2, statistic='sum', bins=bins.bin_edges)

        bicoh_binned = np.abs(bispec_binned)**2 / (norm1_binned * norm2_binned)

        return bicoh_binned


class BinnedBispectrum1D(Bispectrum):
    def __init__(self, lc, bins):
        self.freq = bins.bin_cent
        self.freq_err = bins.x_error()
        self.bispec = self.calculate_bispectrum(lc, bins)
        self.bicoherence = self.calculate_bicoherence(lc, bins)
        self.biphase = np.angle(self.bispec)

    def calculate_bispectrum(self, lclist, bins):
        freq1 = []
        freq2 = []
        bispec = []
        for lc in lclist:
            f1, f2 = self.bispec_freq(lc)
            freq1.append(f1.flatten())
            freq2.append(f2.flatten())
            bispec.append(self.bispec_segment(lc).flatten())

        freq1 = np.hstack(freq1)
        freq2 = np.hstack(freq2)
        sumfreq = freq1 + freq2
        bispec = np.hstack(bispec)
        real = np.real(bispec)
        imag = np.imag(bispec)

        real_binned, _, _ = binned_statistic(sumfreq, real, statistic='mean', bins=bins.bin_edges)
        imag_binned, _, _ = binned_statistic(sumfreq, imag, statistic='mean', bins=bins.bin_edges)
        bispec_binned = real_binned + 1j * imag_binned

        return bispec_binned

    def calculate_bicoherence(self, lclist, bins):
        freq1 = []
        freq2 = []
        norm1 = []
        norm2 = []
        bispec = []
        for lc in lclist:
            f1, f2 = self.bispec_freq(lc)
            n1, n2 = self.bispec_segment_norm(lc)
            freq1.append(f1.flatten())
            freq2.append(f2.flatten())
            norm1.append(n1.flatten())
            norm2.append(n2.flatten())
            bispec.append(self.bispec_segment(lc).flatten())

        freq1 = np.hstack(freq1)
        freq2 = np.hstack(freq2)
        sumfreq = freq1 + freq2
        norm1 = np.hstack(norm1)
        norm2 = np.hstack(norm2)
        bispec = np.hstack(bispec)
        real = np.real(bispec)
        imag = np.imag(bispec)

        real_binned, _, _ = binned_statistic(sumfreq[np.logical_and(freq1 > 0, freq2 > 0)], real[np.logical_and(freq1 > 0, freq2 > 0)], statistic='mean', bins=bins.bin_edges)
        imag_binned, _, _ = binned_statistic(sumfreq[np.logical_and(freq1 > 0, freq2 > 0)], imag[np.logical_and(freq1 > 0, freq2 > 0)], statistic='mean', bins=bins.bin_edges)
        bispec_binned = real_binned + 1j * imag_binned

        norm1_binned, _, _ = binned_statistic(sumfreq[np.logical_and(freq1 > 0, freq2 > 0)], norm1[np.logical_and(freq1 > 0, freq2 > 0)], statistic='mean', bins=bins.bin_edges)
        norm2_binned, _, _ = binned_statistic(sumfreq[np.logical_and(freq1 > 0, freq2 > 0)], norm2[np.logical_and(freq1 > 0, freq2 > 0)], statistic='mean', bins=bins.bin_edges)

        bicoh_binned = np.abs(bispec_binned)**2 / (norm1_binned * norm2_binned)

        return bicoh_binned

    def bicoherence_series(self):
        return DataSeries(x=(self.freq, self.freq_err), y=self.bicoherence, xlabel='f1 + f2 / Hz', xscale='log', ylabel='Bicoherence', yscale='linear')

    def biphase_series(self):
        return DataSeries(x=(self.freq, self.freq_err), y=self.biphase, xlabel='f1 + f2 / Hz', xscale='log', ylabel='Biphase / rad', yscale='linear')