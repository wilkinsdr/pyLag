from .lightcurve import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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


