"""
pylag.event_list

Class for handling FITS event lists

v1.0 - 09/04/2018 - D.R. Wilkins
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

from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import matplotlib.colors as col

from .binning import *
from .lightcurve import *

class EventList(object):
    def __init__(self, filename=None, hduname=None, hdunum=0, time_col='TIME', x_col='X', y_col='Y', pha_col='PI', time=None, x=None, y=None, pha=None):
        self.time = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.pha = np.array([])

        if filename is not None:
            self.time, self.x, self.y, self.pha = self.read_fits(filename, hduname, hdunum, time_col, x_col, y_col, pha_col)
        else:
            if time is not None:
                self.time = time
            if x is not None:
                self.x = x
            if y is not None:
                self.y = y
            if pha is not None:
                self.pha = pha

    @staticmethod
    def read_fits(filename, hduname=None, hdunum=0, time_col='TIME', x_col='X', y_col='Y', pha_col='PI'):
        try:
            fitsfile = pyfits.open(filename)
        except:
            raise AssertionError("pyLag EventList ERROR: Could not open FITS file " + filename)

        try:
            if hduname is not None:
                hdu = fitsfile[hduname]
            else:
                hdu = fitsfile[hdunum]
        except:
            raise AssertionError("pyLag EventList ERROR: Could not open HDU")

        try:
            time = hdu.data[time_col]
            pha = hdu.data[pha_col]
            x = hdu.data[x_col]
            y = hdu.data[y_col]

        except:
            raise AssertionError("pyLag EventList ERROR: Could not read event list columns")

        fitsfile.close()

        return time, x, y, pha

    def lightcurve(self, tbin=10, time=None, calc_rate=True):
        if time is None:
            time = np.arange(self.time.min(), self.time.max(), tbin)
        counts, _ = np.histogram(self.time, time)
        if calc_rate:
            rate = counts / tbin
            err = rate * (np.sqrt(counts) / counts)
            return LightCurve(t=time[:-1], r=rate, e=err)
        else:
            err = np.sqrt(counts)
            return LightCurve(t=time[:-1], r=counts, e=err)

    def spectrum(self, phabins=None, Nbins=50):
        if phabins is None:
            phabins = LogBinning(self.pha.min(), self.pha.max(), Nbins)
        counts = phabins.num_points_in_bins(self.pha)
        rate = counts / (self.time.max() - self.time.min())
        err = rate * np.sqrt(counts) / counts
        return Spectrum(en=phabins.bin_cent, spec=rate, err=err)

    def show_image(self, bins=50):
        img, x_edges, y_edges = np.histogram2d(self.x, self.y, bins=bins)
        plt.imshow(img, norm=col.LogNorm(0.1, 1.1*img.max()), cmap='hot')
        plt.show()

    def plot_image(self, bins=50):
        img, x_edges, y_edges = np.histogram2d(self.x, self.y, bins=bins)
        x = 0.5*(x_edges[:-1] + x_edges[1:])
        y = 0.5 * (y_edges[:-1] + y_edges[1:])
        plt.pcolormesh(x, y, img, norm=col.LogNorm(0.1, 1.1*img.max()), cmap='hot')
        plt.show()

    def filter_region_circle(self, x0, y0, r):
        event_r = np.sqrt((self.x - x0)**2 + (self.y - y0)**2)
        filt_time = self.time[event_r < r]
        filt_pha = self.pha[event_r < r]
        filt_x = self.x[event_r < r]
        filt_y = self.y[event_r < r]
        return EventList(time=filt_time, x=filt_x, y=filt_y, pha=filt_pha)

    def filter_energy(self, enmin, enmax):
        enfilt = np.logical_and(self.pha >= enmin, self.pha < enmax)
        filt_time = self.time[enfilt]
        filt_pha = self.pha[enfilt]
        filt_x = self.x[enfilt]
        filt_y = self.y[enfilt]
        return EventList(time=filt_time, x=filt_x, y=filt_y, pha=filt_pha)

    def filter_time(self, tmin, tmax):
        tfilt = np.logical_and(self.time >= tmin, self.time < tmax)
        filt_time = self.time[tfilt]
        filt_pha = self.pha[tfilt]
        filt_x = self.x[tfilt]
        filt_y = self.y[tfilt]
        return EventList(time=filt_time, x=filt_x, y=filt_y, pha=filt_pha)

    def bayesian_lightcurve(self, gamma=1):
        import astropy.stats.bayesian_blocks
        bin_edges = astropy.stats.bayesian_blocks(self.time, fitness='events', gamma=gamma)
        tcent = 0.5*(bin_edges[1:] + bin_edges[:-1])
        terr = tcent - bin_edges[:-1]
        bin_rate = np.array([np.sum(np.logical_and(self.time>=tstart, self.time<tend)) for tstart, tend in zip(bin_edges[:-1], bin_edges[1:])]).astype(float)
        bin_rate /= (bin_edges[1:] - bin_edges[:-1])
        return VariableBinLightCurve(tcent, terr, bin_rate)


