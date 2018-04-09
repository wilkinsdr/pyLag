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

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import binned_statistic

from .binning import *
from .lightcurve import *

class EventList(object):
    def __init__(self, filename=None, hduname=None, hdunum=0, en_bins=None, t=None, ent=None, tstart=0.):
        self.time = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.pha = np.array([])

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
    def read_fits(filename, hduname=None, hdunum=0):
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
            time = hdu.data['TIME']
            try:
                pha = hdu.data['PI']
            except:
                pha = hdu.data['PHA']


        except:
            raise AssertionError("pyLag ENTResponse ERROR: Could not read axis information from FITS header")

        fitsfile.close()

        return en_bins, t, ent, logbin_en, tstart


