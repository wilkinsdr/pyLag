"""
pylag.continuum

Classes for modelling the continuum response

v1.0 - 14/05/2018 - D.R. Wilkins
"""
from .entresponse import *

import numpy as np
from scipy.interpolate import interp1d


class VerticalPropContinuumENT(ENTResponse):
    def __init__(self, source_file, spin, Gamma, vel_z, z_min, z_max, h_steps=100, logbin_h=False, redshift=True, rel_area=True, rel_vel=False, ent=None, en_bins=None, t=None, tstart=0.):
        if ent is not None:
            self.en_bins = ent.en_bins
            self.time = ent.time
            self.tstart = ent.tstart
        elif en_bins is not None and t is not None:
            self.en_bins = en_bins
            self.time = t
            self.tstart = tstart
        else:
            raise ValueError("Dimensions of response must be specified!")

        self.logbin_en = isinstance(self.en_bins, LogBinning)

        self.t0 = self.time.min()
        self.dt = self.time[1] - self.time[0]

        self.ent = self.calculate(source_file, spin, Gamma, vel_z, z_min, z_max, h_steps, logbin_h, redshift, rel_area, rel_vel)

    def calculate(self, source_file, spin, Gamma, vel_z, z_min, z_max, h_steps=100, logbin_h=False, redshift=True, rel_area=True, rel_vel=False):
        ent = np.zeros((len(self.en_bins), len(self.time)))
        en = self.en_bins.bin_cent

        if logbin_h:
            ratio = np.exp(np.log(z_max / z_min) / num)
            h_points = z_min * ratio ** np.arange(h_steps)
        else:
            dh = float(z_max - z_min) / h_steps
            h_points = np.arange(z_min, z_max, dh)

        source_data = np.genfromtxt(source_file)
        source_h = source_data[:,0]
        source_t = source_data[:, 1]
        source_g = source_data[:, 3]
        source_esc_frac = source_data[:, 10]

        t_interp = interp1d(source_h, source_t)
        g_interp = interp1d(source_h, source_g)
        esc_frac_interp = interp1d(source_h, source_esc_frac)

        t_initial = 0
        t_step_next = 0

        for h in h_points:
            if logbin_h:
                dh = h*ratio

            if rel_area or rel_vel:
                rhosq = h ** 2 + spin ** 2
                delta = h ** 2 - 2 * h + spin ** 2
                grr = rhosq / delta
            if vel_z > 0:
                if rel_vel:
                    t_initial += t_step_next
                    gtt = 1 - 2.*h/rhosq
                    t_step_next = dh / (vel_z * np.sqrt(gtt / grr))
                else:
                    t_initial = float(h - z_min) / vel_z
            else:
                t_initial = 0

            t = t_interp(h) + t_initial
            t_index = int((t - self.tstart - self.t0) / self.dt)

            g = g_interp(h)
            esc_frac = esc_frac_interp(h)

            # this makes no difference for a power law
            # but it would if there was a cutoff
            if redshift:
                spec = (en / g)**-Gamma
            else:
                spec = en**-Gamma
            spec /= np.sum(spec)

            if rel_area:
                area = np.sqrt(grr) * dh
            else:
                area = dh

            # normalisation of this height is equal to the volume/area of the bin
            # multiplied by the escape fraction of rays (solid angle subtended at observer)
            # with one factor of redshift for arrival rate of photons along each ray
            # (the energy factor is taken care of by redshifting the power law continuum)
            norm = area * esc_frac * g

            ent[:, t_index] += norm * spec

        return ent


class GammagradContinuumENT(ENTResponse):
    def __init__(self, t0, dt, gamma_start, gamma_end, ent=None, en_bins=None, t=None, tstart=0.):
        if ent is not None:
            self.en_bins = ent.en_bins
            self.time = ent.time
            self.tstart = ent.tstart
        elif en_bins is not None and t is not None:
            self.en_bins = en_bins
            self.time = t
            self.tstart = tstart
        else:
            raise ValueError("Dimensions of response must be specified!")

        self.t0 = self.time.min()
        self.dt = self.time[1] - self.time[0]

        self.ent = self.calculate(t0, dt, gamma_start, gamma_end)

    def calculate(self, t0, dt, gamma_start, gamma_end):
        ent = np.zeros((len(self.en_bins), len(self.time)))
        en = self.en_bins.bin_cent

        for it, t in enumerate(self.time):
            if (t+self.tstart) >= t0 and (t+self.tstart) < t0 + dt:
                gamma = gamma_start + (t + self.tstart - t0) * (gamma_end - gamma_start)/dt
                norm = 1.0
                spec = en**-gamma
                ent[:, it] += norm * spec/np.sum(spec)
            elif t >= t0 + dt:
                break

        return ent


