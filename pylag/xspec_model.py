import numpy as np
from .plotter import Spectrum

class XspecModel(object):
    def __init__(self, modelstr, enmin=0.1, enmax=100, Nen=1000):
        try:
            import xspec
        except ImportError:
            raise ImportError("XspecModel requires PyXSPEC to be installed and initialised in the current environment")

        xspec.xset.chatter = 0
        xspec.AllModels.clear()
        xspec.AllData.clear()
        xspec.AllData.dummyrsp(enmin, enmax, Nen)

        en_edges = np.logspace(np.log10(enmin), np.log10(enmax), Nen+1)
        self.en_low = en_edges[:-1]
        self.en_high = en_edges[1:]
        self.energy = 0.5 * (en_edges[1:] + en_edges[:-1])
        self.en_width = en_edges[1:] - en_edges[:-1]

        self.modelstr = modelstr
        self.model = xspec.Model(modelstr)

        self.params = []
        for c in self.model.componentNames:
            self.params += ['%s_%s' % (c, p) for p in getattr(self.model, c).parameterNames]

    def find_energy(self, en):
        return np.argwhere(np.logical_and(self.en_low<=en, self.en_high>en))[0,0]

    def spectrum(self, energy=None, norm_spec=False, **kwargs):
        for p, v in kwargs.items():
            if '_' in p:
                p_split = p.split('_')
                comp_name = p_split[0]
                par_name = p_split[1]
            else:
                comp_name = self.model.componentNames[0]
                par_name = p

            component = getattr(self.model, comp_name)
            par = getattr(component, par_name)
            par.values = v

        spec = self.model.values(0) / self.en_width

        if norm_spec:
            spec /= np.sum(spec)

        en = self.energy
        if energy is not None:
            imin = self.find_energy(energy[0])
            imax = self.find_energy(energy[1])
            en = en[imin:imax]
            spec = spec[imin:imax]

        return Spectrum(en, spec, xscale='log', xlabel='Energy / keV', yscale='log', ylabel='Count Rate')

    def __str__(self):
        return "<pylag.xspec_model.XspecModel: %s, parameters: %s>" % (self.modelstr, str(self.params))

    def __repr__(self):
        return "<pylag.xspec_model.XspecModel: %s, parameters: %s>" % (self.modelstr, str(self.params))

