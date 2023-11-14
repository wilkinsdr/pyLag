import numpy as np
from scipy.stats import binned_statistic

class TBabs(object):
    def __init__(self, enmin=0.1, enmax=100, Nen=1000):
        try:
            import xspec
        except ImportError:
            raise ImportError("TBabs requires PyXSPEC to be installed and initialised in the current environment")

        xspec.xset.chatter = 0
        xspec.AllModels.clear()
        xspec.AllData.clear()
        xspec.AllData.dummyrsp(enmin, enmax, Nen)

        en_edges = np.logspace(np.log10(enmin), np.log10(enmax), Nen+1)
        self.en = 0.5 * (en_edges[1:] + en_edges[:-1])
        self.en_width = en_edges[1:] - en_edges[:-1]

        self.model = xspec.Model('tbabs * powerlaw')
        self.model.TBabs.nH.values = 0.01
        self.model.powerlaw.PhoIndex.values = 0
        self.model.powerlaw.norm.values = 1

    def transmission(self, nH, enbins=None):
        self.model.TBabs.nH.values = nH
        return enbins.bin(self.en, self.model.values(0) / self.en_width) if enbins is not None else self.model.values(0) / self.en_width
