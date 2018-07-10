"""
pylag.fit

Provides pyLag functionality for fitting functions to data series

Classes
-------


v1.0 05/07/2018 - D.R. Wilkins
"""

import lmfit
from .plotter import *


def resid(params, x, data, err, model):
    return data - model(params, x)


def chisq(params, x, data, err, model):
    return (data - model(params, x)) / err


def broken_pl_model(params, x):
    norm = params['norm'].value
    slope1 = params['slope1'].value
    xbreak = params['xbreak'].value
    slope2 = params['slope2'].value

    func = np.zeros(x.shape)
    func[x <= xbreak] = norm * x[x <= xbreak] ** -slope1
    func[x > xbreak] = norm * xbreak ** (slope2 - slope1) * x[x > xbreak] ** -slope2
    return func


class Fit(object):
    def __init__(self, data, modelfn, params, statistic=chisq):
        self.data_obj = data
        self.modelfn = modelfn
        self.params = params
        self.statistic = statistic

        self.minimizer = None
        self.fit_result = None

        xd, yd = data._getplotdata()
        if isinstance(xd, tuple):
            self.xdata = xd[0]
            self.xerror = xd[1]
        else:
            self.xdata = xd
            self.xerror = None
        if isinstance(yd, tuple):
            self.ydata = yd[0]
            self.yerror = yd[1]
        else:
            self.ydata = yd
            self.yerror = None

        # weed out NaN values in ydata
        self.xdata = self.xdata[np.logical_not(np.isnan(self.ydata))]
        if self.xerror is not None:
            self.xerror = self.xerror[np.logical_not(np.isnan(self.ydata))]
        if self.yerror is not None:
            self.yerror = self.yerror[np.logical_not(np.isnan(self.ydata))]
        self.ydata = self.ydata[np.logical_not(np.isnan(self.ydata))]

        # and weed out NaN values in yerror if applicable
        if self.yerror is not None:
            self.ydata = self.ydata[np.logical_not(np.isnan(self.yerror))]
            self.xdata = self.xdata[np.logical_not(np.isnan(self.yerror))]
            if self.xerror is not None:
                self.xerror = self.xerror[np.logical_not(np.isnan(self.yerror))]
            self.yerror = self.ydata[np.logical_not(np.isnan(self.yerror))]

    def perform_fit(self, fit_range=None):
        if isinstance(fit_range, tuple):
            xmin, xmax = fit_range
            xd = self.xdata[np.logical_and(self.xdata >= xmin, self.xdata < xmax)]
            yd = self.ydata[np.logical_and(self.xdata >= xmin, self.xdata < xmax)]
            ye = self.yerror[np.logical_and(self.xdata >= xmin, self.xdata < xmax)]
        elif fit_range is None:
            xd = self.xdata
            yd = self.ydata
            ye = self.yerror
        else:
            raise ValueError("pylag Fit perform_fit ERROR: Unexpected value for fit_range")
        self.minimizer = lmfit.Minimizer(self.statistic, self.params, fcn_args=(xd, yd, ye, self.modelfn))
        self.fit_result = self.minimizer.minimize()
        #self.fit_result = lmfit.minimize(self.statistic, self.params, args=(xd, yd, ye, self.modelfn))

    def confidence_intervals(self, default_stderr=0.1):
        for parname in self.fit_result.params:
            if self.fit_result.params[parname].stderr is None:
                self.fit_result.params[parname].stderr = self.fit_result.params[parname].value*default_stderr

        self.ci_result = lmfit.conf_interval(self.minimizer, self.fit_result)
        self.report_ci()

    def report(self):
        lmfit.report_fit(self.fit_result)

    def report_ci(self):
        lmfit.printfuncs.report_ci(self.ci_result)

    def fit_function(self, x=None):
        if x is None:
            x = self.xdata
        return x, self.modelfn(self.fit_result.params, x)

    def _getdataseries(self, x=None):
        x, y = self.fit_function(x)
        return DataSeries(x, y)

    def plot_fit(self, x=None):
        p = Plot([self.data_obj, self._getdataseries(x)])
        p.marker_series = ['+', '-']
        return p