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




class Fit(object):
    def __init__(self, data, model, params=None, statistic=chisq, **kwargs):
        self.data_obj = data
        self.model = model(**kwargs)
        self.modelfn = self.model.eval
        if params is not None:
            self.params = params
        else:
            self.params = self.model.get_params()
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
            self.yerror = self.yerror[np.logical_not(np.isnan(self.yerror))]

    def _dofit(self, params, fit_range=None):
        """
        Internal function for running the minimizer. Returns the fit results so that we can do
        something with them (e.g. for steppar)

        :param params: Starting parameter
        :param fit_range: Range of x values to include in the fit (if None, use all)
        :return:
        """
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
        self.minimizer = lmfit.Minimizer(self.statistic, params, fcn_args=(xd, yd, ye, self.modelfn), xtol=1e-10, ftol=1e-10)
        result = self.minimizer.minimize()
        return result
        #self.fit_result = lmfit.minimize(self.statistic, self.params, args=(xd, yd, ye, self.modelfn))

    def perform_fit(self, fit_range=None):
        self.fit_result = self._dofit(self.params, fit_range)

    def confidence_intervals(self, default_stderr=0.1):
        for parname in self.fit_result.params:
            if self.fit_result.params[parname].stderr is None:
                self.fit_result.params[parname].stderr = self.fit_result.params[parname].value*default_stderr

        self.ci_result = lmfit.conf_interval(self.minimizer, self.fit_result)
        self.report_ci()

    def steppar(self, step_param, start, stop, steps, log_space=False, fit_range=None):
        import copy

        if log_space:
            ratio = np.exp(np.log(stop / start) / steps)
            par_steps = start * ratio ** np.array(range(steps))
        else:
            par_steps = np.linspace(start, stop, steps)

        step_params = copy.copy(self.params)
        step_params[step_param].vary = False

        stat = []
        fit_params = []

        for value in par_steps:
            step_params[step_param].value = value
            result = self._dofit(step_params, fit_range)
            stat.append(result.chisqr)
            fit_params.append(result.params)

        return par_steps, stat, fit_params

    def steppar2(self, step_param1, start1, stop1, steps1, step_param2, start2, stop2, steps2, log_space1=False, log_space2=False, fit_range=None):
        import copy

        if log_space1:
            ratio1 = np.exp(np.log(stop1 / start1) / steps1)
            par_steps1 = start1 * ratio1 ** np.array(range(steps1))
        else:
            par_steps1 = np.linspace(start1, stop1, steps1)
        if log_space2:
            ratio2 = np.exp(np.log(stop2 / start2) / steps2)
            par_steps2 = start2 * ratio2 ** np.array(range(steps2))
        else:
            par_steps2 = np.linspace(start2, stop2, steps2)

        step_params = copy.copy(self.params)
        step_params[step_param1].vary = False
        step_params[step_param2].vary = False

        stat = np.zeros((len(par_steps1), len(par_steps2)))
        fit_params = []

        for i1, value1 in enumerate(par_steps1):
            fit_params.append([])
            step_params[step_param1].value = value1
            for i2, value2 in enumerate(par_steps2):
                step_params[step_param2].value = value2
                result = self._dofit(step_params, fit_range)
                stat[i1,i2] = result.chisqr
                fit_params[-1].append(result.params)

        return par_steps1, par_steps2, stat, fit_params

    def report(self):
        lmfit.report_fit(self.fit_result)

    def report_ci(self):
        lmfit.printfuncs.report_ci(self.ci_result)

    def fit_function(self, x=None, params=None):
        if x is None:
            x = self.xdata
        if params is None:
            params = self.fit_result.params
        return x, self.modelfn(params, x)

    def ratio(self, params=None):
        if params is None:
            params = self.fit_result.params
        r = self.ydata / self.modelfn(params, self.xdata)
        if self.yerror is not None:
            e = r * (self.yerror / self.ydata)
            return r, e
        else:
            return r

    def _getdataseries(self, x=None, params=None):
        x, y = self.fit_function(x, params)
        return DataSeries(x, y)

    def _getratioseries(self, x=None, params=None):
        if self.xerror is not None:
            x = (self.xdata, self.xerror)
        else:
            x = self.xdata
        xlabel, xscale, ylabel, yscale = self.data_obj._getplotaxes()
        return DataSeries(x, self.ratio(params), xlabel=xlabel, xscale=xscale, ylabel='Data / Model', yscale='linear')

    def _getfitdataseries(self):
        # use this function to get a plottable data series of the data points we're actually fitting
        if self.xerror is not None:
            x = (self.xdata, self.xerror)
        else:
            x = self.xdata
        if self.xerror is not None:
            y = (self.ydata, self.yerror)
        else:
            y = self.ydata

        return DataSeries(x, y)


    def plot_fit(self, x=None, params=None):
        p = Plot([self.data_obj, self._getdataseries(x, params)])
        p.marker_series = ['+', '-']
        return p

    def plot_ratio(self, x=None, params=None):
        p = Plot(self._getratioseries(x, params))
        p.marker_series = ['+', '-']
        return p

    def write_fit(self, filename):
        outdata = [self._getfitdataseries(), self._getdataseries(), self._getratioseries()]
        write_multi_data(outdata, filename)

    def run_mcmc(self, burn=100, steps=1000, thin=1, params=None, fit_range=None):
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

        if params is None:
                params = self.fit_result.params if self.fit_result is not None else self.parans

        self.mcmc_result = lmfit.minimize(lambda params : self.statistic(params, xd, yd, ye, self.modelfn), params=params, method='emcee', burn=burn, steps=steps,
                                     thin=thin)

