"""
pylag.Plotter

Plotting and data output classes/functions for pylag data products
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker
import numpy as np

# all_plots is a list of all plot objects that have been created
all_plots = []

class Plot(object):
    """
    pylag.Plot

    Class to produce a plot, either automatically from a data object or manually
    from x,y series. Plotting is done using matplotlib.

    Each pylag plottable data product class (e.g. LagFrequecySpectrum, LagEnergySpectrum)
    has two methods: _getplotdata() which returns the arrays that should be plotted
    as the x and y values with their corresponding errors and _getplotaxes() which
    returns the axis labels and scaling. This enables the plot to be created
    automatically simply by creating a Plot object.

    e.g.
    >>> p = pylag.Plot(myLagEnergySpectrum)

    The plot is created by the constructor function which accepts a number of
    options to customise the plot.

    Further options can be set through member variables after the plot is created.
    After changing options, call the Plot() method to update the plot.

    The plot can be saved to a file using the save() method. For automated
    processing, set show_plot=False in the constructor so that a plot can be
    created or updated and then saved using save() without it being displayed
    on screen.

    When a new plot is created, it is added to the pylab.all_plots list so that
    it can be referenced later.

    Member Variables
    ----------------
    fig            : matplotlib Figure instance
    ax             : matplotlib Axes handle

    Data Series:
    xdata          : tuple of ndarrays or list of tuples of ndarrays
                    x data points to be plotted (or a list of multiple data series)
                    the x data points for one of the series to be plotted)
    xerror        : ndarray or list of ndarrays
                    Symmetric error on x data points
    ydata          : ndarray or list of ndarrays
                    y data points
    yerror        : ndarray or list of ndarrays
                    Symmetric error on y data points
    series_labels : list of strings, optional, default=[]
                    The label for each data series shown in the legend. Each
                    entry should correspond to one of the data objects or one
                    of the manually specified x,y series

    Properties
    ----------
    title         : string
                    Title to be displayed at the top of the plot

    Axes:
    xlabel        : string
                    x axis label
    ylabel        : string
                    y axis label
    xscale        : string
                    'linear' or 'log' to set scaling of x axis
    yscale        : string
                    'linear' or 'log' to set scaling of x axis
    xlim          : list of floats [min, max] (default=None)
                    If set, manually specify the limits of the x axis. If None,
                    the axis will be scaled automatically
    ylim          : list of floats [min, max] (default=None)
                    If set, manually specify the limits of the y axis. If None,
                    the axis will be scaled automatically

    Formatting:
    grid          : string (default='minor')
                    Specify which grid lines are shown: 'major', 'minor' or 'none'
    legend        : boolean
                    If True, a legend is shown on the plot. By default, set
                    to True if series labels are provided
    legend_loc    : string (default='upper right')
                    The matplotlib string specifying the location the legend
                    should be placed ('best', 'upper right', 'center right',
                    'lower left' etc.)
    colour_series : list of strings (default=['k', 'b', 'g', 'r', 'c', 'm'])
                    The repeating sequence of matplotlib colour specifiers
                    setting the order in which colours are assigned to plot
                    series. The sequence is repeated as many times as necessary
                    to cover all the series
    marker_series : list of strings (default=['+', 'x', 'o', 's'])
                    The repeating sequence of matplotlib plot marker specifiers
                    setting the order in which they are applied to data series.
                    To plot all series as lines, use a single entry ['-']
    font_face     : string (default=None)
                    Specify the font face. If None, use the matplotlub default
    font_size     : integer (default=None)
                    Specify the font size. If None, use the matplotlub default

    Constructor: p = pylag.Plot(data_object=None, xdata=None, ydata=None, xscale='', yscale='', xlabel='', ylabel='', title='', labels=[], preset=None, show_plot=True)

    Constructor Arguments
    ---------------------
    data_object   : pyLag plottable data product object or list of objects
                    optional (default = None)
                    If set, the plot is automatically produced from the object.
                    If a list of objects is passed, each one is plotted as a
                    separate data series on the plot
    xdata         : ndarray, tuple or list, optional (default=None)
                    If data_object is not set, the x co-ordinates of the series to
                    be plotted. If x values have symmetric errors, pass a tuple of
                    arrays (value, error). If multiple series are to be plotted,
                    pass a list of arrays or list of tuples
    ydata         : ndarray, tuple or list, optional (default=None)
                    If data_object is not set, the y co-ordinates of the series to
                    be plotted. If y values have symmetric errors, pass a tuple of
                    arrays (value, error). If multiple series are to be plotted,
                    pass a list of arrays or list of tuples
    xscale        : string, 'linear' or 'log', optional (default='')
                    If set, override the default x axis scaling specified by the
                    data object to be plotted (or the 'linear' default for manually
                    specified data series)
    yscale        : string, 'linear' or 'log', optional (default='')
                    If set, override the default y axis scaling
    xlabel        : string, optional (default='')
                    If set, override the default x axis label set by the data object
    ylabel        : string, optional (default='')
                    If set, override the default y axis label set by the data object
    title         : string, optional (default='')
                    The title to be shown at the top of the plot
    series_labels : list of strings, optional, default=[]
                    The label for each data series shown in the legend. Each
                    entry should correspond to one of the data objects or one
                    of the manually specified x,y series. If set, a legend will
                    be displayed on the plot, if not, the legend is hidden
    preset        : for future use
    show_plot     : boolean, optional (default=True)
                    display the plot window on screen automatically when the plot
                    is created or updated. If False, the plot can be displayed
                    by calling the show method()
    """

    def __init__(self, data_object=None, xdata=None, ydata=None, xscale='', yscale='', xlabel='',
                 ylabel='', title='', series_labels=[], grid='minor', lines=False, marker_series=None, colour_series=None,
                 errorbar=True, preset=None, figsize=None, show_plot=True):
        self._fig = None
        self._ax = None

        self._labels = list(series_labels)

        self._xlabel = ''
        self._ylabel = ''

        self._title = title

        self._xscale = 'linear'
        self._yscale = 'linear'

        self._xtickformat = None
        self._ytickformat = None

        self.errorbar = errorbar

        self._figsize = figsize

        # variables to set plot formatting
        if colour_series is not None:
            self._colour_series = colour_series
        else:
            self._colour_series = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

        if marker_series is not None:
            self._marker_series = marker_series
        elif lines:
            self._marker_series = ['-']
        else:
            self._marker_series = ['+', 'x', 'o', 's']

        self._font_face = None
        self._font_size = None
        self._tick_scale = 0.88
        self._grid = grid
        self._legend_location = 'upper right'
        self._xlim = None
        self._ylim = None

        # do we display the plot on screen automatically when calling plot()?
        self.show_plot = show_plot

        self._legend = (len(self._labels) > 0)

        if data_object is not None:
            self.xdata = []
            self.xerror = []
            self.ydata = []
            self.yerror = []

            if not isinstance(data_object, list):
                # if we're only given one object, put it in a list
                data_object = [data_object]
            # go through each object and put in the pointers to the x/y
            # data series and their errors
            for obj in data_object:
                try:
                    xd, yd = obj._getplotdata()
                    if isinstance(xd, tuple):
                        self.xdata.append(xd[0])
                        self.xerror.append(xd[1])
                    else:
                        self.xdata.append(xd)
                        self.xerror.append(None)
                    if isinstance(yd, tuple):
                        self.ydata.append(yd[0])
                        self.yerror.append(yd[1])
                    else:
                        self.ydata.append(yd)
                        self.yerror.append(None)
                    # if no labels are passed in for the series, use a blank string
                    if len(series_labels) == 0:
                        self._labels.append('')
                except:
                    raise ValueError('pylag Plotter ERROR: The object I was passed does not seem to be plottable')
            # read the axis labels from data_object
            self._xlabel, self._xscale, self._ylabel, self._yscale = data_object[0]._getplotaxes()

        elif xdata is not None and ydata is not None:
            self.xdata = []
            self.xerror = []
            self.ydata = []
            self.yerror = []

            # if we're not passed an object, use the data series that are passed in
            if not isinstance(xdata, list):
                xdata = [xdata]
            if not isinstance(ydata, list):
                ydata = [ydata]
            if len(xdata) != len(ydata):
                raise ValueError('pylag Plotter ERROR: I need the same number of data series for x and y!')

            for xd, yd in zip(xdata, ydata):
                if isinstance(xd, tuple):
                    if len(xd[0]) != len(xd[1]):
                        raise ValueError('pylag Plotter ERROR: I need the same number data points and errors in x!')
                    self.xdata.append(xd[0])
                    self.xerror.append(xd[1])
                else:
                    self.xdata.append(xd)
                    self.xerror.append(None)
                if isinstance(yd, tuple):
                    self.ydata.append(yd[0])
                    self.yerror.append(yd[1])
                else:
                    self.ydata.append(yd)
                    self.yerror.append(None)
                if len(self.xdata[-1]) != len(self.ydata[-1]):
                    raise ValueError('pylag Plotter ERROR: I need the same number of data points in x and y!')
                # if no labels are passed in for the series, use a blank string
                if len(series_labels) == 0:
                    self._labels.append('')

        # if we're passed axis labels, these override the labels set in data_object
        if xlabel != '':
            self._xlabel = xlabel
        if ylabel != '':
            self._ylabel = ylabel
        # if we're passed axis log/linear scaling, these override the scaling set in data_object
        if xscale != '':
            self._xscale = xscale
        if yscale != '':
            self._yscale = yscale

        self.plot()
        all_plots.append(self)

    def _setup_axes(self):
        """
        pylag.plot._setup_axes()

        close and recreate the figure and axes, applying the updated settings.
        This function is called automatically by plot()
        """
        # close the old figure (if already plotted)
        if self._fig is not None:
            self.close()

        # create a new figure window and axes
        self._fig, self._ax = plt.subplots(figsize=self._figsize)

        # set log or linear scaling
        self._ax.set_xscale(self._xscale)
        self._ax.set_yscale(self._yscale)

        # set axis labels
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)
        self._ax.set_title(self._title)

        # turn major/minor grid lines on or off
        if self._grid == 'major' or self._grid == 'minor':
            self._ax.grid(which='major')
        if self._grid == 'minor':
            self._ax.grid(which='minor')

        # set the axis ranges if set
        if self._xlim is not None:
            self._ax.set_xlim(self._xlim)
        if self._ylim is not None:
            self._ax.set_ylim(self._ylim)

        if self._xtickformat == 'scalar':
            self._ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if self._xtickformat is not None:
            self._ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: self._xtickformat % val))

        if self._ytickformat == 'scalar':
            self._ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        elif self._ytickformat is not None:
            self._ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: self._ytickformat % val))


    def _plot_data(self):
        """
        pylag.plot._plot_data()

        Add each data series to the plot as points (or lines) with error bars
        """
        # repeat the colour and marker series as many times as necessary to provide for all the data series
        colours = (self._colour_series * int(len(self.xdata) / len(self._colour_series) + 1))[:len(self.xdata)]
        markers = (self._marker_series * int(len(self.xdata) / len(self._marker_series) + 1))[:len(self.xdata)]

        # plot the data series in turn
        for xd, yd, yerr, xerr, marker, colour, label in zip(self.xdata, self.ydata, self.yerror, self.xerror, markers,
                                                             colours, self._labels):
            if not isinstance(xerr, (np.ndarray, list)):
                xerr = np.zeros(len(xd))
            if not isinstance(yerr, (np.ndarray, list)):
                yerr = np.zeros(len(yd))
            if self.errorbar:
                self._ax.errorbar(xd[np.isfinite(yd)], yd[np.isfinite(yd)], yerr=yerr[np.isfinite(yd)] if yerr.ndim==1 else np.vstack([ye[np.isfinite(yd)] for ye in yerr]),
                              xerr=xerr[np.isfinite(yd)], fmt=marker, color=colour, label=label)
            else:
                if marker != '-':
                    mk = marker
                else:
                    mk = None
                self._ax.plot(xd[np.isfinite(yd)], yd[np.isfinite(yd)], marker=mk, color=colour, label=label)

    def _set_fonts(self):
        self._ax.set_xlabel(self._xlabel, fontname=self._font_face, fontsize=self._font_size)
        self._ax.set_ylabel(self._ylabel, fontname=self._font_face, fontsize=self._font_size)

        ticksize = int(self._tick_scale * self._font_size)
        for tick in self._ax.get_xticklabels():
            tick.set_fontname(self._font_face)
            tick.set_fontsize(ticksize)
        for tick in self._ax.get_yticklabels():
            tick.set_fontname(self._font_face)
            tick.set_fontsize(ticksize)

    def show(self, **kwargs):
        """
        pylag.plot.show()

        show the plot window on the screen
        """
        self._fig.show(**kwargs)

    def plot(self, **kwargs):
        """
        pylag.plot.plot()

        Wrapper function to perform all steps necessary to create/update the plot.
        If show_plot is True, display the plot window at the end.

        The _plot_data() function is called to add the data points to the plot.
        This may be overriden in derived classes to produce different plot types.
        """
        self._setup_axes()
        self._plot_data(**kwargs)
        if self._font_face is not None:
            if self._font_size is None:
                self._font_size = 10
            self._set_fonts()
        if self.legend:
            if self._font_face is not None:
                font_prop = font_manager.FontProperties(family=self._font_face, size=self._font_size)
            else:
                font_prop = None
            self._ax.legend(loc=self._legend_location, prop=font_prop)
        self._fig.tight_layout()
        if self.show_plot:
            self.show()

    def close(self):
        """
        pylag.plot.show()

        close this plot's window
        """
        plt.close(self._fig)

    def save(self, filename, **kwargs):
        """
        pylag.plot.show()

        save the plot to a file using matplotlib's savefig() function.

        Arguments
        ---------
        filename : string
                   The name of the file to be created
        **kwargs : passed on to matplotlib savefig() function
        """
        self._fig.savefig(filename, bbox_inches='tight', **kwargs)

    def paper_format(self):
        self._font_face = 'Liberation Serif'
        self._font_size = 18
        self._tick_scale = 0.88
        self.plot()

    def _get_setter(attr):
        """
        setter = pylag.plot._get_setter(attr)

        Returns a setter function for plot attribute attr which updates the
        member variable and then refreshes the plot.

        A re-usable setter functions for all properties that need the plot to
        update

        Arguments
        ---------
        attr : string
             : The name of the member variable to be set

        Returns
        -------
        setter : function
                 The setter function
        """

        def setter(self, value):
            setattr(self, attr, value)
            self.plot()

        return setter

    def _get_getter(attr):
        """
        getter = pylag.plot._get_getter(attr)

        Returns a getter function for plot attribute attr.

        A re-usable getter functions for all properties

        Arguments
        ---------
        attr : string
             : The name of the member variable to get

        Returns
        -------
        getter : function
                 The get function
        """

        def getter(self):
            return getattr(self, attr)

        return getter

    title = property(_get_getter('_title'), _get_setter('_title'))
    labels = property(_get_getter('_labels'), _get_setter('_labels'))
    xlabel = property(_get_getter('_xlabel'), _get_setter('_xlabel'))
    ylabel = property(_get_getter('_ylabel'), _get_setter('_ylabel'))
    xscale = property(_get_getter('_xscale'), _get_setter('_xscale'))
    yscale = property(_get_getter('_yscale'), _get_setter('_yscale'))
    xlim = property(_get_getter('_xlim'), _get_setter('_xlim'))
    ylim = property(_get_getter('_ylim'), _get_setter('_ylim'))
    grid = property(_get_getter('_grid'), _get_setter('_grid'))
    legend = property(_get_getter('_legend'), _get_setter('_legend'))
    legend_location = property(_get_getter('_legend_location'), _get_setter('_legend_location'))
    colour_series = property(_get_getter('_colour_series'), _get_setter('_colour_series'))
    marker_series = property(_get_getter('_marker_series'), _get_setter('_marker_series'))
    font_face = property(_get_getter('_font_face'), _get_setter('_font_face'))
    font_size = property(_get_getter('_font_size'), _get_setter('_font_size'))
    tick_scale = property(_get_getter('_tick_scale'), _get_setter('_tick_scale'))
    xtickformat = property(_get_getter('_xtickformat'), _get_setter('_xtickformat'))
    ytickformat = property(_get_getter('_ytickformat'), _get_setter('_ytickformat'))
    figsize = property(_get_getter('_figsize'), _get_setter('_figsize'))


class ErrorRegionPlot(Plot):
    """
    pylag.ErrorRegionPlot

    Class to plot data objects or series as a shaded error region.

    See plot class for details
    """

    def _plot_data(self, use_xerror=False, alpha=0.5):
        """
        pylag.ErrorRegionPlot._plot_data()

        Add each data series to the plot as a shaded error region
        """
        # repeat the colour series as many times as necessary to provide for all the data series
        colours = (self._colour_series * int(len(self.xdata) / len(self._colour_series) + 1))[:len(self.xdata)]
        # plot each data series in turn
        for xd, yd, yerr, xerr, colour, label in zip(self.xdata, self.ydata, self.yerror, self.xerror, colours,
                                                     self._labels):
            if use_xerror:
                # if we're including the x error points, we need to put in the co-ordinates for the left
                # and right hand sides of each error box
                high_bound = []
                low_bound = []
                xpoints = []
                for x, y, xe, ye in zip(xd, yd, xerr, yerr):
                    high_bound.append(y + ye)
                    high_bound.append(y + ye)
                    low_bound.append(y - ye)
                    low_bound.append(y - ye)
                    xpoints.append(x - xe)
                    xpoints.append(x + xe)
                self._ax.plot(xd, yd, '-', color=colour, label=label)
                self._ax.fill_between(xpoints, low_bound, high_bound, facecolor=colour, alpha=alpha)
            else:
                high_bound = np.array(yd) + np.array(yerr)
                low_bound = np.array(yd) - np.array(yerr)
                self._ax.plot(xd, yd, '-', color=colour, label=label)
                self._ax.fill_between(xd, low_bound, high_bound, facecolor=colour, alpha=alpha)


class QuiverPlot(Plot):
    """
    pylag.QuiverPlot

    Class to plot data objects as a field of arrows between data points.

    See plot class for details
    """
    def _plot_data(self):
        """
        pylag.plot._plot_data()

        Add each data series to the plot as points (or lines) with error bars
        """
        # repeat the colour and marker series as many times as necessary to provide for all the data series
        colours = (self._colour_series * int(len(self.xdata) / len(self._colour_series) + 1))[:len(self.xdata)]
        markers = (self._marker_series * int(len(self.xdata) / len(self._marker_series) + 1))[:len(self.xdata)]

        # plot the data series in turn
        for xd, yd, yerr, xerr, marker, colour, label in zip(self.xdata, self.ydata, self.yerror, self.xerror, markers,
                                                             colours, self._labels):
            if not isinstance(xerr, (np.ndarray, list)):
                xerr = np.zeros(len(xd))
            if not isinstance(yerr, (np.ndarray, list)):
                yerr = np.zeros(len(yd))
            self._ax.quiver(xd[np.isfinite(yd)][:-1], yd[np.isfinite(yd)][:-1], xd[np.isfinite(yd)][1:]-xd[np.isfinite(yd)][:-1],
                            yd[np.isfinite(yd)][1:]-yd[np.isfinite(yd)][:-1], scale_units='xy', angles='xy', scale=1, color=colour, label=label)


class ImagePlot(Plot):
    def __init__(self, x, y, img, cmap='gray_r', log_scale=True, vmin=None, vmax=None, mult_scale=True, xscale='', yscale='', xlabel='',
                 ylabel='', title='', grid='none', show_plot=True):
        self.xdata = x
        self.ydata = y
        self.img_data = img
        self._cmap = cmap
        self._log_scale = log_scale
        self._vmin = vmin
        self._vmax = vmax
        self._mult_scale = mult_scale
        Plot.__init__(self, xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid, show_plot=show_plot)

    def _plot_data(self):
        if self._vmin is None:
            vmin = self.img_data.min()
        elif self._mult_scale:
            vmin = self._vmin * self.img_data.max()
        else:
            vmin = self.vmin

        if self._vmin is None:
            vmax = self.img_data.max()
        elif self._mult_scale:
            vmax = self._vmax * self.img_data.max()
        else:
            vmax = self._vmax

        if vmin == 0 and self._log_scale:
            vmin = 1.e-3 * vmax

        if self._log_scale:
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        img = np.array(self.img_data)
        img[img < vmin] = vmin

        self._mesh = self._ax.pcolormesh(self.xdata, self.ydata, img, norm=norm, cmap=self._cmap)

    def colorbar(self):
        self._fig.colorbar(self._mesh)

    def _get_setter(attr):
        """
        setter = pylag.plot._get_setter(attr)

        Returns a setter function for plot attribute attr which updates the
        member variable and then refreshes the plot.

        A re-usable setter functions for all properties that need the plot to
        update

        Arguments
        ---------
        attr : string
             : The name of the member variable to be set

        Returns
        -------
        setter : function
                 The setter function
        """

        def setter(self, value):
            setattr(self, attr, value)
            self.plot()

        return setter

    def _get_getter(attr):
        """
        getter = pylag.plot._get_getter(attr)

        Returns a getter function for plot attribute attr.

        A re-usable getter functions for all properties

        Arguments
        ---------
        attr : string
             : The name of the member variable to get

        Returns
        -------
        getter : function
                 The get function
        """

        def getter(self):
            return getattr(self, attr)

        return getter

    cmap = property(_get_getter('_cmap'), _get_setter('_cmap'))
    vmin = property(_get_getter('_vmin'), _get_setter('_vmin'))
    vmax = property(_get_getter('_vmax'), _get_setter('_vmax'))
    mult_scale = property(_get_getter('_mult_scale'), _get_setter('_mult_scale'))
    log_scale = property(_get_getter('_log_scale'), _get_setter('_log_scale'))


def write_data(data_object, filename, xdata=None, ydata=None, mode='w', fmt='%15.10g', delimiter=' ', x_mode='error'):
    """
    pylag.write_data

    Write a data product (in a pylag data object) to disk in a text file.

    Each pylag plottable data product class (e.g. LagFrequecySpectrum, LagEnergySpectrum)
    has a number of variables that specify which arrays should be written as the
    x and y values and corresponding errors.

    e.g.
    >>> pylag.write_data(myLagEnergySpectrum, 'lag_energy.txt')

    Arguments
    ---------
    data_object : pylag plottable data product object
    filename    : string
                  The name of the file to be saved
    fmt            : string, optional (default='%15.10g')
                  Python string format specifier to set the formatting of
                  columns in the file.
    delimiter    : string, optional (default=' ')
                  The delimeter to use between columns
    """
    if data_object is not None:
        try:
            xd, yd = data_object._getplotdata()
        except:
            raise ValueError('pylag write_data ERROR: The object I was passed does not seem to be plottable')
    else:
        xd, yd = xdata, ydata

    data = []

    if isinstance(xd, tuple):
        if x_mode == 'error':
            data.append(xd[0])
            if isinstance(xd[1], (np.ndarray, list)):
                data.append(xd[1])
        elif x_mode == 'edges':
            if isinstance(xd[1], (np.ndarray, list)):
                data.append(xd[0] - xd[1])
                data.append(xd[0] + xd[1])
    else:
        data.append(xd)

    if isinstance(yd, tuple):
        data.append(yd[0])
        if isinstance(yd[1], (np.ndarray, list)):
            data.append(yd[1])
    else:
        data.append(yd)

    #data = np.array(data).transpose()
    #data = tuple(data)

    np.savetxt(filename, list(zip(*data)), fmt=fmt, delimiter=delimiter)


def write_multi_data(data_list, filename, mode='w', fmt='%15.10g', delimiter=' ', series_labels=None, xlabel='x'):
    """
    pylag.write_multi_data

    Write multiple data products (in a pylag data objects) to disk in a text file.

    The x-axis is taken from the first object then each data series is written as a
    subsequent column (with its corresponding error, if applicable)

    e.g.
    >>> pylag.write_multi_data([data1, data2], 'mydata.txt')

    Arguments
    ---------
    data_list   : list of pylag plottable data product objects
    filename    : string
                  The name of the file to be saved
    fmt            : string, optional (default='%15.10g')
                  Python string format specifier to set the formatting of
                  columns in the file.
    delimiter    : string, optional (default=' ')
                  The delimeter to use between columns
    """
    # the x-axis comes from the first object in the list
    xd, _ = data_list[0]._getplotdata()
    if isinstance(xd, tuple):
        data_len = len(xd[0])
        data = [xd[0]]
        if isinstance(xd[1], (np.ndarray, list)):
            data.append(xd[1])
    else:
        data_len = len(xd)
        data = [xd]

    # then add columns for each data series (plus errors if applicable)
    for obj in data_list:
        _, yd = obj._getplotdata()
        if isinstance(yd, tuple):
            if len(yd[0]) != data_len:
                raise AssertionError('Data series must be the same length and have common x-axis!')

            data.append(yd[0])
            if isinstance(yd[1], (np.ndarray, list)):
                data.append(yd[1])
        else:
            data.append(yd)

    #data = tuple(data)

    fieldstr = ''
    if series_labels is not None:
        fields = [xlabel]
        xd, _ = data_list[0]._getplotdata()
        if isinstance(xd, tuple):
            fields.append('+-')
        for obj, label in zip(data_list, series_labels):
            fields.append(label)
            _, yd = obj._getplotdata()
            if isinstance(yd, tuple):
                fields.append('+-')
        fieldstr = ' '.join(fields)

    np.savetxt(filename, list(zip(*data)), fmt=fmt, delimiter=delimiter, header=fieldstr)


def close_all_plots():
    """
    pylag.close_all_plots()

    Closes all open plot windows and deletes the objects from memory
    """
    for p in all_plots:
        p.close()
    del all_plots[:]


def plot_txt(filename, xcol=1, ycol=2, xerrcol=None, yerrcol=None, transpose=False, **kwargs):
    """
    p = pylag.plot_txt(filename, xcol=1, ycol=2, xerrcol=None, yerrcol=None, **kwargs)

    Create a plot from a text file with data in columns.

    The columns are read using the numpy genfromtxt function and a Plot object is created
    using specified columns. The new plot is returned and also added to the pylag.all_plots
    list.

    Parameters
    ----------
    filename : string
               Name of text file to be plotted
    xcol     : int, optional (default = 1)
               Number of the column in the file (starting at 1) containing the x data points
    ycol     : intm optional (default = 2)
               Number of the column containing the y data points
    xerrcol  : int, optional (default = None)
               If not None, the values in this column will be used as the x error bars
    yerrcol  : int, optional (default = None)
               If not None, the values in this column will be used as the y error bars
    kwargs   : Arguments passed to Plot constructor

    Returns
    -------
    p : Plot object containing the plot that was created
    """
    dat = np.genfromtxt(filename)
    if transpose:
        dat = dat.T
    if xerrcol is not None:
        xdata = (dat[:,xcol-1], dat[:,xerrcol-1])
    else:
        xdata = dat[:,xcol-1]
    if yerrcol is not None:
        ydata = (dat[:,ycol-1], dat[:,yerrcol-1])
    else:
        ydata = dat[:,ycol-1]

    return Plot(xdata=xdata, ydata=ydata, **kwargs)


def read_txt(filename, xcol=1, ycol=2, xerrcol=None, yerrcol=None, transpose=False, **kwargs):
    """
    ds = pylag.read_txt(filename, xcol=1, ycol=2, xerrcol=None, yerrcol=None, **kwargs)

    Create a DataSeries from a text file with data in columns.

    The columns are read using the numpy genfromtxt function and a DaatSeries object is created
    using specified columns.

    Parameters
    ----------
    filename : string
               Name of text file to be plotted
    xcol     : int, optional (default = 1)
               Number of the column in the file (starting at 1) containing the x data points
    ycol     : intm optional (default = 2)
               Number of the column containing the y data points
    xerrcol  : int, optional (default = None)
               If not None, the values in this column will be used as the x error bars
    yerrcol  : int, optional (default = None)
               If not None, the values in this column will be used as the y error bars
    kwargs   : Arguments passed to Plot constructor

    Returns
    -------
    ds : DataSeries object containing the plot that was created
    """
    dat = np.genfromtxt(filename)
    if transpose:
        dat = dat.T
    if xerrcol is not None:
        xdata = (dat[:,xcol-1], dat[:,xerrcol-1])
    else:
        xdata = dat[:,xcol-1]
    if yerrcol is not None:
        ydata = (dat[:,ycol-1], dat[:,yerrcol-1])
    else:
        ydata = dat[:,ycol-1]

    return DataSeries(x=xdata, y=ydata, **kwargs)


class DataSeries(object):
    def __init__(self, x=np.array([]), y=np.array([]), xlabel='', xscale='linear', ylabel='', yscale='linear'):
        self.xdata = x
        self.ydata = y

        self.xlabel = xlabel
        self.xscale = xscale
        self.ylabel = ylabel
        self.yscale = yscale

    def _getplotdata(self):
        return self.xdata, self.ydata

    def _getplotaxes(self):
        return self.xlabel, self.xscale, self.ylabel, self.yscale

    def append(self, x, y):
        if isinstance(self.xdata, tuple) and isinstance(x, tuple):
            self.xdata[0] = np.concatenate([self.xdata[0], x[0]])
            self.xdata[1] = np.concatenate([self.xdata[1], x[1]])
            self.ydata[0] = np.concatenate([self.ydata[0], y[0]])
            self.ydata[1] = np.concatenate([self.ydata[1], y[1]])
        elif isinstance(self.xdata, (np.ndarray, list)) and isinstance(x, (np.ndarray, list)):
            self.xdata = np.concatenate([self.xdata, x])
            self.ydata = np.concatenate([self.ydata, y])
        else:
            raise AssertionError('Data format mismatch')

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            if isinstance(self.ydata, tuple):
                mul_ydata = (other * self.ydata[0], other * self.ydata[1])
            else:
                mul_ydata = other * self.ydata
            return DataSeries(self.xdata, mul_ydata, self.xlabel, self.xscale, self.ylabel, self.yscale)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, DataSeries):
            if isinstance(self.ydata, tuple):
                yd1 = self.ydata[0]
                ye1 = self.ydata[1]
            elif isinstance(other.ydata, (np.ndarray, list)):
                yd1 = self.ydata
                ye1 = np.zeros(len(self.ydata))
            else:
                raise AssertionError('Data format mismatch')

            if isinstance(other.ydata, tuple):
                yd2 = other.ydata[0]
                ye2 = other.ydata[1]
            elif isinstance(other.ydata, (np.ndarray, list)):
                yd2 = other.ydata
                ye2 = np.zeros(len(other.ydata))
            else:
                raise AssertionError('Data format mismatch')

            if len(yd1) != len(yd2):
                raise AssertionError('Data series must be the same length')

            ratio = yd1 / yd2
            if isinstance(self.ydata, tuple) or isinstance(other.ydata, tuple):
                ratio_err = ratio * np.sqrt((ye1 / yd1) ** 2 + (ye2 / yd2) ** 2)
                ratio = (ratio, ratio_err)

            return DataSeries(self.xdata, ratio, self.xlabel, self.xscale, '%s / %s' % (self.ylabel, other.ylabel), self.yscale)
        else:
            return NotImplemented




class Spectrum(object):
    def __init__(self, en, spec, err=None, xlabel='Energy / keV', xscale='log', ylabel='Count Rate', yscale='log'):
        self.en = en
        self.spec = spec
        self.error = err

        self.xlabel = xlabel
        self.xscale = xscale
        self.ylabel = ylabel
        self.yscale = yscale

    def _getplotdata(self):
        if self.error is not None:
            return self.en, (self.spec, self.error)
        else:
            return self.en, self.spec

    def _getplotaxes(self):
        return self.xlabel, self.xscale, self.ylabel, self.yscale

    def rebin2(self, Nen=None, logbin=True, den=None):
        if logbin:
            if den is None:
                den = np.exp(np.log(self.en.max() / self.en.min()) / (float(Nen) - 1.))
            en_bin = self.en.min() * den ** np.arange(0, Nen+1, 1)
        else:
            if den is None:
                den = (self.en.max() - self.en.min()) / (float(Nen) - 1.)
            en_bin = np.arange(self.en.min(), self.en.max() + 2*den, den)

        spec_bin,_,_ = binned_statistic(self.en, self.spec, statistic='mean', bins=en_bin)
        return Spectrum(en=en_bin, spec=spec_bin, xlabel=self.xlabel, xscale=self.xscale, ylabel=self.ylabel,
                        yscale=self.yscale)

    def moving_average(self, window_size=3):
        window = np.ones(int(window_size)) / float(window_size)
        spec_avg = np.convolve(self.spec, window, 'same')
        return Spectrum(en=self.en, spec=spec_avg, xlabel=self.xlabel, xscale=self.xscale, ylabel=self.ylabel,
                        yscale=self.yscale)

    def abs(self):
        return Spectrum(en=self.en, spec=np.abs(self.spec), xlabel=self.xlabel, xscale=self.xscale, ylabel=self.ylabel,
                        yscale=self.yscale)

    def __truediv__(self, other):
        if isinstance(other, Spectrum):
            if len(self.spec) == len(other.spec):
                ratio = self.spec / other.spec

                if self.error is not None and other.error is not None:
                    err = ratio * np.sqrt((self.error/self.spec)**2 + (other.error/other.spec)**2)
                elif self.error is not None and other.error is None:
                    err = ratio * (self.error / self.spec)
                elif other.error is not None and self.error is None:
                    err = ratio * (other.error / other.spec)
                else:
                    err = None

                return Spectrum(en=self.en, spec=ratio, err=err, xlabel=self.xlabel, xscale=self.xscale, ylabel='Ratio',
                        yscale='linear')
            else:
                raise AssertionError("Lengths of spectra do not match")
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Spectrum):
            if len(self.spec) == len(other.spec):
                sub = self.spec - other.spec

                if self.error is not None and other.error is not None:
                    err = np.sqrt(self.error**2 + other.error**2)
                elif self.error is not None and other.error is None:
                    err = self.error
                elif other.error is not None and self.error is None:
                    err = other.error
                else:
                    err = None

                return Spectrum(en=self.en, spec=sub, err=err, xlabel=self.xlabel, xscale=self.xscale, ylabel=self.ylabel,
                        yscale=self.yscale)
            else:
                raise AssertionError("Lengths of spectra do not match")
        else:
            return NotImplemented

    def rebin(self, bins=None, Nbins=None):
        if bins is None:
            bins = LogBinning(self.en.min(), self.en.max(), Nbins)
        spec_bin = bins.bin(self.en, self.spec)
        return Spectrum(en=bins.bin_cent, spec=spec_bin, xlabel=self.xlabel, xscale=self.xscale, ylabel=self.ylabel,
                        yscale=self.yscale)


def dataset_ratio(ds1, ds2):
    x1, y1 = ds1._getplotdata()
    x2, y2 = ds2._getplotdata()

    xlabel, xscale, ylabel, yscale = ds1._getplotaxes()

    if isinstance(y1, tuple):
        yd1 = y1[0]
        ye1 = y1[1]
    elif isinstance(y1, (np.ndarray, list)):
        yd1 = y1
        ye1 = np.zeros(len(yd1))
    else:
        raise AssertionError('Data format mismatch')

    if isinstance(y2, tuple):
        yd2 = y2[0]
        ye2 = y2[1]
    elif isinstance(y1, (np.ndarray, list)):
        yd2 = y2
        ye2 = np.zeros(len(yd2))
    else:
        raise AssertionError('Data format mismatch')

    ratio = yd1 / yd2
    if isinstance(y1, tuple) or isinstance(y2, tuple):
        ratio_err = ratio * np.sqrt( (ye1/yd1)**2 + (ye2/yd2)**2 )
        ratio = (ratio, ratio_err)

    return DataSeries(x1, ratio, xlabel, xscale, 'ratio', 'linear')


def dataset_difference(ds1, ds2):
    x1, y1 = ds1._getplotdata()
    x2, y2 = ds2._getplotdata()

    xlabel, xscale, ylabel, yscale = ds1._getplotaxes()

    if isinstance(y1, tuple):
        yd1 = y1[0]
        ye1 = y1[1]
    elif isinstance(y1, (np.ndarray, list)):
        yd1 = y1
        ye1 = np.zeros(len(yd1))
    else:
        raise AssertionError('Data format mismatch')

    if isinstance(y2, tuple):
        yd2 = y2[0]
        ye2 = y2[1]
    elif isinstance(y1, (np.ndarray, list)):
        yd2 = y2
        ye2 = np.zeros(len(yd2))
    else:
        raise AssertionError('Data format mismatch')

    diff = yd1 - yd2
    if isinstance(y1, tuple) or isinstance(y2, tuple):
        diff_err = np.sqrt( ye1**2 + ye2**2 )
        diff = (diff, diff_err)

    return DataSeries(x1, diff, xlabel, xscale, 'difference', yscale)


def dataset_frac_difference(ds1, ds2):
    x1, y1 = ds1._getplotdata()
    x2, y2 = ds2._getplotdata()

    xlabel, xscale, ylabel, yscale = ds1._getplotaxes()

    if isinstance(y1, tuple):
        yd1 = y1[0]
        ye1 = y1[1]
    elif isinstance(y1, (np.ndarray, list)):
        yd1 = y1
        ye1 = np.zeros(len(yd1))
    else:
        raise AssertionError('Data format mismatch')

    if isinstance(y2, tuple):
        yd2 = y2[0]
        ye2 = y2[1]
    elif isinstance(y1, (np.ndarray, list)):
        yd2 = y2
        ye2 = np.zeros(len(yd2))
    else:
        raise AssertionError('Data format mismatch')

    diff = (yd1 - yd2) / np.abs(yd2)
    if isinstance(y1, tuple) or isinstance(y2, tuple):
        diff_err = diff * np.sqrt((ye1 + ye2)/(yd1 - yd2)**2 + (ye2/yd2)**2)
        diff = (diff, diff_err)

    return DataSeries(x1, diff, xlabel, xscale, 'frac. difference', yscale)


class AverageDataSeries(DataSeries):
    def __init__(self, data_objects):
        self.xdata = data_objects[0]._getplotdata()[0]
        if isinstance(data_objects[0]._getplotdata()[1], tuple):
            self.mean = np.mean(np.array([o._getplotdata()[1][0] for o in data_objects]), axis=0)
            self.std = np.std(np.array([o._getplotdata()[1][0] for o in data_objects]), axis=0)
        else:
            self.mean = np.mean(np.array([o._getplotdata()[1] for o in data_objects]), axis=0)
            self.std = np.std(np.array([o._getplotdata()[1] for o in data_objects]), axis=0)

        self.ydata = (self.mean, self.std)

        self.xlabel, self.xscale, self.ylabel, self.yscale = data_objects[0]._getplotaxes()


def txt_to_ds(filename, xcol=1, ycol=2, xerrcol=None, yerrcol=None, transpose=False, **kwargs):
    """
    p = pylag.txt_to_ds(filename, xcol=1, ycol=2, xerrcol=None, yerrcol=None, **kwargs)

    Create a DataSet object from a text file with data in columns.

    The columns are read using the numpy genfromtxt function and a Plot object is created
    using specified columns. The new plot is returned and also added to the pylag.all_plots
    list.

    Parameters
    ----------
    filename : string
               Name of text file to be plotted
    xcol     : int, optional (default = 1)
               Number of the column in the file (starting at 1) containing the x data points
    ycol     : intm optional (default = 2)
               Number of the column containing the y data points
    xerrcol  : int, optional (default = None)
               If not None, the values in this column will be used as the x error bars
    yerrcol  : int, optional (default = None)
               If not None, the values in this column will be used as the y error bars
    kwargs   : Arguments passed to Plot constructor

    Returns
    -------
    p : Plot object containing the plot that was created
    """
    dat = np.genfromtxt(filename)
    if transpose:
        dat = dat.T
    if xerrcol is not None:
        xdata = (dat[:,xcol-1], dat[:,xerrcol-1])
    else:
        xdata = dat[:,xcol-1]
    if yerrcol is not None:
        ydata = (dat[:,ycol-1], dat[:,yerrcol-1])
    else:
        ydata = dat[:,ycol-1]

    return DataSeries(x=xdata, y=ydata, **kwargs)


class Histogram(DataSeries):
    def __init__(self, x, bins=100, min=None, max=None, logbin=False, density=True):
        if isinstance(b, Binning):
            self.bins = bins
        else:
            if min is None:
                min = np.min(x)
            if max is None:
                max = np.max(x)
            self.bins = LogBinning(min, max, bins) if logbin else LinearBinning(min, max, bins)

        self.hist, _ = np.histogram(x, bins=self.bins.bin_edges, density=density)

    def _getplotdata(self):
        if self.error is not None:
            return self.en, (self.spec, self.error)
        else:
            return self.bins.bin_cent, self.hist

    def _getplotaxes(self):
        return self.xlabel, self.xscale, self.ylabel, self.yscale


