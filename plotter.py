"""
pylag.Plotter

Plotting and data output classes/functions for pylag data products
"""
import matplotlib.pyplot as plt
import numpy as np

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

	The plot can be saved to a file using the Save() method. For automated
	processing, set show_plot=False in the constructor so that a plot can be
	created or updated and then saved using Save() without it being displayed
	on screen.

	Member Variables
	----------------
	fig		: matplotlib Figure instance
	ax		: matplotlib Axes handle

	Data Series:
	xdata	      : ndarray or list of ndarrays
			  		x data points to be plotted (or a list of arrays, each containing
			  	    the x data points for one of the series to be plotted)
	xerror  	  : ndarray or list of ndarrays
			  		Symmetric error on x data points
	ydata		  : ndarray or list of ndarrays
			        y data points
	yerror        : ndarray or list of ndarrays
			        Symmetric error on y data points
	series_labels : list of strings, optional, default=[]
					The label for each data series shown in the legend. Each
					entry should correspond to one of the data objects or one
					of the manually specified x,y series

	Plot Options:
	title			: string
					  Title to be displayed at the top of the plot

	Axes:
	xlabel 			: string
					  x axis label
	ylabel			: string
					  y axis label
	xscale 			: string
					  'linear' or 'log' to set scaling of x axis
	yscale 			: string
					  'linear' or 'log' to set scaling of x axis
	xlim			: list of floats [min, max] (default=None)
					  If set, manually specify the limits of the x axis. If None,
					  the axis will be scaled automatically
	ylim			: list of floats [min, max] (default=None)
					  If set, manually specify the limits of the y axis. If None,
					  the axis will be scaled automatically

	Formatting:
	grid 			: string (default='minor')
					  Specify which grid lines are shown: 'major', 'minor' or 'none'
	legend			: boolean
					  If True, a legend is shown on the plot. By default, set
					  to True if series labels are provided
	legend_loc 		: string (default='upper right')
					  The matplotlib string specifying the location the legend
					  should be placed ('best', 'upper right', 'center right',
					  'lower left' etc.)
	colour_series   : list of strings (default=['k', 'b', 'g', 'r', 'c', 'm'])
					  The repeating sequence of matplotlib colour specifiers
					  setting the order in which colours are assigned to plot
					  series. The sequence is repeated as many times as necessary
					  to cover all the series
	marker_series 	: list of strings (default=['+', 'x', 'o', 's'])
					  The repeating sequence of matplotlib plot marker specifiers
					  setting the order in which they are applied to data series.
					  To plot all series as lines, use a single entry ['-']
	font_face		: string (default=None)
					  Specify the font face. If None, use the matplotlub default
	font_size		: integer (default=None)
					  Specify the font size. If None, use the matplotlub default

	Constructor: p = pylag.Plot(, xdata=None, xerr=None, ydata=None, yerr=None, xscale='', yscale='', xlabel='', ylabel='', title='', series_labels=[], preset=None, show_plot=True)

	Constructor Arguments
	---------------------
	data_object   : pyLag plottable data product object or list of objects
				    optional (default) = None
				    If set, the plot is automatically produced from the object.
				    If a list of objects is passed, each one is plotted as a
				    separate data series on the plot
	xdata		  : ndarray or list of ndarray, optional (default=None)
				    If data_object is not set, the x co-ordinates of the series to
				    be plotted. If a list of arrays is passed, each is plotted as
				    a separate data series
	xerr		  : ndarray or list of ndarray, optional (default=None)
				    The symmetric error bars on the x values
	ydata		  : ndarray or list of ndarray, optional (default=None)
				    If data_object is not set, the y co-ordinates of the series to
				    be plotted. If a list of arrays is passed, each is plotted as
				    a separate data series - should line up with xdata
	yerr		  : ndarray or list of ndarray, optional (default=None)
				    The symmetric error bars on the y values
	xscale		  : string, 'linear' or 'log', optional (default='')
				    If set, override the default x axis scaling specified by the
				    data object to be plotted (or the 'linear' default for manually
				    specified data series)
	yscale		  : string, 'linear' or 'log', optional (default='')
				    If set, override the default y axis scaling
	xlabel		  : string, optional (default='')
				    If set, override the default x axis label set by the data object
	ylabel		  : string, optional (default='')
				    If set, override the default y axis label set by the data object
	title	   	  : string, optional (default='')
				    The title to be shown at the top of the plot
	series_labels : list of strings, optional, default=[]
					The label for each data series shown in the legend. Each
					entry should correspond to one of the data objects or one
					of the manually specified x,y series. If set, a legend will
					be displayed on the plot, if not, the legend is hidden
	preset		  : for future use
	show_plot	  : boolean, optional (default=True)
					display the plot window on screen automatically when the plot
					is created or updated. If False, the plot can be displayed
					by calling the Show method()
	"""
	def __init__(self, data_object=None, xdata=None, xerr=None, ydata=None, yerr=None, xscale='', yscale='', xlabel='', ylabel='', title='', series_labels=[], preset=None, show_plot=True):
		self.fig = None
		self.ax = None

		self.xdata = []
		self.xerror = []
		self.ydata = []
		self.yerror = []

		self.series_labels = list(series_labels)

		self.xlabel = ''
		self.ylabel = ''

		self.title = title

		self.xscale = 'linear'
		self.yscale = 'linear'

		# variables to set plot formatting
		self.colour_series = ['k', 'b', 'g', 'r', 'c', 'm']
		self.marker_series = ['+', 'x', 'o', 's']
		self.font_face = None
		self.font_size = None
		self.grid = 'minor'
		self.legend_loc = 'upper right'
		self.xlim = None
		self.ylim = None

		# do we display the plot on screen automatically when calling Plot()?
		self.show_plot = show_plot

		self.legend = (len(self.series_labels)>0)

		if(data_object != None):
			if not isinstance(data_object, list):
				# if we're only given one object, put it in a list
				data_object = [data_object]
			# go through each object and put in the pointers to the x/y
			# data series and their errors
			for obj in data_object:
				try:
					xd, yd, xe, ye = obj._getplotdata()
					self.xdata.append(xd)
					self.ydata.append(yd)
					self.xerror.append(xe)
					self.yerror.append(ye)
					# if no labels are passed in for the series, use a blank string
					if(len(series_labels)==0):
						self.series_labels.append('')
				except:
					raise ValueError('pylag Plotter ERROR: The object I was passed does not seem to be plottable')
			# read the axis labels from data_object
			self.xlabel, self.xscale, self.ylabel, self.yscale = data_object[0]._getplotaxes()


		else:
			# if we're not passed an object, use the data series that are passed in
			if not isinstance(xdata, list):
				xdata = [xdata]
			if not isinstance(ydata, list) :
				ydata = [ydata]
			if len(xdata) != len(ydata):
				raise ValueError('pylag Plotter ERROR: I need the same number of data series for x and y!')

			if(xerr != None):
				if not isinstance(xerr, list):
					xerr = [xerr]
				if len(xerr) != len(xdata):
					raise ValueError('pylag Plotter ERROR: I need the same number of data series for x and xerror!')

			if(yerr != None):
				if not isinstance(yerr, list):
					yerr = [yerr]
				if len(yerr) != len(ydata):
					raise ValueError('pylag Plotter ERROR: I need the same number of data series for y and yerror!')

			for xd, yd in zip(xdata,ydata):
				if len(xd) != len(yd):
					raise ValueError('pylag Plotter ERROR: I need the same number of data points in x and y!')
				self.xdata.append(xd)
				self.ydata.append(yd)
				if(xerr==None):
					self.xerror.append(None)
				if(yerr==None):
					self.yerror.append(None)
				if(len(labels)==0):
					self.series_labels.append('')

		# if we're passed axis labels, these override the labels set in data_object
		if(xlabel != ''):
			self.xlabel = xlabel
		if(ylabel != ''):
			self.ylabel = ylabel
		# if we're passed axis log/linear scaling, these override the scaling set in data_object
		if(xscale != ''):
			self.xscale = xscale
		if(ylabel != ''):
			self.yscale = yscale

		self.Plot()


	def SetupAxes(self):
		"""
		pylag.Plot.SetupAxes()

		Close and recreate the figure and axes, applying the updated settings.
		This function is called automatically by Plot()
		"""
		# close the old figure (if already plotted)
		if(self.fig != None):
			self.Close()
		# create a new figure window and axes
		self.fig, self.ax = plt.subplots()

		# if specified, set the font
		if(self.font_face != None):
			plt.rc('font', **{'family' : self.font_face})
		if(self.font_size != None):
			plt.rc('font', **{'size' : self.font_size})

		# set log or linear scaling
		self.ax.set_xscale(self.xscale)
		self.ax.set_yscale(self.yscale)

		# set axis labels
		self.ax.set_xlabel(self.xlabel)
		self.ax.set_ylabel(self.ylabel)
		self.ax.set_title(self.title)

		# turn major/minor grid lines on or off
		if(self.grid == 'major' or self.grid == 'minor'):
			self.ax.grid(which='major')
		if(self.grid == 'minor'):
			self.ax.grid(which='minor')

		# set the axis ranges if set
		if(self.xlim != None):
			self.ax.set_xlim(self.xlim)
		if(self.ylim != None):
			self.ax.set_ylim(self.ylim)

	def Plot(self, **kwargs):
		"""
		pylag.Plot.Plot()

		Wrapper function to perform all steps necessary to create/update the plot.
		If show_plot is True, display the plot window at the end.

		The PlotData() function is called to add the data points to the plot.
		This may be overriden in derived classes to produce different plot types.
		"""
		self.SetupAxes()
		self.PlotData(**kwargs)
		if self.legend:
			self.ax.legend(loc=self.legend_loc)
		if self.show_plot:
			self.Show()

	def PlotData(self):
		"""
		pylag.Plot.PlotData()

		Add each data series to the plot as points (or lines) with error bars
		"""
		# repeat the colour and marker series as many times as necessary to provide for all the data series
		colours = (self.colour_series * (len(self.xdata)/len(self.colour_series) + 1))[:len(self.xdata)]
		markers = (self.marker_series * (len(self.xdata)/len(self.colour_series) + 1))[:len(self.xdata)]

		# plot the data series in turn
		for xd, yd, yerr, xerr, marker, colour, label in zip(self.xdata, self.ydata, self.yerror, self.xerror, markers, colours, self.series_labels):
			if not isinstance(xerr, (np.ndarray, list)):
				xerr = np.zeros(len(xd))
			if not isinstance(yerr, (np.ndarray, list)):
				yerr = np.zeros(len(yd))
			self.ax.errorbar(xd[np.isfinite(yd)], yd[np.isfinite(yd)], yerr=yerr[np.isfinite(yd)], xerr=xerr[np.isfinite(yd)], fmt=marker, color=colour, label=label)

	def Show(self, **kwargs):
		"""
		pylag.Plot.Show()

		Show the plot window on the screen
		"""
		self.fig.show(**kwargs)

	def Close(self):
		"""
		pylag.Plot.Show()

		Close this plot's window
		"""
		plt.close(self.fig)

	def Save(self, filename, **kwargs):
		"""
		pylag.Plot.Show()

		Save the plot to a file using matplotlib's savefig() function.

		Arguments
		---------
		filename : string
				   The name of the file to be created
		**kwargs : passed on to matplotlib savefig() function
		"""
		self.fig.savefig(filename, bbox_inches='tight', **kwargs)


class ErrorRegionPlot(Plot):
	"""
	pylag.ErrorRegionPlot

	Class to plot data objects or series as a shaded error region.

	See Plot class for details
	"""
	def PlotData(self, use_xerror=False, alpha=0.5):
		"""
		pylag.ErrorRegionPlot.PlotData()

		Add each data series to the plot as a shaded error region
		"""
		# repeat the colour series as many times as necessary to provide for all the data series
		colours = (self.colour_series * (len(self.xdata)/len(self.colour_series) + 1))[:len(self.xdata)]
		# plot each data series in turn
		for xd, yd, yerr, xerr, colour, label in zip(self.xdata, self.ydata, self.yerror, self.xerror, colours, self.series_labels):
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
				self.ax.plot(xd, yd, '-', color=colour, label=label)
				self.ax.fill_between(xpoints, low_bound, high_bound, facecolor=colour, alpha=alpha)
			else:
				high_bound = np.array(yd) + np.array(yerr)
				low_bound = np.array(yd) - np.array(yerr)
				self.ax.plot(xd, yd, '-', color=colour, label=label)
				self.ax.fill_between(xd, low_bound, high_bound, facecolor=colour, alpha=alpha)


def WriteData(data_object, filename, fmt='%15.10g', delimiter=' '):
	"""
	pylag.WriteData

	Write a data product (in a pylag data object) to disk in a text file.

	Each pylag plottable data product class (e.g. LagFrequecySpectrum, LagEnergySpectrum)
	has a number of variables that specify which arrays should be written as the
	x and y values and corresponding errors.

	e.g.
	>>> pylag.WriteData(myLagEnergySpectrum, 'lag_energy.txt')

	Arguments
	---------
	data_object : pylag plottable data product object
	filename	: string
				  The name of the file to be saved
	fmt			: string, optional (default='%15.10g')
				  Python string format specifier to set the formatting of
				  columns in the file.
	delimiter	: string, optional (default=' ')
				  The delimeter to use between columns
	"""
	data = [data_object.xdata]
	if isinstance(data_object.xerror, (np.ndarray, list)):
		data.append(data_object.xerror)
	data.append(data_object.ydata)
	if isinstance(data_object.yerror, (np.ndarray, list)):
		data.append(data_object.yerror)

	data = np.array(data).transpose()

	np.savetxt(filename, data, fmt=fmt, delimiter=delimiter)
