"""
pylag.lag_energy_spectrum

Provides pyLag class for automating calculation of the lag-energy spectrum

Classes
-------
LagEnergySpectrum : Calculate the lag-energy spectrum from a set of light curves
                    in different energy bands

v1.0 09/03/2017 - D.R. Wilkins
"""
from pylag.lightcurve import *
from pylag.cross_spectrum import *
from pylag.coherence import *
from pylag.binning import *
from pylag.plotter import *

import numpy as np
import glob
import re

class LagEnergySpectrum(object):
	"""
	pylag.LagEnergySpectrum

	Class for computing the lag-energy spectrum from a set of light curves, one
	in each energy band (or a set of light curve segments in each energy band),
	relative to a reference band that is the summed time series over all energy
	bands. The lag at each energy is averaged over some	frequency range. For each
	energy band, the present energy range is subtracted from the reference band
	to avoid correlated noise.

	The resulting lag-enery spectrum is stored in the member variables.

	This class automates calculation of the lag-energy spectrum and its errors
	from the CrossSpectrum (or StackedCrossSpectrum) and the Coherence classes.

	Member Variables
	----------------
	en       : ndarray
	           numpy array containing the central energy of each band
	en_error : ndarray
	           numpy array containing the error bar of each energy band (the
			   central energy minus the minimum)
	lag      : ndarray
	           numpy array containing the time lag of each energy band relative
			   to the reference band
	error    : ndarray
	           numpy array containing the error in the lag measurement in each
			   band

	Constructor: pylag.LagEnergySpectrum(lclist, fmin, fmax, enmin, enmax, lcfiles, interp_gaps=False, refband=None)

	Constructor Arguments
	---------------------
	fmin        : float
				  Lower bound of frequency range
	fmin          : float
		  		  Upper bound of frequency range
	lclist      : list of LightCurve objects or list of lists of LightCurve objects
	          	  optional (default=None)
	          	  This is either a 1-dimensional list containing the pylag
			  	  LightCurve objects for the light curve in each of the energy bands,
			  	  i.e. [en1_lc, en2_lc, ...]
			  	  or a 2-dimensional list (i.e. list of lists) if multiple observation
			  	  segments are to be stacked. In this case, the outer index
				  corresponds to the energy band. For each energy band, there is a
				  list of LightCurve objects that represent the light curves in that
				  energy band from each observation segment.
		 		  i.e. [[en1_obs1, en1_obs2, ...], [en2_obs1, en2_obs2, ...], ...]
	enmin       : ndarray or list, optional (default=None)
	              numpy array or list containing the lower energy bound of each band
			      (each entry corresponds to one light curve, in order)
	enmax       : ndarray or list, optional (default=None)
	              numpy array or list containing the upper energy bound of each band
			      (each entry corresponds to one light curve, in order)
	lcfiles     : string, optional (default='')
	              If not empty, the filesystem will be searched using this glob to
			      automatically build the list of light curves and energies
	interp_gaps : boolean (default=False)
				  Interpolate over gaps in the light curves
	refband     : list of floats
			      If specified, the reference band will be restricted to this
			      energy range [min, max]. If not specified, the full band will
			      be used for the reference
	"""
	def __init__(self, fmin, fmax, lclist=None, enmin=None, enmax=None, lcfiles='', interp_gaps=False, refband=None):
		self.en = np.array([])
		self.en_error = np.array([])
		self.lag = np.array([])
		self.error = np.array([])

		if(lcfiles != ''):
			enmin, enmax, lclist = self.FindLightCurves(lcfiles, interp_gaps=interp_gaps)

		print "Constructing lag energy spectrum from ", len(lclist[0]), " light curves in each of ", len(lclist), " energy bins"

		self.en = (0.5*(np.array(enmin) + np.array(enmax)))
		self.en_error = self.en - np.array(enmin)

		if isinstance(lclist[0], LightCurve):
			self.lag, self.error = self.Calculate(lclist, fmin, fmax, refband, self.en)
		elif isinstance(lclist[0], list) and isinstance(lclist[0][0], LightCurve):
			self.lag, self.error = self.CalculateStacked(lclist, fmin, fmax, refband, self.en)

		# pointers to the energy and lag arrays to specify the axes for Plotter
		self.xdata = self.en
		self.xerror = self.en_error
		self.ydata = self.lag
		self.yerror = self.error
		self.xlabel = 'Energy / keV'
		self.ylabel = 'Lag / s'
		self.xscale = 'log'
		self.yscale = 'linear'


	def Calculate(self, lclist, fmin, fmax, refband=None, energies=None):
		"""
		lag, error = pylag.LagEnergySpectrum.Calculate(lclist, fmin, fmax, refband=None, energies=None)

		Calculate the lag-energy spectrum from a list of light curves, one in
		each energy band, averaged over some frequency range.

		The lag is calculated with respect to a reference light curve that is
		computed as the sum of all energy bands, but subtracting the energy band
		of interest for each lag/energy point, so to avoid correlated noise
		between the subject and reference light curves.

		Arguments
		---------
		lclist   : list of LightCurve objects
				   1-dimensional list containing the pylag
   			       LightCurve objects for the light curve in each of the energy
				   bands, i.e. [en1_lc, en2_lc, ...]
		fmin     : float
				   Lower bound of frequency range
		fmin     : float
				   Upper bound of frequency range
		refband  : list of floats
				 : If specified, the reference band will be restricted to this
				   energy range [min, max]. If not specified, the full band will
				   be used for the reference
		energies : ndarray (default=None)
				 : If a specific range of energies is to be used for the reference
				   band rather than the full band, this is the list of central
				   energies of the bands represented by each light curve

		Return Values
		-------------
		lag   : ndarray
		        numpy array containing the lag of each energy band with respect
			    to the reference band
		error : ndarray
		        numpy array containing the error in each lag measurement
		"""
		reflc = LightCurve(t = lclist[0].time)
		for energy_num, lc in enumerate(lclist):
			if refband != None:
				if (energies[energy_num] < refband[0] or energies[energy_num] > refband[1]):
					continue
			reflc = reflc + lc

		lag = []
		error = []
		for lc in lclist:
			thisref = reflc - lc
			# if we're only using a specific reference band, we did not need to
			# subtract the current band if it's outside the range
			if refband != None:
				if (energies[energy_num] < refband[0] or energies[energy_num] > refband[1]):
					thisref = reflc
			lag.append( CrossSpectrum(lc, thisref).LagAverage(fmin, fmax) )
			error.append( Coherence(lc, reflc, fmin=fmin, fmax=fmax).LagError() )

		return np.array(lag), np.array(error)

	def CalculateStacked(self, lclist, fmin, fmax, refband=None, energies=None):
		"""
		lag, error = pylag.LagEnergySpectrum.CalculateStacked(lclist, fmin, fmax, refband=None, energies=None)

		Calculate the lag-energy spectrum from a list of light curves, averaged
		over some frequency range. The lag-energy spectrum is calculated from
		the cross spectrum and coherence stacked over multiple light curve
		segments in each energy band.

		The lag is calculated with respect to a reference light curve that is
		computed as the sum of all energy bands, but subtracting the energy band
		of interest for each lag/energy point, so to avoid correlated noise
		between the subject and reference light curves.

		Arguments
		---------
		lclist   : list of lists of LightCurve objects
				   This is a 2-dimensional list (i.e. list of lists). The outer index
				   corresponds to the energy band. For each energy band, there is a
				   list of LightCurve objects that represent the light curves in that
				   energy band from each observation segment.
				   i.e. [[en1_obs1, en1_obs2, ...], [en2_obs1, en2_obs2, ...], ...]
		fmin     : float
				   Lower bound of frequency range
		fmin     : float
				   Upper bound of frequency range
		refband  : list of floats
				 : If specified, the reference band will be restricted to this
				   energy range [min, max]. If not specified, the full band will
				   be used for the reference
		energies : ndarray (default=None)
				 : If a specific range of energies is to be used for the reference
				   band rather than the full band, this is the list of central
				   energies of the bands represented by each light curve

		Return Values
		-------------
		lag   : ndarray
		        numpy array containing the lag of each energy band with respect
			    to the reference band
		error : ndarray
		        numpy array containing the error in each lag measurement
		"""
		reflc = []
		# initialise a reference light curve for each of the observations/light
		# curve segments
		for lc in lclist[0]:
			reflc.append( LightCurve(t = lc.time) )
		# sum all of the energies (outer index) together to produce a reference
		# light curve for each segment (inner index)
		for energy_num, energy_lcs in enumerate(lclist):
			# if a reference band is specifed, skip any energies that do not fall
			# in that range
			if refband != None:
				if (energies[energy_num] < refband[0] or energies[energy_num] > refband[1]):
					continue
			for segment_num, segment_lc in enumerate(energy_lcs):
				reflc[segment_num] = reflc[segment_num] + segment_lc

		lag = []
		error = []
		for energy_num, energy_lclist in enumerate(lclist):
			# subtract this energy band from the reference light curve for each
			# light curve segment to be stacked (subtracting the current band
			# means we don't have correlated noise between the subject and
			# reference bands)
			ref_lclist = []
			for segment_num, segment_lc in enumerate(energy_lclist):
				# if a reference band is specifed and this energy falls outside that
				# band, no need to subtract the current band
				if refband != None:
					if (energies[energy_num] < refband[0] or energies[energy_num] > refband[1]):
						ref_lclist.append( reflc[segment_num] )
						continue
				ref_lclist.append( reflc[segment_num] - segment_lc )
			# now get the lag and error from the stacked cross spectrum and
			# coherence between the sets of lightcurves for this energy band and
			# for the reference
			lag.append( StackedCrossSpectrum(energy_lclist, ref_lclist).LagAverage(fmin, fmax) )
			error.append( Coherence(energy_lclist, ref_lclist, fmin=fmin, fmax=fmax).LagError() )

		return np.array(lag), np.array(error)

	@staticmethod
	def FindLightCurves(searchstr, **kwargs):
		"""
		enmin, enmax, lclist = pylag.LagEnergySpectrum.FindLightCurves(searchstr)

		Search the filesystem for light curve files and return a list of light
		curve segments for each available observation segment in each energy
		band. A 2-dimensional list of LightCurve objects from each segment
		(inner index) in each energy band (outer index) is returned,
		i.e. [[en1_obs1, en1_obs2, ...], [en2_obs1, en2_obs2, ...], ...]
		suitable for calcualation of a stacked lag-energy spectrum. Lists of
		lower and upper energies for each bin are also returned.

		If only one light curve is found for each energy band, a 1 dimensional
		list of light curves is returned.

		Light curves are sorted by lower energy bound, then alphabetically by
		filename such that if a common prefix convention is adopted identifying
		the segment, the segments listed in each energy bin will match up.

		Light curve filenames must have the substring enXXX-YYY where XXX and YYY
		are the lower and upper bounds of the energy bin in eV.
		e.g. obs1_src_en300-400.lc

		Note: Make sure that the search string returns only the light curves to
		be used in the observation and that there are the same number of segments
		in each energy band!

		Arguments
		---------
		searchstr : string
		          : Wildcard for searching the filesystem to find the light curve
				    filesystem

		Return Values
		-------------
		enmin :  ndarray
		         numpy array countaining the lower energy bound of each band
		enmax :  ndarray
		         numpy array containing the upper energy bound of each band
		lclist : list of list of LightCurve objects
		         The list of light curve segments in each energy band for
				 computing the lag-energy spectrum
		"""
		lcfiles = sorted(glob.glob(searchstr))
		enlist = list(set([re.search('(en[0-9]+\-[0-9]+)', lc).group(0) for lc in lcfiles]))

		enmin = []
		enmax = []
		for estr in enlist:
			matches = re.search('en([0-9]+)\-([0-9]+)', estr)
			enmin.append(float(matches.group(1)))
			enmax.append(float(matches.group(2)))
		# zip the energy bins to sort them, then unpack
		entuples = sorted(zip(enmin, enmax))
		enmin, enmax = zip(*entuples)

		lclist = []
		for emin, emax in zip(enmin, enmax):
			estr = 'en%d-%d' % (emin, emax)
			energy_lightcurves = sorted([lc for lc in lcfiles if estr in lc])
			# see how many light curves match this energy - if there's only one, we
			# don't want nested lists so stacking isn't run
			if(len(energy_lightcurves) > 1):
				energy_lclist = []
				for lc in energy_lightcurves:
					energy_lclist.append( LightCurve(lc, **kwargs) )
				lclist.append(energy_lclist)
			else:
				lclist.append( LightCurve(energy_lightcurves[0], **kwargs) )

		return np.array(enmin)/1000., np.array(enmax)/1000., lclist

	def Plot(self):
		"""
		pylag.LagEnergySpectrum.Plot()

		Plot the spectrum to the screen using matplotlib.
		"""
		return Plotter(self)
