"""
pylag.time

Time conversion utilities and time classes to add to astropy.time for various missions
"""
import astropy.time
import erfa


class TimeXmmSec(astropy.time.TimeFromEpoch):
    """
    XMM seconds from 1998-01-01 00:00:00 TT.
    """
    name = "xmmsec"
    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)
    epoch_val = "1998-01-01 00:00:00"
    epoch_val2 = None
    epoch_scale = "tt"
    epoch_format = "iso"


class TimeNustarSec(astropy.time.TimeFromEpoch):
    """
    NuSTAR seconds from 2010-01-01 00:00:00 TT.
    """
    name = "nustarsec"
    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)
    epoch_val = "2010-01-01 00:00:00"
    epoch_val2 = None
    epoch_scale = "tt"
    epoch_format = "iso"
