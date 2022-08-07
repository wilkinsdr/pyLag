from .lightcurve import *

__PYLAG_CHATTER__ = 0


def printmsg(level, *args):
    global __PYLAG_CHATTER__
    if level <= __PYLAG_CHATTER__:
        print(*args)


def chatter(level=None):
    global __PYLAG_CHATTER__
    if level is not None:
        __PYLAG_CHATTER__ = level
    return __PYLAG_CHATTER__


def get_nustar_lclist(src_files_fpma='nu*A01_sr.lc', bkg_files_fpma='nu*A01_bk.lc', src_files_fpmb='nu*B01_sr.lc', bkg_files_fpmb='nu*B01_bk.lc', **kwargs):
    #
    # returns list of background-subtracted NuSTAR light curves
    #
    src_lc_fpma = get_lclist(src_files_fpma, add_tstart=True, **kwargs)
    bkg_lc_fpma = get_lclist(bkg_files_fpma, add_tstart=True, **kwargs)
    src_lc_fpmb = get_lclist(src_files_fpmb, add_tstart=True, **kwargs)
    bkg_lc_fpmb = get_lclist(bkg_files_fpmb, add_tstart=True, **kwargs)

    sub_lc_fpma = []
    for s, b, in zip(src_lc_fpma, bkg_lc_fpma):
        s_sim, b_sim = extract_sim_lightcurves(s, b)
        sub_lc_fpma.append(s_sim - b_sim)

    sub_lc_fpmb = []
    for s, b, in zip(src_lc_fpmb, bkg_lc_fpmb):
        s_sim, b_sim = extract_sim_lightcurves(s, b)
        sub_lc_fpmb.append(s_sim - b_sim)

    sum_lc = []
    for a, b in zip(sub_lc_fpma, sub_lc_fpmb):
        a_sim, b_sim = match_lc_timebins(a, b)
        sum_lc.append(a_sim + b_sim)

    return sum_lc, sub_lc_fpma, sub_lc_fpmb


def orbit_lightcurve(lc, error_mode='counts'):
    from .lightcurve import VariableBinLightCurve
    orbit_time = []
    orbit_time_err = []
    orbit_rate = []
    orbit_rate_err = []

    tdiff = np.diff(lc.time)
    tbin = np.min(tdiff)

    t_points = [lc.time[0]]
    r_points = [lc.rate[0]]
    counts = lc.rate[0]
    for i in range(len(lc) - 1):
        if tdiff[i] > tbin:
            orbit_time.append(np.mean(t_points))
            orbit_time_err.append(np.max(t_points) - np.mean(t_points))
            orbit_time_interval = np.max(t_points) - np.min(t_points) + tbin
            orbit_rate.append(counts / orbit_time_interval)
            if(error_mode == 'counts'):
                orbit_rate_err.append(np.sqrt(counts) / orbit_time_interval)
            elif(error_mode == 'std'):
                orbit_rate_err.append(np.std(r_points))
            t_points = []
            r_points = []
            counts = 0
        else:
            t_points.append(lc.time[i + 1])
            counts += lc.rate[i + 1] * tbin

    orbit_lc = VariableBinLightCurve(t=orbit_time, te=orbit_time_err, r=orbit_rate, e=orbit_rate_err)
    return orbit_lc
