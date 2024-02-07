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


def save_obj(obj, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(filename):
    import pickle
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


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


def sum_mos_pn_energy_lightcurves(pn_lcpath, mos_lcpath, outpath):
    from .lightcurve import LightCurve, get_lclist, extract_sim_lightcurves
    import re
    import glob
    import os

    # define a regex to extract useful information from the filenames
    lc_re = re.compile(r'([0-9]+)_.*?src_(mos|pn)_tbin([0-9]+)_en([0-9\-]+).lc_corr')
    lc_obsid = lambda f: lc_re.match(f).group(1)
    lc_inst = lambda f: lc_re.match(f).group(2)
    lc_tbin = lambda f: lc_re.match(f).group(3)
    lc_enband = lambda f: lc_re.match(f).group(4)

    # find the unique OBSIDs, time binnings and energy bands
    # since we want to create a new light curve for each of these
    pn_lcfiles = sorted(glob.glob(pn_lcpath + '/*.lc_corr'))
    obsids = set([lc_obsid(os.path.basename(f)) for f in pn_lcfiles])
    tbins = set([lc_tbin(os.path.basename(f)) for f in pn_lcfiles])
    enbands = set([lc_enband(os.path.basename(f)) for f in pn_lcfiles])

    for obsid in obsids:
        for tbin in tbins:
            for enband in enbands:
                lcl_mos1 = get_lclist(mos_lcpath + '/%s_EMOS1_*src_mos_tbin%s_en%s.lc_corr' % (obsid, tbin, enband))
                lcl_mos2 = get_lclist(mos_lcpath + '/%s_EMOS2_*src_mos_tbin%s_en%s.lc_corr' % (obsid, tbin, enband))
                lcl_pn = get_lclist(pn_lcpath + '/%s_*src_pn_tbin%s_en%s.lc_corr' % (obsid, tbin, enband))

                lcl_mos1 = [l[1:-1] for l in lcl_mos1 if l.rate.max() > 0.1]
                lcl_mos2 = [l[1:-1] for l in lcl_mos2 if l.rate.max() > 0.1]

                mos1_lc = LightCurve().concatenate(lcl_mos1).remove_gaps().fill_time(dt=float(tbin)).interp_gaps()
                mos2_lc = LightCurve().concatenate(lcl_mos2).remove_gaps().fill_time(dt=float(tbin)).interp_gaps()
                pn_lc = lcl_pn[0].interp_gaps()

                mos1_lc, mos2_lc = extract_sim_lightcurves(mos1_lc, mos2_lc)
                mos_sum_lc = mos1_lc + mos2_lc

                mos_sum_lc, pn_lc = extract_sim_lightcurves(mos_sum_lc, pn_lc)
                sum_lc = mos_sum_lc + pn_lc
                sum_lc.write_fits(outpath + '/%s_src_mospn_tbin%s_en%s.lc' % (obsid, tbin, enband))

