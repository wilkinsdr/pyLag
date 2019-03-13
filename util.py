from .lightcurve import *


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
