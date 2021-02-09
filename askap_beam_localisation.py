#!/opt/conda/bin/python
#!/usr/bin/python
#
#   ASKAP Multi Beam Localization Recipe............
#   What it does:
#        -
#
# Future Additiom
# Posterior plot of flux
#	- need correct width of the pulse
#	- fit the pulse with gaussian
#
#

################################################################################
"""
    IMPORT things here.....
"""

import matplotlib
matplotlib.use('Agg')
from astropy.coordinates import SkyCoord
import os, sys, json, argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy import log, exp, sqrt
from collections import OrderedDict
import corner
import fitsio
import pymultinest
from scipy.special import erfinv
from scipy.constants import c
import pandas as pd


def write_fits(postdata, bins, outfile, raref, decref, weights=None):
    """
    write posterior data as fits file
    """
    xmin = postdata[:,0].min() - 0.1*(postdata[:,0].max() - postdata[:,0].min())
    xmax = postdata[:,0].max() + 0.1*(postdata[:,0].max() - postdata[:,0].min())

    ymin = postdata[:,1].min() - 0.1*(postdata[:,1].max() - postdata[:,1].min())
    ymax = postdata[:,1].max() + 0.1*(postdata[:,1].max() - postdata[:,1].min())

    H, X, Y = np.histogram2d(postdata[:,0].flatten(), postdata[:,1].flatten(), \
                             bins=bins, range=[[xmin, xmax], [ymin, ymax]], \
                             weights=weights)
    H = H.astype(np.float32)

    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    filename = "%s_post.fits" % (outfile)

    header   = OrderedDict()
    header["CTYPE1"]  = "RA---TAN"
    header["CTYPE2"]  = "DEC--TAN"
    header["CUNIT1"]  = "deg"
    header["CUNIT2"]  = "deg"
    header["CRVAL1"]  = raref
    header["CDELT1"]  = dx
    header["CRVAL2"]  = decref
    header["CDELT2"]  = dy
    header["WCSAXES"] = 2
    header["CRPIX1"]  = (-X[0])/dx
    header["CRPIX2"]  = (-Y[0])/dy
    header["EQUINOX"] = 2000

    fitsio.write(filename, H, header=header)

def beam_fwhm(freq):
    """
    ASKAP fwhm calculation (McConnell et al. 2016)
    freq : (in GHz)
    """
    c_light    = 299792458.0
    diameter   = 12.0
    wavelength = c_light / (freq*1e9)
    fwhm       = 1.1*wavelength / diameter

    return fwhm*180./np.pi

def beam_select(beam_data, radius=0.45):
    nbeam     = len(beam_data)
    flux_data = beam_data.flux
    highbeam  = beam_data.flux.idxmax()
    beam_name = beam_data.beam
    beamx     = beam_data.xpos_new
    beamy     = beam_data.ypos_new

    fig      = plt.figure(figsize=(8, 8))
    main_ax  = plt.axes()

    mask   = np.ones(len(beamx), dtype=np.bool)
    mask[flux_data > 10.0] = 0
    #radinside = 2.9*radius
    radinside = 4.2*radius
    mask[((beamx - beamx[highbeam])**2. + (beamy - beamy[highbeam])**2.) < radinside**2.] = 0

    if np.sum(~mask) < 4:
        radinside = 4.2*radius
        mask[((beamx - beamx[highbeam])**2. + (beamy - beamy[highbeam])**2.) < radinside**2.] = 0

    for ibeam in range(nbeam):
        circ = plt.Circle((beamx[ibeam], beamy[ibeam]), radius=radius, fill=False, \
                           linestyle="--", ec="k")
        if mask[ibeam] == 0:
            circ = plt.Circle((beamx[ibeam], beamy[ibeam]), radius=radius, fill=True, \
                           linestyle="--")
        if ibeam==highbeam:
            circ = plt.Circle((beamx[ibeam], beamy[ibeam]), radius=radius, fill=True, \
                           linestyle="--", ec="r", fc="r", alpha=0.8)

        main_ax.annotate(beam_name[ibeam], xy=(beamx[ibeam], beamy[ibeam]), \
                           verticalalignment='center', horizontalalignment='center')
        main_ax.add_patch(circ)


    main_ax.set_xlim(np.floor(beamx.min())-1, np.ceil(beamx.max())+1)
    main_ax.set_ylim(np.floor(beamy.min())-1, np.ceil(beamy.max())+1)
    plt.xlabel(r'Offset (deg.)', size="large")
    plt.ylabel(r'Offset (deg.)', size="large")
    plt.title(r'beam radius = %.2f (deg.)' %(radius), size="large")

    return fig, mask

def plot_beam_pos(postdata, frb_data, radius, bins=50, color="blue"):
    xposmin = -2; xposmax =  2;
    yposmin = -2; yposmax =  2;
    beamx     = frb_data.xpos_new
    beamy     = frb_data.ypos_new
    beam_name = frb_data.beam

    fig     = plt.figure(figsize=(8, 8))
    main_ax = plt.axes((0.1, 0.1, 0.6, 0.6))                 # [left, bottom, width, height]

    corner.hist2d(postdata[:,0], postdata[:,1], bins=bins, color=color, plot_datapoints=False, \
                  smooth=True, plot_density=True, plot_contours=False, ax=main_ax)

    for ibeam in range(len(beamx)):
        circ = plt.Circle((beamx[ibeam], beamy[ibeam]), radius=radius[ibeam], fill=False, \
                           linestyle="--", ec="k")
        main_ax.annotate("beam " + beam_name[ibeam], xy=(beamx[ibeam], beamy[ibeam]), \
                          verticalalignment='center', horizontalalignment='center')
        main_ax.add_patch(circ)

    main_ax.set_xlim(xposmin, xposmax)
    main_ax.set_ylim(yposmin, yposmax)
    plt.xlabel(r'$\Delta \alpha \, Cos (\delta)$ (deg)', size="large")
    plt.ylabel(r'$\Delta \delta $ (deg)', size="large")

    postx_ax = plt.axes((0.1, 0.7, 0.6, 0.2), sharex=main_ax)
    postx_ax.hist(postdata[:,0], bins=bins, normed=True, edgecolor='blue', lw=0.5)
    postx_ax.set_xlim(xposmin, xposmax)
    plt.ylabel(r'$\rho $ (arb.)', size="large")
    postx_ax.get_yaxis().set_ticks([])
    plt.setp(postx_ax.xaxis.get_ticklabels(), visible=False)

    posty_ax = plt.axes((0.7, 0.1, 0.2, 0.6), sharey=main_ax)
    posty_ax.hist(postdata[:,1], bins=bins, orientation="horizontal", normed=True, \
                  edgecolor='blue', lw=0.5)
    posty_ax.set_ylim(yposmin, yposmax)
    plt.xlabel(r'$\rho $ (arb.)', size="large")
    posty_ax.get_xaxis().set_ticks([])
    plt.setp(posty_ax.yaxis.get_ticklabels(), visible=False)

    return fig

################################################################################

def get_bestprof_data(bestprof_files):
    col_name  = ['beam', 'xpos', 'ypos', 'flux', 'ferr', 'freq', 'sefd']
    dtype     = ['S2', 'f8', 'f8', 'f4', 'f4', 'f4', 'f4']
    beam_data = pd.DataFrame([], columns=col_name, dtype=dtype)

    for i, bestprof in enumerate(bestprof_files):
        # File name should be in the format obsid_ra_dec*.bestprof
        obsid, raj, decj = bestprof.split("_")[:3]
        
        # Get bestprof data from file
        with open(file_loc,"r") as bestprof:
            lines = bestprof.readlines()



def localize_frb(multibeam_file, outroot, save=True, verbose=False):

    xoff  = beam_data.loc[beam_data['flux'].idxmax()].xpos
    yoff  = beam_data.loc[beam_data['flux'].idxmax()].ypos

    # The offset, (x,y), is related to a true sky coordinate, (RA,DEC), by
    #     x = (RA - RA0) * cos(DEC0)
    #     y = DEC - DEC0
    # correct beam positions to relative positions accounting for cos(dec)
    beam_data['xpos_new'] = (beam_data.xpos - xoff)*np.cos(yoff*np.pi/180.)
    beam_data['ypos_new'] = (beam_data.ypos - yoff)

    # select beams surrounding highest detection SNR
    beam_mapfig, mask = beam_select(beam_data, radius=0.45)

    frb_data = beam_data[~mask]
    frb_data.reset_index(drop=True, inplace=True)
    nbeam    = len(frb_data)

    xpos  = frb_data.xpos_new
    ypos  = frb_data.ypos_new

    beam_name = frb_data.beam
    flux_data = frb_data.flux
    ferr_data = frb_data.ferr
    freq_beam = frb_data.freq

    # normalize sensitivity
    sens  = 2000./frb_data.sefd

    fwhm    = beam_fwhm(freq_beam)
    rad     = fwhm/2.
    rad1400 = rad*(freq_beam/1.4)

    #########################################################################################
    def beam_gain(x, y, fwhm):
        w   = fwhm / (2.0*sqrt(2*log(2.)))
        arg = x*x +y*y
        val = exp(-arg/2./w/w)
        return val

    def prior(cube, ndim, nparams):
        cube[0] = -2 + 4*cube[0]                   # uniform prior between -2:2
        cube[1] = -2 + 4*cube[1]                   # uniform prior between -2:2
        cube[2] = 10**(4*cube[2] - 1)              # log-uniform prior between 10^-1 and 10^3
        #nbeam = (ndim-3) / 4
        for ibeam in range(nbeam):
            mean = 1.0 ; sigma = 0.1               # Gaussian prior: mean=1, sigma=0.1
            cube[3+4*ibeam+0] = mean + (2**0.5)*sigma*erfinv(2*cube[3+4*ibeam+0] - 1)
            cube[3+4*ibeam+1] = mean + (2**0.5)*sigma*erfinv(2*cube[3+4*ibeam+1] - 1)

            mean = 0.0 ; sigma = 1./60.            # Gaussian prior: mean=0, sigma=1.0 arcminute
            cube[3+4*ibeam+2] = (2**0.5)*sigma*erfinv(2*cube[3+4*ibeam+2]-1)
            cube[3+4*ibeam+3] = (2**0.5)*sigma*erfinv(2*cube[3+4*ibeam+3]-1)

    def loglike(cube, ndim, nparams):
        loglike = 0

        x, y, flux = cube[0], cube[1], cube[2]
        #nbeam = (ndim-3) / 4
        for ibeam in range(nbeam):
            gain_err  = cube[3+4*ibeam+0]
            width_err = cube[3+4*ibeam+1]
            xpos_err  = cube[3+4*ibeam+2]
            ypos_err  = cube[3+4*ibeam+3]

            gain  = beam_gain(x-(xpos[ibeam]+xpos_err), y-(ypos[ibeam]+ypos_err), \
                              width_err*fwhm[ibeam])
            gain *= sens[ibeam]

            arg      = gain*gain_err*flux - flux_data[ibeam]
            loglike += -arg*arg/(2.0*ferr_data[ibeam]**2.0)

            cube[3+4*nbeam+ibeam] = gain

        return loglike

    ############################################################################

    # number of dimensions our problem has
    parameters = ["x_pos", "y_pos", "SNR"]
    for ibeam in range(nbeam):
        parameters.append("gain_err_" + beam_name[ibeam])
        parameters.append("width_err_" + beam_name[ibeam])
        parameters.append("xpos_err_" + beam_name[ibeam])
        parameters.append("ypos_err_" + beam_name[ibeam])

    n_dims  = len(parameters)           # dimensionality (no. of free parameters)
    for ibeam in range(nbeam):
        parameters.append("gain_" + beam_name[ibeam])

    n_params = len(parameters)           # total parameters (free + derived)
    rootbase = "chains/%s-" % (outroot)  # root for output files
    chaindir = os.path.join(os.path.dirname(multibeam_file), "chains")
    root     = os.path.join(os.path.dirname(multibeam_file), rootbase)

    os.system("mkdir -p -v " + chaindir)  # create temprory sub-dir

    # Tunable MultiNest Parameters
    mmodal   = False                     # do mode separation
    nlive    = 1000                      # number of live points
    tol      = 0.1                       # defines the stopping criteria (0.5, good enough)
    efr      = 1.0                       # sampling efficiency. 0.8 and 0.3 are recommended
    updInt   = 1000                      # after # iterations feedback & the posterior files update
    resume   = False                     # resume from a previous job
    maxiter  = 0                         # max no. of iteration. 0 is unlimited
    initMPI  = False                     # initialize MPI routines?, False if main program handles init

    # run MultiNest
    pymultinest.run(loglike, prior, n_dims, n_params=n_params, multimodal=mmodal, n_live_points=nlive, \
                 evidence_tolerance=tol, sampling_efficiency=efr, n_iter_before_update=updInt, \
                 outputfiles_basename=root, verbose = verbose, resume=resume, max_iter=maxiter, \
                 init_MPI=initMPI)

    # other default inputs
    # n_clustering_params=None, wrapped_params=None, importance_nested_sampling=True,
    # const_efficiency_mode=False,  null_log_evidence=-1e+90, max_modes=100, seed=-1,
    # mode_tolerance=-1e+90, context=0, write_output=True, log_zero=-1e+100, dump_callback=None

    #########################################################################################

    a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = root)
    s = a.get_stats()

    # store name of parameters, always useful
    with open('%sparams.json' % root, 'w') as f:
	    json.dump(parameters, f, indent=2)
    # store derived stats
    with open('%sstats.json' % root, mode='w') as f:
	    json.dump(s, f, indent=2)

    print("  marginal likelihood:")
    print("    ln Z = %.1f +- %.1f'" % (s['global evidence'], s['global evidence error']))
    print("  parameters:")
    for p, m in zip(parameters, s['marginals']):
	    lo, hi = m['1sigma']
	    med    = m['median']
	    sigma  = (hi - lo) / 2
	    if sigma == 0:
	    	i = 3
	    else:
	    	i = max(0, int(-np.floor(np.log10(sigma))) + 1)
	    fmt  = '%%.%df' % i
	    fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
	    print(fmts % (p, med, sigma))

    #########################################################################################
    print("creating marginal plot ...")
    bins    = 50

    postdata = a.get_equal_weighted_posterior()[:,:3]         # only x_pos, y_pos, and SNR

    cornerdata = postdata.copy()
    cornerdata[:,2] = np.log10(cornerdata[:,2])
    cornerfig     = corner.corner(cornerdata, labels=["x_pos", "y_pos", "log SNR"], show_titles=True, \
                                  bins=50, smooth=True, smooth1d=True)

    beam_posfig = plot_beam_pos(postdata, frb_data, rad1400, bins, color="blue")


    outfile = os.path.join(os.path.dirname(multibeam_file), outroot)
    if save:
        beam_mapfig.savefig(outfile+'_beammap.pdf', bbox_inches='tight', papertype='letter', \
                                           orientation='landscape')
        cornerfig.savefig(outfile+'_cornerplot.pdf', bbox_inches='tight', papertype='letter', \
                                           orientation='landscape')
        beam_posfig.savefig(outfile+'_beampos.pdf', bbox_inches='tight', papertype='letter', \
                                           orientation='landscape')
        plt.close("all")

    else:
        plt.show()

    # write posterior image in fits file
    write_fits(postdata, bins, outfile, xoff, yoff, weights=None)

    return frb_data, postdata, rad1400


def askap_fluence(tsamp, bandwidth=336.0, antenna=36):
    """
    ASKAP Flux scale factor calculation (Based on radiometer equation/SEFD)
    """
    SEFD      = 2000.0 #Jy
    NPOL      = 2
    bandwidth = 336.0 # MHz
    fluence   = SEFD/np.sqrt(NPOL*bandwidth*1e6*tsamp) \
                        * tsamp * (1/np.sqrt(antenna))
    return fluence * 1000

if __name__ == "__main__":
    description   = " FRB Localization Recipe."
    parser        = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--beam', dest='multibeam_file', type=str, metavar='',
                              help='Path of the beam position SNR file')
    parser.add_argument('-b', '--bestprof', dest='bestprof_files', type=str, metavar='', nargs='*',
                              help='Path of the beam position SNR file')
    parser.add_argument('-f', '--source', dest='sourcename', type=str, metavar='',
                              help='Source Name', default="None")
    parser.add_argument('-s', '--save', dest='save', action='store_true',
                         help='Save plots')

    args = parser.parse_args()

    if (not args.multibeam_file):
        print('Input files are required!')
        print(parser.print_help())
        sys.exit(1)

    outroot     = args.sourcename                                    # output of frb_detector.py

    if args.bestprof_files and args.multibeam_file:
        print('Please you either --beam or --bestprof, not both.')
        print(parser.print_help())
        sys.exit(1)
    if args.multibeam_file:
        # Load multi_beam_global file
        col_name  = ['beam', 'xpos', 'ypos', 'flux', 'ferr', 'freq', 'sefd']
        dtype     = {'beam': 'S2', 'xpos': 'f8', 'ypos': 'f8', 'flux': 'f4',
                    'ferr': 'f4', 'freq': 'f4', 'sefd': 'f4'}
        beam_data = pd.read_csv(args.multibeam_file, delim_whitespace=True, comment='#',
                                names=col_name, dtype=dtype)
    if args.bestprof_files:
        beam_data = get_bestprof_data(args.bestprof_files)
    
    frb_data, postdata, rad1400 = localize_frb(beam_data, outroot, save=args.save, verbose=True)

    # put these information for report. Initialize info dict
    localizeinfo = {}

    xoff  = frb_data.loc[frb_data['flux'].idxmax()].xpos
    yoff  = frb_data.loc[frb_data['flux'].idxmax()].ypos


    # Calculate confidence limits
    xpos_mn, ypos_mn, snr_mn = list(map(list, [(v[2], \
                                v[3]-v[2], v[2]-v[1], v[4]-v[2], v[2]-v[0]) for v in zip(*np.percentile(postdata, \
                                [2.5, 16, 50, 84, 97.5], axis=0))]))

    # The offset, (x,y), is related to a true sky coordinate, (RA,DEC), by
    #     x = (RA - RA0) * cos(DEC0)
    #     y = DEC - DEC0
    # Obtain true sky coordinate
    xpos_mn[0]  = xpos_mn[0]/np.cos(yoff*np.pi/180.) + xoff
    ypos_mn[0] += yoff

    # with error propagation (only in RA)
    xpos_mn[1:] = xpos_mn[1:]/np.cos(yoff*np.pi/180.)

    # convert coordinates from deg to hh:mm:ss
    c = SkyCoord(xpos_mn, ypos_mn, frame='icrs', unit='deg')

    ra_arcmin  = c.ra.hms.s/60 + c.ra.hms.m + 60*c.ra.hms.h
    xpos_mn[0] = "%.2d:%05.2f" % (c.ra.hms.h[0], c.ra.hms.s[0]/60 + c.ra.hms.m[0])
    xpos_mn[1] = "%.2f" % ra_arcmin[1]
    xpos_mn[2] = "%.2f" % ra_arcmin[2]
    xpos_mn[3] = "%.2f" % ra_arcmin[3]
    xpos_mn[4] = "%.2f" % ra_arcmin[4]

    dec_arcmin = c.dec.signed_dms.s/60 + c.dec.signed_dms.m + 60*c.dec.signed_dms.d
    ypos_mn[0] = "%.2d:%.2f" % (c.dec.dms.d[0], c.dec.signed_dms.s[0]/60 + c.dec.signed_dms.m[0])
    ypos_mn[1] = "%.2f" % dec_arcmin[1]
    ypos_mn[2] = "%.2f" % dec_arcmin[2]
    ypos_mn[3] = "%.2f" % dec_arcmin[3]
    ypos_mn[4] = "%.2f" % dec_arcmin[4]

    # Write information for report file
    localizeinfo["xpos"] = xpos_mn
    localizeinfo["ypos"] = ypos_mn

    localizeinfo["frb_ra"] = c.ra.to_string(unit='hour', precision=1, pad=2)[0]
    localizeinfo["frb_dec"] = c.dec.to_string(unit='degree', precision=1, pad=2)[0]

    print("Reporting values....")
    for key, value in localizeinfo.items():
        print(key, value)



#############################################################################################