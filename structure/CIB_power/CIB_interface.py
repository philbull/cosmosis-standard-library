'''

Compute CIB cl, using Hall model, the single-spectra-energy-distribution (SSED) model of the  from Hall et. al. 2010 (arxiv:0912.4315)



'''


from cosmosis.datablock import names, option_section
import numpy as np
import hall as cib_hall
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, interp1d

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.

cosmo = names.cosmological_parameters
distances = names.distances


def powerspec(k, z, rbs):
    return rbs.ev(k, z)


def clint(z, l, hspline, chispline, wa, wb, rbs):

    k = l / chispline(z)
    tmp = wa(z) * wb(z) * powerspec(k, z, rbs) / chispline(z) ** 2. / hspline(z)
    return tmp


def setup(options):

    # L BINS
    llmin = options.get_double(option_section, "llmin", default=1.)
    llmax = options.get_double(option_section, "llmax", default=3.5)
    dlnl = options.get_double(option_section, "dlnl", default=.1)
    zmin = options.get_double(option_section, "zmin", default=1e-2)
    zmax = options.get_double(option_section, "zmax", default=10.)
    # Frequency of the CIB
    nu = options.get_double(option_section, "nu", default=150.e9)
    # What matter power spectrum to use, linear Halofit etc
    blockname = options.get_string(option_section, "matter_power", default="matter_power_lin")

    print 'llmin = ', llmin
    print 'llmax = ', llmax
    print 'dlnl = ', dlnl
    print 'nu = ', nu
    print 'matter_power = ', blockname
    print ' '

 #  the cib has a lot of parameter for the spectra ssed etc we can get them optionally here

    # maybe the suffix for saving the spectra
    # zmax (or take it from CAMB)
    # maybe choose between kappa and others
    lbins = np.arange(llmin, llmax, dlnl)
    lbins = 10. ** lbins
    return (lbins, blockname, nu, zmin, zmax)


def execute(block, config):
    # Just a simple rename for clarity.
    lbins, blockname, nu, zmin, zmax = config

    # LOAD POWER

    zpower = block[blockname, "z"]
    kpower = block[blockname, "k_h"]
    powerarray = block[blockname, "p_k"].reshape([np.size(zpower), np.size(kpower)]).T
    rbs = RectBivariateSpline(kpower, zpower, powerarray)

    # Cosmological parameters
    # =======================
    omega_m = block[cosmo, "omega_m"]
    h0 = block[cosmo, "h0"]
    # =======================

    # Distances
    # =======================
    # reverse them so they are going in ascending order
    h = block[distances, "h"]
    tmp = h[::-1]
    h = tmp

    zdist = block[distances, "z"]
    tmp = zdist[::-1]
    zdist = tmp

    d_m = block[distances, "d_m"]
    tmp = d_m[::-1]
    d_m = tmp
    # =======================

    lbins = np.arange(1000, 6000, 500)
    # These have dimensions of Mpc; change to h^{-1} Mpc
    d_m *= h0
    h /= h0
    # now in units of h^{-1} Mpc or the inverse
    chispline = interp1d(zdist, d_m)
    hspline = interp1d(zdist, h)
    # =======================
    # DEFINE KERNEL
    lkern = cib_hall.ssed_kern(h0, zdist, chispline, nu, jbar_kwargs={'zc': 2.0})

    cl = np.zeros(np.size(lbins))

    # COMPUTE CLs
    # =======================

    for i, l in enumerate(lbins):
        cl[i] = scipy.integrate.quad(
            clint, zmin, zmax, limit=100,  args=(l, hspline, chispline, lkern.w_interp, lkern.w_interp, rbs))[0]
    # =======================
    # SAVE IN DATABLOCK

    section = "limber_spectra"
    block[section, "cl_cib"] = cl
    block[section, "l_cib"] = lbins

    return 0