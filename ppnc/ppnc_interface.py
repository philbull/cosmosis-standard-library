from builtins import str
from cosmosis import constants
from cosmosis.datablock import names, option_section
import os, sys, traceback
import numpy as np
from scipy.interpolate import interp1d

# Import CCL and PPNC
import pyccl as ccl
from ppnc_eqs import PPNCosmology

# These are pre-defined strings we use as datablock section names
cosmo = names.cosmological_parameters
growth = names.growth_parameters
distances = names.distances
cmb_cl = names.cmb_cl
flrw_block = 'ppn_flrw_eff'


def setup(options):
    """
    Read options from the ini file that are fixed across the length of the chain.
    """
    config = {
        'kmax': options.get_double(option_section, 'kmax', default=0.1),
        'zref': options.get_double(option_section, 'zref', default=5.)
    }
    return config


def get_cosmology(block):
    """
    Get parameters from block and construct CCL Cosmology() object.
    """
    h = block[cosmo, 'h0']
    params = {
        'A_s':       block[cosmo, 'A_s'],
        'n_s':       block[cosmo, 'n_s'],
        'h':         h,
        'Omega_b':   block[cosmo, 'ombh2'] / h**2.,
        'Omega_c':   block[cosmo, 'omch2'] / h**2.,
        #'N_eff':     block.get_double(cosmo, 'massless_nu', default=3.046),
        #'mnu':       block.get_double(cosmo, 'mnu', default=0.0),
        # FIXME: Should use the same neutrino model
    }
    
    # Create new CCL Cosmology() object
    try:
        c = ccl.Cosmology(**params)
    except:
        sys.stderr.write("Error initialising CCL\n")
        traceback.print_exc(file=sys.stderr)
    return c


def get_flrw_cosmology(block):
    """
    Get parameters from block and construct CCL Cosmology() object for the 
    effective FLRW model.
    """
    h_eff = block[flrw_block, 'h_eff']
    params = {
        'h':         h_eff,
        'n_s':       block[cosmo, 'n_s'],
        'sigma8':    block[flrw_block, 'sigma8_eff'],
        'Omega_b':   block[flrw_block, 'ombh2_eff'] / h_eff**2.,
        'Omega_c':   block[flrw_block, 'omch2_eff'] / h_eff**2.,
        #'N_eff':     block.get_double(cosmo, 'massless_nu', default=3.046),
        #'mnu':       block.get_double(cosmo, 'mnu', default=0.0),
        # FIXME: Should use the same neutrino model
    }
    
    # Create new CCL Cosmology() object
    try:
        c = ccl.Cosmology(**params)
    except:
        sys.stderr.write("Error initialising CCL\n")
        traceback.print_exc(file=sys.stderr)
        c = None
    return c


def get_ppnc_params(block, ccl_cosmo):
    """
    Get PPNC parameters from block.
    """
    ppnc_params = {
        'alpha_0':       block['ppn_parameters', 'alpha0_pn'],
        'gamma_0':       block['ppn_parameters', 'gamma0_pn'],
        'gamma_cm0':     block['ppn_parameters', 'gamma_cm0_pn'],
        'gamma_cDE0':    block['ppn_parameters', 'gamma_cDE0_pn'],
        'p':             block['ppn_parameters', 'p_pn'],
        'q':             block['ppn_parameters', 'q_pn'],
        'r':             block['ppn_parameters', 'r_pn'],
        'alpha_inf':     block['ppn_parameters', 'alphainf_pn'],
        'gamma_inf':     block['ppn_parameters', 'gammainf_pn'],
        'gamma_cminf':   block['ppn_parameters', 'gamma_cminf_pn'],
        'gamma_cDEinf':  block['ppn_parameters', 'gamma_cDEinf_pn'],
    }
    ppnc_params['H0'] = ccl_cosmo['H0'] # FIXME
    ppnc_params['Omega_M0'] = ccl_cosmo['Omega_c'] + ccl_cosmo['Omega_b'] # FIXME
    ppnc_params['k'] = 0. # FIXME
    
    return ppnc_params


def rescale_cmb_spectra(block, chi_rec_eff, chi_rec_out):
    """
    Rescale CMB angular power spectra to account for the change in the distance 
    to last scattering between the input effective FLRW model and the output 
    PPNC model.
    """
    # Get CMB C_ell's
    ell = block[cmb_cl, 'ell']
    #f = ell * (ell + 1.) / (2. * np.pi)
    Dl_tt = block[cmb_cl, "tt"]
    Dl_te = block[cmb_cl, "te"]
    Dl_ee = block[cmb_cl, "ee"]
    
    # Interpolate and shift CMB power spectrum (D_ell)
    # NB. This doesn''t handle the low-ell CMB very well
    shift = chi_rec_out / chi_rec_eff
    interp_tt = interp1d(ell * shift, Dl_tt, kind='linear', bounds_error=False)
    interp_te = interp1d(ell * shift, Dl_te, kind='linear', bounds_error=False)
    interp_ee = interp1d(ell * shift, Dl_ee, kind='linear', bounds_error=False)
    block[cmb_cl, "tt"] = interp_tt(ell)
    block[cmb_cl, "te"] = interp_te(ell)
    block[cmb_cl, "ee"] = interp_ee(ell)
    

def get_ppnc_outputs(block, ccl_cosmo, ccl_flrw, ppnc_cosmo, config):
    """
    Distances, growth rate, and power spectrum returned from CCL, and modified 
    by PPNC equations.
    """
    h0 = block[cosmo, 'h0']
    
    # Set z arrays 
    zmin, zmax, nz = 0., config['zref']+0.1, 500
    z = np.linspace(zmin, zmax, nz)
    a = 1. / (1. + z)
    
    # (1) Calclate distances in PPN Cosmology
    chi = ppnc_cosmo.comoving_distance(a)
    block[distances, 'z'] = z
    block[distances, 'nz'] = nz
    block[distances, 'd_l'] = chi / a
    block[distances, 'd_a'] = chi * a
    block[distances, 'd_m'] = chi
    
    # Background quantities needed for BAO (H in units of Mpc^-1)
    block[distances, 'h'] = ppnc_cosmo.expansion_rate(a) / (constants.c / 1e3)
    
    # (2) Growth factor and growth rate
    # Get PPN growth
    Dz_ppn = ppnc_cosmo.growth_factor(a)
    fz_ppn = ppnc_cosmo.growth_rate(a)
    
    # Pack PPN growth into output structure
    block[growth, 'z'] = z
    block[growth, 'd_z'] = Dz_ppn
    block[growth, 'f_z'] = fz_ppn
    
    """
    # (3) Matter power spectrum, if requested
    # Get CCL growth
    Dz_ccl = ccl.growth_factor(ccl_cosmo, a) # normed to D=1 at a=1
    fz_ccl = ccl.growth_rate(ccl_cosmo, a)
    
    # Calculate P(k, z) using CCL
    kmin, kmax, nk = 1e-4, config['kmax'], 200
    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    pk = np.array( [ccl.linear_matter_power(ccl_cosmo, k, _a) for _a in a] )
    
    # Rescale by (PPNC growth factor / CCL growth factor)^2
    pk *= np.atleast_2d((Dz_ppn / Dz_ccl)**2.).T
    
    # Store as grid
    block.put_grid("matter_power_lin", 
                   "k_h", k / h0, 
                   "z", z, 
                   "p_k", pk * h0**3.)
    """
    
        #h_eff:  "real, value of h = H0/(100 km/s/Mpc) in effective FLRW model"
        #omch2_eff:  "real, value of Omega_M h^2 in effective FLRW model"
        #ombh2_eff:  "real, value of Omega_b h^2 in effective FLRW model"
        #ra_rec_eff: "real, comoving dist. to recombination in eff. FLRW, in Mpc"
        #sigma8_eff: "real, sigma8 at z=0 in effective FLRW model"
    
    # (4) Rescale CMB power spectra by applying shift in distance to recombination
    # Reference redshift where effective FLRW takes over
    a_ref = 1. / (1. + config['zref'])
    
    # Dist. to last scattering in effective FLRW cosmology (from CLASS)
    chi_rec_eff = block[flrw_block, 'ra_rec_eff']
    
    # Comoving distance to reference redshift in effective FLRW cosmology
    chi_zref_eff = ccl.comoving_radial_distance(ccl_flrw, a_ref)
    
    # Comoving distance to reference redshift in PPN Cosmology
    chi_zref_ppn = ppnc_cosmo.comoving_distance(a_ref)
    
    # Comoving distance to LSS in PPN Cosmology
    chi_rec_ppn = chi_rec_eff - chi_zref_eff + chi_zref_ppn
    
    # Rescale CMB power spectra (applies shift parameter)
    rescale_cmb_spectra(block, chi_rec_eff, chi_rec_ppn)
    
    # (5) Rescale sigma_8 at z=0 in PPN Cosmology
    sigma8_eff = block[flrw_block, 'sigma8_eff']
    Dzref_eff = ccl.growth_factor(ccl_flrw, a_ref) # assumes D=1 at a=1
    Dzref_ppn = ppnc_cosmo.growth_factor(a_ref) # assumes D=1 at a=1
    block[cosmo, 'sigma_8'] = sigma8_eff * (Dzref_eff / Dz_ppn)


def execute(block, config):
    """
    Run CCL, with output rescaled by PPNC
    """
    try:
        # Set input parameters
        ccl_cosmo = get_cosmology(block)
        ccl_flrw = get_flrw_cosmology(block)
        ppnc_params = get_ppnc_params(block, ccl_cosmo)
        ppnc_cosmo = PPNCosmology(**ppnc_params)

        # Calculate cosmological functions
        get_ppnc_outputs(block, ccl_cosmo, ccl_flrw, ppnc_cosmo, config)
        
    except:
        sys.stderr.write("Error running CCL with PPNC modifications.\n")
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        pass
    return 0


def cleanup(config):
    pass
    
