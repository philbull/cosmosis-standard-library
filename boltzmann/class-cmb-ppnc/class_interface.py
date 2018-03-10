from builtins import str
import os, sys, traceback
from cosmosis.datablock import names, option_section

# add class directory to the path
dirname = os.path.split(__file__)[0]
# enable debugging from the same directory
if not dirname.strip():
    dirname = '.'
install_dir = dirname + "../class/class_v2.4.1/classy_install/lib/python2.7/site-packages/"
sys.path.insert(0, install_dir)

import classy
import numpy as np

# These are pre-defined strings we use as datablock
# section names
cosmo = names.cosmological_parameters
distances = names.distances
cmb_cl = names.cmb_cl
ppn_block = 'ppn_parameters'
ppn_flrw_block = 'ppn_flrw_eff'

def setup(options):
    # Read options from the ini file which are fixed across
    # the length of the chain
    config = {
        'lmax': options.get_int(option_section, 'lmax', default=2000),
        'debug': options.get_bool(option_section, 'debug', default=False),
    }

    # Create the object that connects to Class
    config['cosmo'] = classy.Class()

    # Return all this config information
    return config


def get_class_inputs(block, config):
    
    # Background cosmology parameters
    h_ppn = block[cosmo, 'h0']
    ombh2_ppn = block[cosmo, 'ombh2']
    omch2_ppn = block[cosmo, 'omch2']
    om_ppn = (ombh2_ppn + omch2_ppn) / h_ppn**2.
    
    # PPNC parameters
    alpha_inf = block[ppn_block, 'alphainf_pn']
    gamma_inf = block[ppn_block, 'gammainf_pn']
    gamma_cm_inf = block[ppn_block, 'gamma_cminf_pn']
    gamma_cDE_inf = block[ppn_block, 'gamma_cDEinf_pn']
    
    # Convert PPNC cosmological parameters into parameters of an effective 
    # FLRW model (valid at high-z only)
    omh2 = h_ppn**2. * om_ppn \
         * (gamma_inf - 2.*gamma_cm_inf/(3.*om_ppn*(100.*h_ppn)**2.))
    odeh2 = -2. * gamma_cDE_inf / 100.**2. # Divide by (100 km/s/Mpc)^2
    h_eff = np.sqrt(omh2 + odeh2)
    om_eff = omh2 / h_eff**2.
    ode_eff = odeh2 / h_eff**2.
    
    # Calculate effective FLRW parameters to be input into CLASS
    f_baryon = ombh2_ppn / (ombh2_ppn + omch2_ppn)
    omch2_eff = (1. - f_baryon) * om_eff * h_eff**2.
    ombh2_eff = f_baryon * om_eff * h_eff**2.
    block[ppn_flrw_block, 'h_eff'] = h_eff
    block[ppn_flrw_block, 'omch2_eff'] = omch2_eff
    block[ppn_flrw_block, 'ombh2_eff'] = ombh2_eff
    
    # Get parameters from block and give them the
    # names and form that class expects
    params = {
        'output': 'tCl pCl mPk',
        'l_max_scalars': config["lmax"],
        'lensing': 'no',
        'A_s':       block[cosmo, 'A_s'],
        'n_s':       block[cosmo, 'n_s'],
        'H0':        100 * h_eff,
        'omega_b':   ombh2_eff,
        'omega_cdm': omch2_eff,
        'tau_reio':  block[cosmo, 'tau'],
        'T_cmb':     block.get_double(cosmo, 't_cmb', default=2.726),
        'N_eff':     block.get_double(cosmo, 'massless_nu', default=3.046),
    }
    return params


def get_class_outputs(block, c, config):
    """
    Calculate CMB angular power spectra.
    """
    # Sound horizon at drag epoch, and comoving distance to recombination
    block[distances, 'rs_zdrag'] = c.rs_drag()
    derived = c.get_current_derived_parameters(['z_rec', 'ra_rec'])
    block[ppn_flrw_block, 'ra_rec_eff'] = derived['ra_rec']
    block[ppn_flrw_block, 'sigma8_eff'] = c.sigma8()

    # CMB C_ell
    c_ell_data = c.raw_cl()
    ell = c_ell_data['ell']
    ell = ell[2:]
    block[cmb_cl, "ell"] = ell

    # t_cmb is in K, convert to mu_K, and add ell(ell+1) factor
    tcmb_muk = block[cosmo, 't_cmb'] * 1e6
    f = ell * (ell + 1.) / 2. / np.pi * tcmb_muk**2
    for s in ['tt', 'ee', 'te', 'bb']:
        block[cmb_cl, s] = c_ell_data[s][2:] * f


def execute(block, config):
    c = config['cosmo']

    try:
        # Set input parameters
        params = get_class_inputs(block, config)
        c.set(params)

        # Run calculations
        c.compute()

        # Extract outputs
        get_class_outputs(block, c, config)
    except classy.CosmoError as error:
        if config['debug']:
            sys.stderr.write("Error in class. You set debug=T so here is more debug info:\n")
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write("Error in class. Set debug=T for info: {}\n".format(error))
        return 1
    finally:
        # Reset for re-use next time
        c.struct_cleanup()
    return 0


def cleanup(config):
    config['cosmo'].empty()
