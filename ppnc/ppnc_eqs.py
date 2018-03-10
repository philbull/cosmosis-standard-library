#!/usr/bin/env python
import numpy as np
import pylab as plt
from scipy.integrate import odeint, cumtrapz
from scipy.interpolate import interp1d

c = 2.99792458e5 # Speed of light in km/s

"""
Code notations and conventions

We will use 'a' to represent the scale factor and 'z' to represent the redshift.
'k' will denote the spatial curvature. We choose the scale factor today, a_0 to 
be 1. H0 is the Hubble rate today. Omega_M0 is the total energy density of all 
matter today.

There are 4 PPNC parameters - alpha, alpha_c, gamma, gamma_c.
Since we have one constraint on the parameters due to energy-momentum 
conservation we only need to specify 3 of them out of which two are given by:

    alpha = alpha_0 a^(-p)
    gamma = gamma_0 a^(-q)

where alpha_0 and gamma_0 are the values of the parameters today and p and q 
are constants. The third parameter is gamma_c which has a contriution from both 
the matter sector and dark energy-like contribution and is represnted by:

    gamma_c = gamma_cm a^(-3) + gamma_cDE

where gamma_cm is the matter contribution and gamma_cDE is the dark energy 
contribution. Motivated by Brans-Dicke theory, we assume that gamma_cm scales 
in the same way as gamma so that
    
    gamma_cm = gamma_cm0 a^(-q) 

and the dark energy contribution scales as

    gamma_cDE = gamma_cDE0 a^(-r)

where r is a constant. The fourth PPNC parameter is alpha_c, which is fully 
specified once we have alpha, gamma, gamma_c, H0 and Omega_M0. For LCDM, 
gamma_cm = 0 and alpha = gamma = 1.

Hence we have a function that requires us to input the standard cosmological 
parameters H0, Omega and k. We also need to input the value of the PPNC 
parameters and their scaling relationship with the scale factor p, q, r.
"""

def lcdm_params(H0, OmegaM0):
    """
    Return dict containing values of the PPNC parameters corresponding to 
    flat LambdaCDM.
    """
    # Assume flatness
    OmegaL0 = 1. - OmegaM0
    k = 0.0 # Spatial curvature
    
    # Omega_lam0 = rho_lam / (3 H0^2 / 8 pi G) where rho_lam = (lam c^2)/(8 pi G)
    lam = 3. * OmegaL0 * (H0/c)**2.

    # Set parameters to LCDM values
    params = {
        'alpha_0':      1.0,
        'gamma_0':      1.0,
        'gamma_cm0':    0.0,
        'gamma_cDE0':   -lam/2.,
        'p':            0.0,
        'q':            0.0,
        'r':            0.0,
        'k':            k,
        'H0':           H0,
        'Omega_M0':     OmegaM0,
    }
    return params
    

class PPNCosmology(object):
    
    def __init__(self, H0, Omega_M0, k, alpha_0, p, gamma_0, gamma_cm0, 
                       q, gamma_cDE0, r, alpha_inf=1., gamma_inf=1., 
                       gamma_cminf=0., gamma_cDEinf=0., zmax=2.):
        """
        Cosmology defined in terms of PPN parameters.
        """
        # Set cosmological parameters
        self.H0 = H0
        self.Omega_M0 = Omega_M0
        self.k = k
        
        # Set PPNC low-z amplitude parameters
        self.alpha_0 = alpha_0
        self.gamma_0 = gamma_0
        self.gamma_cm0 = gamma_cm0
        self.gamma_cDE0 = gamma_cDE0
        
        # Set PPNC high-z parameters
        self.alpha_inf = alpha_inf
        self.gamma_inf = gamma_inf
        self.gamma_cminf = gamma_cminf
        self.gamma_cDEinf = gamma_cDEinf
        
        # Set PPNC redshift depence index parameters
        self.p = p
        self.q = q
        self.r = r
        
        # Derived parameters
        self.omh2 = self.H0**2. * self.Omega_M0
        
        # Define step function width and central redshift (smoothly 
        # interpolates between constant at high-z and z-dep. fn. at low-z)
        self.step_zc = 10.
        self.step_dz = 0.5
        
        # Define maximum redshift for growth spline
        self.zmax = zmax
        
        # Initialise distance and growth spline variables
        self.chi_spline = None
        self.f_spline, self.D_spline = None, None
    
    #---------------------------------------------------------------------------
    
    def stepfn(self, a, zc=None, dz=None):
        """
        Smooth transition from 0 at high z to 1 at low z, using a tanh 
        function. Centred on a redshift zc, with width dz.
        """
        if zc is None: zc = self.step_zc
        if dz is None: dz = self.step_dz
        z = 1./a - 1.
        f = 1. - 0.5 * np.tanh((z - zc) / dz)
        return f
    
    def alphaf(self, a):
        return self.alpha_0 * a**(-self.p) * self.stepfn(a) + self.alpha_inf
    
    def gammaf(self, a):
        return self.gamma_0 * a**(-self.q) * self.stepfn(a) + self.gamma_inf
    
    def gamma_cmf(self, a):
        """
        Matter contribution to gamma_c
        """
        return self.gamma_cm0 * a**(-self.q) * self.stepfn(a) + self.gamma_cminf
    
    def gamma_cDEf(self, a):
        """
        DE contribution to gamma_c
        """
        return self.gamma_cDE0 * a**(-self.r) * self.stepfn(a) + self.gamma_cDEinf
    
    def gamma_cf(self, a):
        return self.gamma_cmf(a) * a**(-3.) + self.gamma_cDEf(a)
    
    #---------------------------------------------------------------------------
    def omegam_eff(self):
        """
        Value of Omega_M,0 in effective FLRW model at high-z, where the PPNC 
        functions become constant.
        """
        pass
    
    def omegaDE_eff(self):
        """
        Value of Omega_DE,0 in effective FLRW model at high-z, where the PPNC 
        functions become constant.
        """
        pass
        
    #---------------------------------------------------------------------------
    
    def con_Hf(self, a):
        """
        The conformal Hubble rate, con_H, using Eq. (7.1) in notes.
        """
        y = self.omh2 * self.gammaf(a) / a \
          - (2./3.) * self.gamma_cf(a) * (a*c)**2. \
          - self.k*(c*c)
        return np.sqrt(y)
    
    def con_H_primef(self, a):
        """
        Conformal time derivative of the conformal Hubble rate, using 
        Eq. (7.12) in notes. For this we need the 4th PPNC parameter alpha_c.
        For the 4th PPNC parameter we need the time derivatives gamma and 
        gamma_c.
        """
        conH = self.con_Hf(a)
        gammaf = self.gammaf(a)
        alphaf = self.alphaf(a)
        
        # Conformal time derivative of gamma
        gamma_prime = -self.q * gammaf * conH
        
        # Conformal time derivative of gamma_prime
        gamma_c_prime = -(3. + self.q) * self.gamma_cmf(a) \
                      - self.r * self.gamma_cDEf(a)
        gamma_c_prime *= conH
        
        # alpha_c can be computed using the constraint equation, Eq. (7.11) 
        # in the notes
        alpha_c = (3./2.) * self.omh2 * a**-3.0 \
                * (alphaf - gammaf + gamma_prime / conH) \
                - 2. * self.gamma_cf(a) - gamma_c_prime / conH
        
        return -self.omh2 * alphaf / (2.*a) + (1./3.) * alpha_c * (a*c)**2.
    
    
    def precompute_distance(self):
        """
        Precompute distance splines.
        """
        if self.k != 0.:
            raise NotImplementedError("Non-zero curvature not currently "
                                      "supported.")
        
        # Grid in scale factor
        a = np.logspace(0., np.log10(1./(1.+self.zmax)), 500)
        
        # Integrate to get comoving distance
        integ = -1. / (a * self.con_Hf(a))
        r = c * cumtrapz(integ, a, initial=0.) # Mpc
        
        # Construct interpolation function
        r_spl = interp1d(np.log(a[::-1]), r[::-1], 
                         kind='linear', bounds_error=False)
        self.chi_spline = lambda a: r_spl(np.log(a))
        
    
    def dD_da(self, D, loga):
        """
        Define function to evaluate rate of change of growth factor D.
        """
        a = np.exp(loga)
        conH = self.con_Hf(a)
        
        # D[0] = D and D[1] = D_{, ln a}, return D_{,ln a} and D_{,ln a ln a}, 
        # using Eq. (7.10) in notes
        dd = (  D[1] * (-conH**2. - self.con_H_primef(a)) \
              + (3./2.) * D[0] * self.omh2 * self.alphaf(a)/a ) \
           / conH**2.
        return np.array([D[1], dd])
    
    def precompute_growth(self):
        """
        Precompute growth factor and growth rate splines.
        """
        # Set initial conditions for growth factor, D=1, D_{,a}=0.5.
        # FIXME: Choose this better!
        D0 = np.array([1.0, 0.51278103]) # Fixed using CCL growth rate today
        
        # Grid in (log) scale factor to calculate growth on
        loga = np.log( np.logspace(0., np.log10(1./(1.+self.zmax)), 500) )
      
        # Integrate to obtain growth factor solutions
        Ds = odeint(self.dD_da, D0, loga)
        Dz = Ds[:,0]
        
        # DS[0] = D, Ds[1] = D_{, ln a}
        # Growth rate, f, using Eq. (7.10) in notes
        fz = Ds[:,1] / Ds[:,0]
        
        # Construct splines
        D_spl = interp1d(loga[::-1], Dz[::-1], kind='linear', bounds_error=False)
        f_spl = interp1d(loga[::-1], fz[::-1], kind='linear', bounds_error=False)
        self.D_spline = lambda a: D_spl(np.log(a))
        self.f_spline = lambda a: f_spl(np.log(a))
    
    def expansion_rate(self, a):
        """
        Return the Hubble expansion rate, H(a), in km/s/Mpc.
        """
        return self.con_Hf(a) / a
    
    def comoving_distance(self, a):
        """
        Return comoving distance, chi(a).
        """
        if self.chi_spline is None: self.precompute_distance()
        return self.chi_spline(a)
    
    def growth_rate(self, a):
        """
        Return linear growth rate, f(a).
        """
        if self.f_spline is None: self.precompute_growth()
        return self.f_spline(a)
    
    def growth_factor(self, a):
        """
        Return linear growth factor, D(a).
        """
        if self.D_spline is None: self.precompute_growth()
        return self.D_spline(a)
    
    def growth_index(self, a):
        """
        Compute the growth index, gamma, such that f ~ [Omega_M(a)]^gamma.
        """
        # To compute the growth index, compute Omega_m, For this we need
        gamma = self.gamma_0 * a**(-self.q)
        
        H = self.con_Hf(a) / a
        Omega_M = self.Omega_M0 * a**(-3.0) * (self.H0 / H)**2.
        Omega_DE0 = 1.0 - self.Omega_M0
        Omega_DE = Omega_DE0 * (self.H0 / H)**2.
        f = self.growth_rate(a)
        
        # growth_index, gamma_growth = ln f / ln (gamma*Omega_M)
        gamma_growth = np.log(f) / np.log(gamma * Omega_M)
        return gamma_growth


if __name__ == '__main__':
    import pyccl as ccl
    
    # Set basic cosmological parameters
    H0 = 73.24 # km/s/Mpc
    Omega_lam0 = 0.7 # DE density today
    Omega_b = 0.05
    Omega_c = 1. - Omega_lam0 - Omega_b

    # Get PPN parameters for vanilla LCDM cosmology and initialise PPN cosmology
    ppnc_params = lcdm_params(H0=H0, OmegaM0=Omega_c+Omega_b)
    ppncosmo = PPNCosmology(**ppnc_params)
    print ppnc_params['gamma_cDE0']
    
    # Calculate PPNC growth rate and growth factor
    z = np.linspace(0., 1.99, 100)
    a = 1. / (1. + z)
    ppn_D = ppncosmo.growth_factor(a)
    ppn_f = ppncosmo.growth_rate(a)
    ppn_chi = ppncosmo.comoving_distance(a)
    
    # Initialise CCL cosmology
    cclcosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=H0/100., 
                             A_s=2.1e-9, n_s=0.96)
    
    # Calculate CCL growth rate and growth factor
    ccl_D = ccl.growth_factor(cclcosmo, a)
    ccl_f = ccl.growth_rate(cclcosmo, a)
    ccl_chi = ccl.comoving_radial_distance(cclcosmo, a)
    
    # Plot comparison
    plt.subplot(111)
    
    plt.plot(z, ccl_D, 'k-', lw=1.8)
    plt.plot(z, ppn_D, 'r--', lw=1.8)
    
    plt.plot(z, ccl_f, 'k-', lw=1.8)
    plt.plot(z, ppn_f, 'y--', lw=1.8)
    
    plt.plot(z, ccl_chi, 'k-', lw=1.8)
    plt.plot(z, ppn_chi, 'c--', lw=1.8)
    
    plt.xlabel("z")
    plt.tight_layout()
    plt.show()
