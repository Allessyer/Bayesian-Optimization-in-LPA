import numpy as np
import scipy.constants as ct
from wake_t import GaussianPulse, PlasmaStage, ParticleBunch
import aptools.plasma_accel.general_equations as ge
from aptools.utilities.bunch_generation import generate_gaussian_bunch_from_twiss

import sys
import argparse

#----- assel functions -----

def createParser ():
    parser = argparse.ArgumentParser()

    # Base laser parameters.
    parser.add_argument ('-e', '--energy_laser',type=float, default=1) # J
    parser.add_argument ('-tau', '--tau_laser', type=float, default=30e-15) # s (fwhm)
    parser.add_argument ('-lmd', '--lambda0', type=float, default=0.8e-6) # m
    parser.add_argument ('-w0', '--w0_laser', type=float, default=20e-6) # m

    # Get optimization parameters.
    parser.add_argument ('-task','--task',default='wake-t') # {{task}}   
    parser.add_argument ('-bq','--beam_q',type=float,default=1.e-12) # Qulon    
    parser.add_argument ('-bzh','--beam_z_head',type=float,default=0.)
    parser.add_argument ('-blgth','--beam_length',type=float,default=3.)
    parser.add_argument ('-npsrt','--n_p_start',type=float,default=1e24 * 2.)
    # n_p_end is initialized in main(), because it has dependency with n_p_start  
    parser.add_argument ('-lplt','--l_plateau',type=float,default=3. * 1.e-3)

    # Beam parameters.
    # They are initialized in main(), because they have dependency with beam_z_head and beam_length.
    
    return parser

#---------------------------




def density_profile(z):
    # Allocate relative density
    n = np.ones_like(z)
    # Make zero before plateau
    n = np.where(z < 0, 0, n)
    # Add taper
    n = np.where(
            (z >= 0) & (z <=l_plateau),
            1 + (n_p_end - n_p_start) / l_plateau /n_p_start * z,
            n
        )
    # Make zero after plateau
    n = np.where( z > l_plateau, 0, n)
    return n * n_p_start


def run_simulation():

    # Determine peak normalized vector potential of the laser.
    a0 = determine_laser_a0(E_laser, tau_laser, w0_laser, lambda0)

    # Create laser.
    laser = GaussianPulse(xi_c=0., l_0=lambda0, w_0=w0_laser, a_0=a0, tau=tau_laser, z_foc=0.)

    # Base beam parameters.
    E_beam = 20  # MeV
    gamma_beam = E_beam/0.511
    n_emitt = 1e-6
    ene_sp = 0.1  # %
    n_part = 1e5
    st0 = 1e-15  # gaussian decay [s]
    kp = np.sqrt(n_p_start * ct.e**2 / (ct.epsilon_0 * ct.m_e * ct.c**2))
    kbeta = kp / np.sqrt(2. * gamma_beam)  # betatron wavenumber (blowout)
    betax0 = 1. / kbeta   # matched beta

    # Generate bunch
    x, y, z, ux, uy, uz, q = generate_gaussian_bunch_from_twiss(
        a_x=0, a_y=0, b_x=betax0, b_y=betax0, en_x=n_emitt, en_y=n_emitt,
        ene=E_beam/0.511, ene_sp=ene_sp, s_t=l_beam/ct.c, q_tot=beam_q,
        n_part=int(n_part), z_c=0., lon_profile='flattop_smoothed', smooth_sigma=st0,
        smooth_trunc=3)

    # Reposition bunch.
    z -= l_beam/2 + z_beam

    # Create Wake-T ParticleBunch.
    bunch = ParticleBunch(q, x, y, z, ux, uy, uz, name='bunch')

    # Distance between right boundary and laser centroid.
    dz_lb = 4. * ct.c * tau_laser

    # Maximum radial extension of the plasma.
    p_rmax = 2.5 * w0_laser

    # Box lenght.
    l_box = dz_lb + 50e-6

    # Bow width
    r_max = w0_laser * 4

    # Grid resolution
    s_d = ge.plasma_skin_depth(n_p_start*1e-6)
    dr = s_d / 20
    dz = tau_laser * ct.c / 40

    # Number of diagnostics
    n_out = 10    

    # Determine guiding channel.
    r_e = ct.e**2 / (4. * np.pi * ct.epsilon_0 * ct.m_e * ct.c**2)
    rel_delta_n_over_w2 = 0. # 1. / (np.pi * r_e * w0_laser**4 * n_p_start)

    # Plasma stage.
    plasma = PlasmaStage(
        length=l_plateau,
        density=density_profile,
        wakefield_model='quasistatic_2d',
        n_out=n_out,
        laser=laser,
        laser_evolution=True,
        r_max=r_max,
        r_max_plasma=p_rmax,
        xi_min=dz_lb-l_box,
        xi_max=dz_lb,
        n_r=int(r_max / dr),
        n_xi=int(l_box / dz),
        dz_fields=l_box,
        ppc=2,
        parabolic_coefficient=rel_delta_n_over_w2
    )

    # Do tracking.
    plasma.track(bunch, opmd_diag=True, out_initial=False, diag_dir='diags')


def determine_laser_a0(ene, tau_fwhm, w0, lambda0):
    tau = tau_fwhm / np.sqrt(2. * np.log(2))
    k0 = 2. * np.pi / lambda0  # Laser wavenumber
    PA = ct.epsilon_0 * ct.c**5 * ct.m_e**2 / ct.e**2  # Power constant
    P0 = ene / (np.sqrt(2 * np.pi) * (tau / 2))
    i0 = P0 / ((np.pi / 2)* w0**2)
    a0 = np.sqrt(i0 / (PA * k0**2 / 2))
    return a0


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    
    #----- assel changes ------

    # Initialize parameters.
    # Base laser parameters.
    E_laser = namespace.energy_laser # J
    tau_laser = namespace.tau_laser  # s (fwhm)
    lambda0 = namespace.lambda0   # m
    w0_laser = namespace.w0_laser    # m

    # Get oprimization parameters.
    task = namespace.task  # {{task}}
    beam_q = namespace.beam_q
    beam_z_head = namespace.beam_z_head
    beam_length = namespace.beam_length
    n_p_start = namespace.n_p_start
    n_p_end = n_p_start * 1.
    l_plateau = namespace.l_plateau


    # Beam parameters.
    z_beam = (10 + beam_z_head) * 1e-6
    l_beam = beam_length * 1e-6

#--------------------------    


    run_simulation()




















