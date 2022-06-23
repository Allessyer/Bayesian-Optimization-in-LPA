import numpy as np
import scipy.constants as ct
from wake_t import GaussianPulse, PlasmaStage, ParticleBunch
import aptools.plasma_accel.general_equations as ge
from aptools.utilities.bunch_generation import generate_gaussian_bunch_from_twiss

#----------
from aptools.data_processing.beam_filtering import filter_beam
import visualpic as vp
import os
import shutil
import scipy.constants as scc

def delete_dir(dir_name):
    # location
    location = "./"
    
    # path
    path = os.path.join(location, dir_name)

    # removing directory
    shutil.rmtree(path)

def analyse_simulation(varied_param,filtered,obj_type):
    
    E_laser, head_current = varied_param
    
    E_laser = E_laser * 6.242* 1e12 # MeV
    
    dc = vp.DataContainer('openpmd',
                                  os.path.join('./', 'diags/hdf5'))
    dc.load_data()

    bunch = dc.get_species('bunch')
    ts = bunch.timesteps
    bunch_data = bunch.get_data(ts[-1])
    x = bunch_data['x'][0]
    y = bunch_data['y'][0]
    px = bunch_data['px'][0]
    py = bunch_data['py'][0]
    pz = bunch_data['pz'][0]
    q = bunch_data['q'][0]
    
    if filtered:
        x, y, px, py, pz, q = filter_beam(
            np.array([x, y, px, py, pz, q]),
            [None, None, None, None, 150 / 0.511, None],
            [None, None, None, None, None, None]
        )
    
    E_total = np.sqrt(1 + px**2 + py**2 + pz**2)* 0.511 # in MeV
    

    Energy_beam = [] # different calculation of E_beam (don't know which is correct)

    # Beam energy (total energy of all electrons) calculation
    Energy_beam_init = 20
    Energy_z_electron = ((pz-1) * 0.511) - Energy_beam_init # Energy z-component of electron
    E_beam = np.abs(np.sum(Energy_z_electron * q / scc.e)) # Total energy of the beam (calculated as desy proposed)
    Energy_beam.append(E_beam)

    # without * q / e (assel calculation of Beam energy)
    E_beam = np.sum(E_total)
    Energy_beam.append(E_beam)

    # with * q / e (assel calculation of Beam energy)
    E_beam = np.abs(np.sum(E_total * q / scc.e))
    Energy_beam.append(E_beam)

    # Objective func proposed by DESY
    Energy_electron = 150
    E_beam = np.abs(np.sum(Energy_electron * q / scc.e))
    Energy_beam.append(E_beam)

    E_conv = []

    for E_beam in Energy_beam:
        E_conv.append(100 * E_beam / E_laser)

    obj_func = E_conv

    # Absolute error
    E_average = np.average(E_total, weights=q)
    obj_func.append(np.abs(E_average-150))

    # Mean Square Error
    N_electrons = E_total.shape[0]
    MSE = 1/N_electrons * np.sum(np.square(np.subtract(E_total,150)))
    obj_func.append(MSE)

    print(f'obj_func = {obj_func}')

    # e_conv_z, e_conv_noq, e_conv_q, desy_obj, abs_error, MSE

    # Задача 1: конвертация энергии до 100% ---> maximize it
    if obj_type == "e_conv_z":
        return obj_func[0]
    elif obj_type == "e_conv_noq":
        return obj_func[1]
    elif obj_type == "e_conv_q":
        return obj_func[2]
    elif obj_type == "desy_obj":
        return obj_func[3]

    # Задача 2: ускорить до 150 МэВ
    # absolute error ---> minimize it
    elif obj_type == "abs_error":
        return obj_func[4]

    # Mean Square Error ---> minimize it
    elif obj_type == "MSE":
        return obj_func[5]


def generate_gaussian_bunch_from_twiss_assel(
        a_x, a_y, b_x, b_y, en_x, en_y, ene, ene_sp, s_t, q_tot, n_part, head_current, x_c=0,
        y_c=0, z_c=0, lon_profile='gauss', min_len_scale_noise=None,
        sigma_trunc_lon=None, smooth_sigma=None, smooth_trunc=None,
        save_to_file=False, save_to_code='astra',
        save_to_path=None, file_name=None, perform_checks=False):
    """
    Creates a transversely Gaussian particle bunch with the specified Twiss
    parameters.

    Parameters
    ----------
    a_x : float
        Alpha parameter in the x-plane.

    a_y : float
        Alpha parameter in the y-plane.

    b_x : float
        Beta parameter in the x-plane in units of m.

    b_y : float
        Beta parameter in the y-plane in units of m.

    en_x : float
        Normalized trace-space emittance in the x-plane in units of m*rad.

    en_y : float
        Normalized trace-space emittance in the y-plane in units of m*rad.

    ene: float
        Mean bunch energy in non-dimmensional units (beta*gamma).

    ene_sp: float
        Relative energy spread in %.

    s_t: float
        Bunch duration in seconds. If lon_profile='gauss', this corresponds to
        the RMS duration. If lon_profile='flattop' or
        lon_profile='flattop_smoothed', this instead the whole flat-top lenght.

    q_tot: float
        Total bunch charge in C.

    n_part: int
        Total number of particles in the bunch.
    
    head_current: float
        takes a value between 0 and 1.
        if it is 1 all the current is in the head, if it 0 all the current is in the tail, and if it is 0.5 both head and tail have the same current.

    x_c: float
        Central bunch position in the x-plane in units of m.

    y_c: float
        Central bunch position in the y-plane in units of m.

    z_c: float
        Central bunch position in the z-plane in units of m.

    lon_profile: string
        Longitudonal profile of the bunch. Possible values are 'gauss' and
        'flattop'.

    min_len_scale_noise: float
        (optional) If specified, a different algorithm to generate a less noisy
        longitudinal profile is used. This algorithm creates a profile that is
        smooth for a longitudinal binning of the bunch with
        bin lengths >= min_len_scale_noise

    sigma_trunc_lon: float
        (optional) If specified, it truncates the longitudinal distribution of
        the bunch between [z_c-sigma_trunc_lon*s_z, z_c+sigma_trunc_lon*s_z].
        Only used when lon_profile = 'gauss' and required if
        min_len_scale_noise is specified.

    smooth_sigma: float
        The sigma of the Gaussian longitudinal smoothing applied to the
        flat-top profile when lon_profile='flattop_smoothed'. Units are in
        seconds.

    smooth_trunc: float
        Number of sigmas after which to truncate the Gaussian smoothing when
        lon_profile='flattop_smoothed'

    save_to_file: bool
        Whether to save the generated distribution to a file.

    save_to_code: string
        (optional) Name of the target code that will use the saved file.
        Possible values are 'csrtrack', 'astra' and 'fbpic'. Required if
        save_to_file=True.

    save_to_path: string
        (optional) Path to the folder where to save the data. Required if
        save_to_file=True.

    file_name: string
        (optional) Name of the file where to store the beam data. Required if
        save_to_file=True.

    perform_checks: bool
        Whether to compute and print the parameters of the generated bunch.

    Returns
    -------
    The 6D components and charge of the bunch in 7 arrays.

    """
    print('Generating particle distribution... ', end='')
    # Calculate necessary values
    n_part = int(n_part)
    ene_sp = ene_sp/100
    ene_sp_abs = ene_sp*ene
    s_z = s_t*ct.c
    em_x = en_x/ene
    em_y = en_y/ene
    g_x = (1+a_x**2)/b_x
    g_y = (1+a_y**2)/b_y
    s_x = np.sqrt(em_x*b_x)
    s_y = np.sqrt(em_y*b_y)
    s_xp = np.sqrt(em_x*g_x)
    s_yp = np.sqrt(em_y*g_y)
    p_x = -a_x*em_x/(s_x*s_xp)
    p_y = -a_y*em_y/(s_y*s_yp)
    # Create longitudinal distributions
    if lon_profile == 'gauss':
        z = _create_gaussian_longitudinal_profile(z_c, s_z, n_part,
                                                  sigma_trunc_lon,
                                                  min_len_scale_noise)
    elif lon_profile == 'flattop':
        z = _create_flattop_longitudinal_profile(z_c, s_z, n_part,
                                                 min_len_scale_noise)
    elif lon_profile == 'flattop_smoothed':
        z = _create_flattop_longitudinal_profile_with_smoothing(
            z_c, s_z, n_part, min_len_scale_noise, smooth_sigma, smooth_trunc)
    elif lon_profile == 'rectan_trapezoidal':
        z = _create_rectan_trapezoidal_longitudinal_profile(z_c, s_z, n_part,head_current,
                                         min_len_scale_noise)
    elif lon_profile == 'rectan_trapezoidal_smoothed':
        z = _create_rectan_trapezoidal_longitudinal_profile_smoothed(z_c, s_z, n_part,
                                                                           head_current,
                                                                           min_len_scale_noise)

    # Define again n_part in case it changed when crealing long. profile
    n_part = len(z)
    pz = np.random.normal(ene, ene_sp_abs, n_part)
    # Create normalized gaussian distributions
    u_x = np.random.standard_normal(n_part)
    v_x = np.random.standard_normal(n_part)
    u_y = np.random.standard_normal(n_part)
    v_y = np.random.standard_normal(n_part)
    # Calculate transverse particle distributions
    x = s_x*u_x
    xp = s_xp*(p_x*u_x + np.sqrt(1-np.square(p_x))*v_x)
    y = s_y*u_y
    yp = s_yp*(p_y*u_y + np.sqrt(1-np.square(p_y))*v_y)
    # Change from slope to momentum
    px = xp*pz
    py = yp*pz
    # Charge
    q = np.ones(n_part)*(q_tot/n_part)
    print('Done.')
    # Save to file
    if save_to_file:
        print('Saving to file... ', end='')
        ds.save_beam(
            save_to_code, [x, y, z, px, py, pz, q], save_to_path, file_name)
        print('Done.')
    if perform_checks:
        _check_beam_parameters(x, y, z, px, py, pz, q)
    return x, y, z, px, py, pz, q


def _create_rectan_trapezoidal_longitudinal_profile(z_c, length, n_part, head_current,
                                         min_len_scale_noise):
    """ Creates a rectangular trapezoidal longitudinal profile """
    
    # Make sure number of particles is an integer
    n_part = int(n_part)
    if min_len_scale_noise is None:
        a = z_c-length/2
        b = z_c+length/2
        
        head_current = head_current * 2/(b-a)
        z = rectan_trapezoid(n_part,a=a,b=b,h1=head_current)
    else:
        raise NotImplementedError('Noise reduction not implemented for `trapezoidal` profile.')
    return z


def _create_rectan_trapezoidal_longitudinal_profile_smoothed(z_c, length, n_part, head_current,
                                         min_len_scale_noise):
    
    """ Creates a rectangular trapezoidal smoothed longitudinal profile """
    
    # Make sure number of particles is an integer
    n_part = int(n_part)
    if min_len_scale_noise is None:
        z = rectan_trapezoid_smoothed(n_part,a=z_c-length/2,b=z_c+length/2,h1=head_current)
    else:
        raise NotImplementedError('Noise reduction not implemented for `trapezoidal` profile.')
    return z
    

def rectan_trapezoid(n_part,a,b,h1):
    
    """
    Creates a longitudinal rectangular trapezoid particle bunch with the specified
    parameters.
    Parameters
    ----------
    n_part : float
        Number of particles in the beam
       
             __       
          __/  |
       __/     |
     /         |
    |          |
    |h1        |h2 
    |__________|
    a          b
    
    a : float
        start position of the trapezoid
    b : float
        end position of the trapezoid
    h1 : float
        % from the maximum height
    h2 : float
        second height
    plot : bool
        If True, then plot histogram of the distribution,
    
    ----------
    Returns
    
    x : array with size [n_part]
        distribution of random variables with rectangular trapezoidal shape
        
    """
    if np.abs(h1 - 0.5) < 1e-12:
        x = np.random.uniform(a, b, n_part)
    else:
        h2 = 2/(b-a) - h1
        if h2 < 0:
            raise ValueError("h2 = 2/(b-a) - h1 < 0!")
            
    n_part = int(n_part)
    y = np.random.uniform(0,1,n_part)
    x = np.zeros(len(y))
    for i, y_i in enumerate(y):
        if y_i >= 0 and y_i <= 1:
            n = h1
            m = (h2-h1)/(2*(b-a))
            D = n**2 + 4 * m * y_i
            x[i] = (-n+np.sqrt(D))/(2*m) + a
    
    return x


def half_gaussian(n_part, mu, peak, part='left'):
    sigma = 1 / (peak / 2 * np.sqrt(2 * np.pi))
    x = np.random.normal(loc=mu, scale=sigma, size=n_part)
    if part == 'right':
        idx = x < mu
    elif part == 'left':
        idx = x > mu
    else:
        raise ValueError('Wrong part')
    x[idx] = 2 * mu - x[idx]
    return x

def rectan_trapezoid_smoothed(n_part, h1, a, b, left_bound=0.05, right_bound=0.95):
    
    h1 = h1 * 2/(b-a)
    weight_trapz = right_bound - left_bound
    h2 = weight_trapz * (2/(b-a) - h1 / weight_trapz)
    if h2 < 0:
        raise ValueError("h2 = 2/(b-a) - h1 < 0!")
    p = np.random.rand(n_part)
    x = np.zeros_like(p)
    
    # left gaussian:
    idx_left = p < left_bound
    x[idx_left] = half_gaussian(np.count_nonzero(idx_left), mu=a, peak=h1 / left_bound, part='left')
    
    # right gaussian:
    idx_right = p > right_bound
    x[idx_right] = half_gaussian(np.count_nonzero(idx_right), mu=b, peak=h2 / (1 - right_bound), part='right')
    
    # trapezoidal:
    idx = np.logical_not(np.logical_or(idx_left, idx_right))
    x[idx] = rectan_trapezoid(np.count_nonzero(idx), a, b, h1 / weight_trapz)

    return x


#----------------------------------------------------

