
import numpy as np
from visualpic import DataContainer
import matplotlib.pyplot as plt
import scipy.constants as ct


def analyze_sim(path2dir,plot=True):
    data_folder = f'{path2dir}/diags/hdf5'
    sim_code = 'openPMD'
    dc = DataContainer(sim_code, data_folder)
    dc.load_data()
    a_env = dc.get_field('a_mod')
    a_phi = dc.get_field('a_phase')
    its = a_env.timesteps
    laser_w = np.zeros(len(its))
    laser_ene = np.zeros(len(its))
    for i, it in enumerate(its):
        env, md = a_env.get_data(it)
        a_phi_data,*_ = a_phi.get_data(it) #, slice_i=0.5, slice_dir_i='r')
        r = md['axis']['r']['array']
        z = md['axis']['z']['array']
        Nr = int(len(r) / 2)
        Nz = int(len(z))
        dr = np.max(r) / Nr
        dz = (z[-1]-z[0]) / (Nz - 1)
        
#         plt.subplot(121)
#         plt.imshow(a_phi_data)
#         plt.subplot(122)
#         plt.imshow(env)
        
        e_env = get_e_env(env, a_phi_data, dz)
        laser_w[i] = calculate_spot_size(e_env, dr)
        laser_ene[i] = calculate_energy(e_env, dr, dz, np.linspace(dr / 2, np.max(r) - dr / 2, Nr))
        
        
    rows = 1
    columns = 2
    fig = plt.figure(figsize=(10*columns, 7*rows))
    grid = plt.GridSpec(rows, columns, wspace = .25, hspace = .25)
    
    if plot:
        exec (f"plt.subplot(grid{[0]})")
        plt.plot(laser_w*1e6)
        plt.xlabel('Iteration')
        plt.ylabel('w [µm]')
        plt.title("Laser Width")
        exec (f"plt.subplot(grid{[1]})")
        plt.plot(laser_ene)
        plt.xlabel('Iteration')
        plt.title("Laser Energy")
        plt.ylabel('Laser Energy [J]')
        plt.show()
    
    plt.savefig(f'{path2dir}/laser_analysis.png')
        
        
    
    return laser_ene

# def analyze_sim2(path2dir,plot=True):
#     data_folder = f'{path2dir}/diags/hdf5'
#     sim_code = 'openPMD'
#     dc = DataContainer(sim_code, data_folder)
#     dc.load_data()
#     a_env = dc.get_field('a_mod')
#     a_phi = dc.get_field('a_phase')
#     its = a_env.timesteps
#     laser_w = np.zeros(len(its))
#     laser_ene = np.zeros(len(its))
#     for i, it in enumerate(its):
#         env, md = a_env.get_data(it)
#         a_phi_data,*_ = a_phi.get_data(it) #, slice_i=0.5, slice_dir_i='r')
#         r = md['axis']['r']['array']
#         z = md['axis']['z']['array']
#         Nr = int(len(r) / 2)
#         Nz = int(len(z))
#         dr = np.max(r) / Nr
#         dz = (z[-1]-z[0]) / (Nz - 1)
#         e_env = get_e_env(env, a_phi_data, dz)
#         laser_w[i] = calculate_spot_size(e_env, dr)
#         laser_ene[i] = calculate_energy(e_env, dr, dz, np.linspace(dr / 2, np.max(r) - dr / 2, Nr))
        
#     if plot:
#         plt.plot(laser_ene)
#         plt.xlabel('Iteration')
#         plt.ylabel('Energy [J]')
    

def calculate_spot_size(a_env, dr):
    # Project envelope to r
    a_proj = np.sum(np.abs(a_env), axis=1)

    # Remove lower half (field has radial symmetry)
    nr = len(a_proj)
    a_proj = a_proj[int(nr/2):]

    # Maximum is on axis
    a_max = a_proj[0]

    # Get first index of value below a_max / e
    i_first = np.where(a_proj <= a_max / np.e)[0][0]

    # Do linear interpolation to get more accurate value of w.
    # We build a line y = a + b*x, where:
    #     b = (y_2 - y_1) / (x_2 - x_1)
    #     a = y_1 - b*x_1
    #
    #     y_1 is the value of a_proj at i_first - 1
    #     y_2 is the value of a_proj at i_first
    #     x_1 and x_2 are the radial positions of y_1 and y_2
    #
    # We can then determine the spot size by interpolating between y_1 and y_2,
    # that is, do x = (y - a) / b, where y = a_max/e
    y_1 = a_proj[i_first - 1]
    y_2 = a_proj[i_first]
    x_1 = (i_first-1) * dr + dr/2
    x_2 = i_first * dr + dr/2
    b = (y_2 - y_1) / (x_2 - x_1)
    a = y_1 - b*x_1
    w = (a_max/np.e - a) / b
    return w


def calculate_energy(e_env, dr, dz, r):
    i_axis = int(e_env.shape[0] / 2)
    I = ct.epsilon_0 * ct.c * e_env[i_axis:]**2
    ene = I.T*np.pi*((r+dr/2)**2-(r-dr/2)**2) * dz/ct.c
    ene = np.sum(ene)
    return ene


def get_e_env(a_env, a_phase, dz):
    """
    Calculate electric field envelope 
    
    Parameters:
    -----------
    a_env : array
        2D array containing the (absolute) laser envelope
    a_phase : array
        1D array containing the complex envelope phase on axis
    dz = float
        Longitudinal grid size.
        
    """
    dk = dz_phi(a_phase, dz)
    k_0 = 2*np.pi / 800e-9 
    k = k_0 + dk
    w = ct.c * k
    E_env = a_env * w * (ct.m_e * ct.c / ct.e)
    return E_env


def dz_phi_old(phi, dz):
    """ Calculate longitudinal derivative of the complex phase """
    dz_phi = np.ediff1d(phi, to_end=0)
    dz_phi = np.where(dz_phi>1.5 * np.pi, dz_phi-2*np.pi, dz_phi)
    dz_phi = np.where(dz_phi<-1.5 * np.pi, dz_phi+2*np.pi, dz_phi)
    dz_phi /= dz
    return dz_phi

def dz_phi(phi, dz):
    """ Calculate longitudinal derivative of the complex phase """
    dz_phi = np.diff(phi, axis=1, append=0)
    dz_phi = np.where(dz_phi>1.5 * np.pi, dz_phi-2*np.pi, dz_phi)
    dz_phi = np.where(dz_phi<-1.5 * np.pi, dz_phi+2*np.pi, dz_phi)
    dz_phi /= dz
    return dz_phi

# analyze_sim()