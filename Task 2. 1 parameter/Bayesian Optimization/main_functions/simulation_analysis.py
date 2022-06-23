import matplotlib.pyplot as plt
import sys, time
import numpy as np
import scipy.constants as ct
import math

from openpmd_viewer import OpenPMDTimeSeries
from visualpic import DataContainer

import glob
from PIL import Image

import os
from os import listdir
from os.path import isfile, join
import re

def gif_creation(path2images,path2gif,gif_name):
    # filepaths
    fp_in = f"{path2images}/*.png"
    fp_out = f"{path2gif}/{gif_name}.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=1000, loop=0)

def create_dir(path,dir_name):

    try:
        os.mkdir(f'{path}/{dir_name}')
    except OSError:
        print ("Creation of the directory %s failed" % dir_name)
    else:
        print ("Successfully created the directory %s " % dir_name)


def find_folder(file,files_list):
    for text in files_list:
        string = file
        match = re.search(string,text)
        if match:
            text_pos = match.span()
            return text[match.start():]
        
def look_objective(obj_name,path2sim,N_sim,plot=True):
    obg_func = []
    
    mypath = path2sim
    onlyfiles = [f for f in listdir(mypath)]
    for i in range(N_sim+1):
        try:
            folder = f'sim{i}_worker*'
            folder = find_folder(folder,onlyfiles)

            f = open(f"{mypath}/{folder}/{obj_name}.txt", "r")
            f = f.read().split('\n')
            f = float(f[0])
            obg_func.append(f)
        except:
#             print(i)
              obg_func.append(obg_func[-1])
              pass
            
            
    with open(f'{path2sim}/look_objective_{obj_name}.txt', 'w') as f:
        f.write(f'Minimum is in {np.argmin(obg_func)}')
        f.write('\n')
        f.write(f'Best value is {obg_func[np.argmin(obg_func)] * -1}')
        f.write('\n')
        
    print('Minimum is in ',np.argmin(obg_func))
    print('Best value is ',obg_func[np.argmin(obg_func)] * -1)
    
    
    
    if plot:    
        plt.plot(range(len(obg_func)),obg_func)
        plt.plot(np.minimum.accumulate(obg_func))
        plt.xlabel('Optimization step')
        plt.ylabel(f'{obj_name}')
        plt.title(f'{obj_name}:N_sim = {N_sim}')
        plt.savefig(f'{path2sim}/look_objective_{obj_name}_Nsim_{N_sim}.png')
    
    return np.argmin(obg_func), obg_func

def obj_func_history(obg_func):
    '''
    This function creates dictionary { Simulation number: objective function value }
    and sorted it in descent order
    and returns it
    '''
    dict_obj = {}

    for i in range(len(obg_func)):
        dict_obj[i] = []
        dict_obj[i].append(obg_func[i])

    sorted_dict_obj = {k: v for k, v in sorted(dict_obj.items(), key=lambda item: item[1])}
    
    return sorted_dict_obj

def give_input_params(N_sim, path2sim,input_params_names):
    
    mypath = path2sim
    onlyfiles = [f for f in listdir(mypath)]
    
    folder = f'sim{N_sim}_worker*'
    folder = find_folder(folder,onlyfiles)

    path2diags = f'{path2sim}/{folder}'
    
    values_input_params = []
    for param in input_params_names:
        try:
            value = [ line.rstrip('\n') for line in open(f'{path2diags}/simulation.py') if f'{param}' in line]
            value = value[0].split('  #')[0]
            values_input_params.append(value)
        except:
            continue
        
    input_params = {}
    for i in range(len(values_input_params)):
        name, value = values_input_params[i].split(' = ')
        
        if value.find("*") == -1:
            value = float(value)
        else:
            a, b = value.split(" * ")
            if name != 'n_p_end':
                value = float(a) * float(b)
            else:
                value = float(b)

        input_params[name] = value
    
    return input_params

def give_param_convergence(first_N_simulations,obj_func,input_params_names,path2sim):
    
    sorted_dict_obj = obj_func_history(obj_func)
    for N_sim in list(sorted_dict_obj.keys())[:first_N_simulations]:
        input_params = give_input_params(N_sim, path2sim,input_params_names)
        sorted_dict_obj[N_sim].append(input_params)
        
    params_convergence = {}
    for param in input_params_names:
        param_values = []
        for N_sim in list(sorted_dict_obj.keys())[:first_N_simulations]:
            t = sorted_dict_obj[N_sim][1][param]
            param_values.append(t)

        params_convergence[param] = param_values
        
    return params_convergence

def plot_results(path2diags,path2sim,input_params_names,from_iter=9,till_iter=9):
    
    ts = OpenPMDTimeSeries(f'{path2diags}/diags/hdf5/', backend='h5py')
    
    values_input_params = []
    for param in input_params_names:
        try:
            value = [ line.rstrip('\n') for line in open(f'{path2diags}/simulation.py') if f'{param}' in line]
            value = value[0].split('  #')[0]
            values_input_params.append(value)
        except:
            continue
    with open(f'{path2sim}/best_sim_parameters.txt', 'w') as f:
        for i in values_input_params:
            f.write(i)
            f.write('\n')
            print(i)
        
    
    rows = 4
    columns = till_iter+1 - from_iter
    fig = plt.figure(figsize=(5*columns, 3*rows))
    grid = plt.GridSpec(rows, columns, wspace = .25, hspace = .25)
    N_iterations = till_iter+1 - from_iter
    
    Beam_energy = []

    for index,iteration in enumerate(range(from_iter,till_iter+1)):
        z,uz = ts.get_particle(var_list = ['z','uz'],species='bunch',iteration=iteration)
        above150_uz = []
        above150_z = []
        below150_uz = []
        below150_z = []
        for i in range(len(uz)):
            if (uz[i]-1)*0.511 >= 150:
                above150_uz.append(uz[i])
                above150_z.append(z[i])
            else:
                below150_uz.append(uz[i])
                below150_z.append(z[i]) 
        Ez, m = ts.get_field(iteration=iteration, field='E', coord='z')

        exec (f"plt.subplot(grid{[columns * 0 + index]})")

        plt.plot(above150_z,above150_uz,'.',ms=0.5,color='orange', label = 'above 150 MeV')
        plt.plot(below150_z,below150_uz,'.',ms=0.5,color='#beab9e',label = 'below 150 MeV')
        plt.grid()
        plt.xlabel('z [m]')
        plt.ylabel('uz [$m_e \cdot c$]')
        plt.title(f"Iteration N {iteration}")
        plt.xlim([m.zmin, m.zmax])

        exec (f"plt.subplot(grid{[columns * 1 + index]})")
        plt.plot(m.z, Ez[Ez.shape[0]//2,:],color='#723505')
        plt.ylabel('Ez [MeV]')
        plt.xlabel('z [m]')
        plt.xlim([m.zmin, m.zmax])
        plt.grid()


        x, z, ux, uz = ts.get_particle(iteration=iteration, var_list=['x', 'z', 'ux', 'uz'], species='bunch')
        print("Energy [MeV]", .511*(np.mean(uz)-1))
        Beam_energy.append(.511*(np.mean(uz)-1))
        
        F, m = ts.get_field(iteration=iteration, field='E', coord='z')
        L, m = ts.get_field(iteration=iteration, field='a_mod')


        exec (f"plt.subplot(grid{[columns * 2 + index]})")
        plt.imshow(F, extent=m.imshow_extent, aspect='auto')
        plt.clim(-1.e11,1.e11)
        plt.colorbar()

        plt.plot(z,x,'k.',ms=.1)
        plt.xlim([m.zmin, m.zmax])
        plt.ylim([m.rmin, m.rmax])
        plt.grid()

        exec (f"plt.subplot(grid{[columns * 3 + index]})")
        plt.imshow(L, extent=m.imshow_extent, aspect='auto')
        plt.colorbar()

        plt.plot(z,x,'k.',ms=.1)
        plt.xlim([m.zmin, m.zmax])
        plt.ylim([m.rmin, m.rmax])
        plt.grid()

    plt.savefig(f'{path2sim}/best_sim_analysis_plots.png')
    
    Beam_energy = np.array(Beam_energy)
    np.save(f'{path2sim}/beam_energy.npy',Beam_energy)
        
def param_analysis(obj_func,first_N_simulations,input_params_names,path2sim):
    
    params_convergence = give_param_convergence(first_N_simulations,obj_func,input_params_names,path2sim)
    obj_func = sorted(obj_func)
    
    N_plots = len(list(params_convergence.keys()))

    
    rows = 3
    columns = math.ceil(len(input_params_names)/rows)
    fig = plt.figure(figsize=(10*columns, 5*rows))
    
    grid = plt.GridSpec(rows, columns, wspace = .25, hspace = .25)

    for i,param in enumerate(params_convergence.keys()):
        exec (f"plt.subplot(grid{[i]})")
        plt.plot(obj_func[:first_N_simulations], params_convergence[param],'.',ms=5)
        plt.title(f"{param}. First {first_N_simulations} best simulations")
        plt.xlabel('Objective function')
        plt.ylabel(f'{param}')
    
    plt.savefig(f'{path2sim}/param_analysis.png')
    
def best_sim_analysis(path2diags,best_sim,input_params_names,from_iter,till_iter):

    plot_results(path2diags,path2sim,input_params_names,from_iter=from_iter,till_iter=till_iter)
    
   