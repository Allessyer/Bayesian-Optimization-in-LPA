from my_functions.assel_functions import analyse_simulation
from my_functions.assel_functions import delete_dir
from my_functions.run_simulation import run_simulation

from timeit import default_timer as timer

import numpy as np

def LPA_stage(varied_param,filtered,obj_type):
    E_laser = varied_param
    start = timer()
    bunch, bunch_list = run_simulation(varied_param)
    objective_function = analyse_simulation(E_laser,filtered=filtered,obj_type=obj_type)
    delete_dir('diags')
    end = timer()
    print(f"Elapsed time: {(end - start) / 60} minutes")
    print(f"objective_function = {objective_function}")
    return objective_function

def forrester(varied_param):
    x = varied_param
    return (6 * x - 2)**2 * np.sin(12 * x - 4)