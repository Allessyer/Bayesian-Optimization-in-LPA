from my_functions.assel_functions2d import analyse_simulation
from my_functions.assel_functions2d import delete_dir
from my_functions.run_simulation2d import run_simulation

from timeit import default_timer as timer

import numpy as np

def LPA_stage(varied_param,filtered,obj_type):
    start = timer()
    bunch, bunch_list = run_simulation(varied_param)
    objective_function = analyse_simulation(varied_param,filtered=filtered,obj_type=obj_type)
    delete_dir('diags')
    end = timer()
    print(f"Elapsed time: {(end - start) / 60} minutes")
    print(f"objective_function = {objective_function}")
    
    return objective_function

#The range is bounded to -10.0 and 10.0 and one global optimal at [0.0, 0.0].
# objective function
def unimodal_func2(X1, X2):
    return 0.26 * (X1**2 + X2**2) - 0.48 * X1 * X2

#The range is bounded to -5.0 and 5.0 and one global optimal at [0.0, 0.0].
def unimodal_func1(X1,X2):
    return X1**2.0 + X2**2.0

# objective function
def unimodal_func3(X1, X2):
    return -torch.cos(X1) * torch.cos(X2) * torch.exp(-((X1 - torch.pi)**2 + (X2 - torch.pi)**2))