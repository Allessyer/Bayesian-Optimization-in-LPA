from my_functions.assel_functions import analyse_simulation
from my_functions.run_simulation import run_simulation
import numpy as np

import sys
import argparse
from timeit import default_timer as timer

def createParser ():
    parser = argparse.ArgumentParser()

    # Base laser parameters.
    parser.add_argument ('-e_laser', '--energy_laser',type=float, default=1) # J

    return parser

def LPA_stage(varied_param):
    
    start = timer()
    
    E_laser = varied_param
    bunch, bunch_list = run_simulation(varied_param)
    end = timer()
    print(f"Elapsed time in minutes =  {(end - start) / 60}")
    
    # without filter
    for i in range(1,4):
        objective_function = analyse_simulation(E_laser,bunch_list,filtered=False,obj_type=i)
        np.savetxt(f'NoFilter_obj_type_{i}.txt', np.array([objective_function]))
        
    #with filter
    for i in range(1,4):
        objective_function = analyse_simulation(E_laser,bunch_list,filtered=True,obj_type=i)
        np.savetxt(f'Filter_obj_type_{i}.txt', np.array([objective_function]))
        
    
    
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    # Initialize parameters.
    E_laser = namespace.energy_laser # J
    np.savetxt('E_laser.txt', np.array([E_laser]))
    LPA_stage(E_laser)
    
    
    
    
    