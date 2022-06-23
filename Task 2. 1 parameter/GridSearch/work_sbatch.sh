#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --job-name  waket
#SBATCH --output    output.txt
#SBATCH --error     error.txt

name=$1 # Do not forget name goes with "--"
value=$2
python3 LPA_stage.py $name $value
# python3 get_objective_function.py
#python3 simulation_parsed.py -e_laser $value

