#!/bin/bash

name="energy_laser"
i=41
for value in `seq 5 0.1 20`
do
    mkdir sa$i
    cp LPA_stage.py sa$i
    cp -r my_functions sa$i
    cp work_sbatch.sh sa$i
    cd sa$i
    sbatch work_sbatch.sh --$name $value 
    cd ../
    pwd
    
    i=$((i+1))
    
done




