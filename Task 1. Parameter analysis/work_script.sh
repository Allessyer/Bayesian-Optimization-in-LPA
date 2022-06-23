#!/bin/zsh

name=$1 # Don't forget name with "--" # что хотим варьировать
values=($2 $3 $4 $5)

simulation_name=$6 # example - s4. Папка куда сохранятся все симуляции

# Создаем папку s4
MY_PATH="/beegfs/desy/group/mpa/mpa1/ayermek/simulations/simulations_diags"
MY_FOLDER="$simulation_name"

mkdir -p "$MY_PATH/$MY_FOLDER"


exp_number=0
for value in "${values[@]}"
do
    exp_number=$(( $exp_number + 1 ))
    MY_SUBFOLDER="$MY_FOLDER$exp_number"
    mkdir -p "$MY_PATH/$MY_FOLDER/$MY_SUBFOLDER"
    cp "/beegfs/desy/group/mpa/mpa1/ayermek/simulations/work_files/work_sbatch.sh" "$MY_PATH/$MY_FOLDER/$MY_SUBFOLDER"
    cp "/beegfs/desy/group/mpa/mpa1/ayermek/simulations/work_files/work_waket_script.py" "$MY_PATH/$MY_FOLDER/$MY_SUBFOLDER"
 
    cd "$MY_PATH/$MY_FOLDER/$MY_SUBFOLDER"
    sbatch work_sbatch.sh $name $value 
done

