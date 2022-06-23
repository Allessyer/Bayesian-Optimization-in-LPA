#!/bin/zsh

#SBATCH --partition=mpa
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --job-name  waket
#SBATCH --output    output.txt
#SBATCH --error     error.txt

name=$1 # Do not forget name goes with "--"
value=$2
python work_waket_script.py $name $value

#FILE=diags
#if [ -d "$FILE" ]; then
#    echo "$FILE directory has been found."
#    mv diags "$MY_PATH/$MY_FOLDER"
#    rm error.txt
#    rm output.txt
#else
#    mv error.txt "$MY_PATH/$MY_FOLDER"
#    mv output.txt "$MY_PATH/$MY_FOLDER"
#fi


