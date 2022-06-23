#!/bin/zsh

energy_laser=(1 2 4 8) #s4
tau_laser=(7e-15 15e-15 60e-15 120e-15) #s5
lambda0=(0.2e-6 0.4e-6 1.6e-6 3.2e-6) #s6
w0_laser=(5e-6 10e-6 40e-6 80e-6) #s7

beam_q=(1.e-12 50.e-12 100.e-12 150e-12 200e-12) #s8
beam_length=(1 3 5 7 10) #s9
n_p_start=(1e24 2e24 4e24 8e24 10e24) #s10
l_plateau=(1.e-3 3.e-3 5.e-3 10.e-3) #s11

zsh work_script.sh --energy_laser 1. 2. 4. 8. s4
zsh work_script.sh --tau_laser 7.e-15 15.e-15 60.e-15 120.e-15 s5
zsh work_script.sh --lambda0 0.2e-6 0.4e-6 1.6e-6 3.2e-6 s6
zsh work_script.sh --w0_laser 5.e-6 10.e-6 40.e-6 80.e-6 s7

zsh work_script.sh --beam_q 50.e-12 100.e-12 150.e-12 200.e-12 s8
zsh work_script.sh --beam_length 1. 3. 5. 7. s9
zsh work_script.sh --n_p_start 1.e24 2.e24 4.e24 8.e24 s10
zsh work_script.sh --l_plateau 1.e-3 3.e-3 5.e-3 10.e-3 s11

echo work is finished...........


