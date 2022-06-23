# Bayesian-Optimization-in-LPA
Application of Bayesian optimization to improve beam quality of laser plasma accelerators.

## Introduction to the problem

At the moment, traditional accelerators have their limits, and we need to find alternatives for them. Laser plasma accelerators (LPA) are very inter-
esting as a replacement for conventional accelerators, because of their small size and large accelerating fields. However producing high-quality beams
concerning industrial needs requires a meticulous balancing of a variety of physical effects and, as a result, is both conceptually and experimentally
hard. 

This work investigates how to optimize the source of electrons in the LPA for given requests from industry or academic institutions. Particularly,
the use of Bayesian Optimization on LPA parameters to achieve 150 MeV acceleration with a maximum energy conversion efficiency from the laser to the electron beam is studied.

The study can be divided into 4 main tasks:
Task 1. Create appropriate objective functions and evaluation metric, and analyze and select the set of parameters to be optimized.
Task 2. Investigate Bayesian Optimization with 1 parameter and compare results with optimum found by GridSearch
Task 3. 2 parameters optimization
Task 4. 9 parameters optimization

## Results

### Task 1

Three objective functions were used: absoulte error, energy conversion and artificial energy conversion. 
As evaluation metrics energy of the beam after acceleration and energy conversion were used.

During the study a new function has been created that changes the longitudinal profile of the beam. 





