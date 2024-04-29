# STAT778 Final Project
***Sean Steinle***

This repository contains my final project for STAT778 at GMU.

## How To Run

To replicate my results for the project, you can either execute the notebooks in *notebooks/* or run the *applied_study.py* and *simulation_study.py* scripts. If you'd like to replicate the HPC results, you must submit the shell scripts in *scripts/* using Slurm. The default experiment length for the scripts is 100,000 timesteps, which should be computed on a HPC. To get results on a personal computer, run the notebooks and drop the timesteps as necessary. As a final note for running the study scripts, you should either make a *results/* directory or redirect the output of the files via the last argument to *compare_convergences()*.