#!/bin/bash

#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --time=03:00:00

module load hosts/hopper gnu10/10.3.0-ya
module load python/3.10.1-5r
pip install statsmodels
python simulation_study.py