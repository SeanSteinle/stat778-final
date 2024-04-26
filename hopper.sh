#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --array=1-10 #remove this for single job runs

module load hosts/hopper gnu10/10.3.0-ya
module load python/3.10.1-qb
mkdir "outputs/array_runs/" -p
python model_analysis.py 12 10 "outputs/array_runs/"