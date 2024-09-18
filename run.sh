#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=0-06:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=UR_parallel
#SBATCH --output=slurm_%j.out

julia -t auto solver/main.jl