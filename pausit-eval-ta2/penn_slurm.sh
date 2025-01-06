#!/bin/bash
#
#SBATCH --time=168:00:00
#SBATCH --partition=p_nlp
#SBATCH --job-name=pausit-eval
#SBATCH --output=penn_slurm.stdout
#SBATCH --constraint=24GBgpu
#SBATCH --gpus=1
#SBATCH --mem=30G

# Run the program
srun ./penn_slurm_script.sh
