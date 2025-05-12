#!/bin/bash --login
# Job name:
#SBATCH --job-name=tm_5

# Number of tasks (processes)
# SLURM defaults to 1 but we specify anyway
#SBATCH --ntasks=1

# Type and number of GPUs
# The type is optional.
#SBATCH --gpus=v100:1

# Memory per node
# Specify "M" or "G" for MB and GB respectively
#SBATCH --mem=64G

# Wall time
# Format: "minutes", "hours:minutes:seconds", 
# "days-hours", or "days-hours:minutes"
#SBATCH --time=1:30:00

# Mail type
# e.g., which events trigger email notifications
#SBATCH --mail-type=ALL

# Mail address
#SBATCH --mail-user=lellaom@msu.edu

# Standard output and error to file
# %x: job name, %j: job ID
#SBATCH --output=%x-%j.SLURMout

#SBATCH --output=tm_5.out
#SBATCH --error=tm_5.err

export CONDA3PATH=/mnt/home/lellaom/anaconda3
conda activate venv

# Run our job
cd /mnt/scratch/lellaom/narrative_theme_identification/code/itr5_250
srun python topic_modeling.py

# Print resource information
scontrol show job $SLURM_JOB_ID
js -j $SLURM_JOB_ID