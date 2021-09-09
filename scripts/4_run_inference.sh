#!/bin/bash

#SBATCH --job-name=sbi
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:0
#SBATCH --mem=3G
#SBATCH --output ../output/slurm/slurm-%A_%a.out


if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "Running script directly, queue with sbatch to use slurm instead."
    python inference.py
else
    cd "${SLURM_SUBMIT_DIR}"
    echo Running as user "$(whoami)" in directory "$(pwd)" with job ID "${SLURM_JOB_ID}"
    module load languages/miniconda/3
    . ~/.bashrc
    conda activate wildcats_env
    echo Running "$(python --version)" from "$(which python)"
    python inference.py
    conda deactivate
fi

