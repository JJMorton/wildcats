#!/bin/bash

#SBATCH --job-name=slim
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:45:0
#SBATCH --mem=3G
#SBATCH --array=700-701
#SBATCH --output ../output/slurm/slurm-%A_%a.out

sims_per_task=10

cd "${SLURM_SUBMIT_DIR}"
echo Running as user "$(whoami)" in directory "$(pwd)" with job ID "${SLURM_JOB_ID}"

module load languages/anaconda3/2021.05-Popcorn
source activate wildcats
echo Running "$(python --version)" from "$(which python)"

# Ensure directory structure is correct
[[ -e ../output ]] || mkdir ../output
[[ -e ../output/stats ]] || mkdir ../output/stats
if ! [[ -e ../data/simulate_params.feather ]]; then
	echo "Parameters data file doesn't exist, aborting..."
	exit 1
fi

first=$((SLURM_ARRAY_TASK_ID * sims_per_task))
last=$((first + sims_per_task - 1))
echo Will run simulations $first to $last

for id in $(seq $first $last); do
	echo Running simulation id ${id}...
	time python simulate.py $id
done

conda deactivate

