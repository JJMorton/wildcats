#!/bin/bash

#SBATCH --job-name=slim
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:45:0
#SBATCH --mem=3G
#SBATCH --output ../output/slurm/slurm-%A_%a.out

sims_per_task=10

# Use exit or return depending on whether the script has be executed or sourced
[[ "$0" == "$BASH_SOURCE" ]] && ret=exit || ret=return

cd "${SLURM_SUBMIT_DIR}"
echo Running as user "$(whoami)" in directory "$(pwd)" with job ID "${SLURM_JOB_ID}"

module load languages/miniconda/3
# If, instead of the next two lines, we use 'source activate wildcats_env', conda doesn't seem to
# activate the environment correctly on BlueCrystal, and uses the wrong python interpreter.
# Make sure you have run 'conda init' once before in your login shell before running this script,
# as this creates the initialisation script in your ~/.bashrc.
. ~/.bashrc
conda activate wildcats_env
echo Running "$(python --version)" from "$(which python)"

if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "This task must be run as an array task, e.g. include '--array=0-500' to run 500 simulations. See 'man sbatch' for help"
    $ret 1
fi

first=$((SLURM_ARRAY_TASK_ID * sims_per_task))
last=$((first + sims_per_task - 1))
echo Will run simulations $first to $last

for id in $(seq $first $last); do
    echo Running simulation id ${id}...
    time python simulate.py $id
done

conda deactivate

