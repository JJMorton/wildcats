#!/bin/bash

#SBATCH --job-name=slim_check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:30
#SBATCH --mem=100M
#SBATCH --output ../output/slurm/slurm-%j.out

if [[ -n "$SLURM_JOB_ID" ]]; then
	cd "${SLURM_SUBMIT_DIR}"
	echo Running as user "$(whoami)" in directory "$(pwd)" with job ID "${SLURM_JOB_ID}"
fi

num_simulations=$1
if ! [[ -n $num_simulations ]]; then
	echo "Please specify the number of simulations to check as an argument to this script"
	exit 1
fi
logfile="../output/failed_simulations.log"
echo "Writing failed simulations to ${logfile}..."
rm "$logfile"
touch "$logfile"
for id in $(seq 0 $((num_simulations - 1))); do
	if [[ -e ../output/stats/stats_${id}.csv ]]; then
		[[ -n "$(cat ../output/stats/stats_${id}.csv | grep nan)" ]] && echo $id >> "$logfile"
	else
		echo $id >> "$logfile"
	fi
done
echo "$(cat "$logfile" | wc -l) simulations failed total"

