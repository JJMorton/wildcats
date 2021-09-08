import inference.config as config
from subprocess import run
from os.path import exists
import math

def main():
    
    if not exists(config.parameters_file):
        print(f'Samples file "{config.parameters_file}" doesn\'t exist, aborting.')
        exit(1)

    batches = int(math.floor(config.num_simulations/10))
    print(f'Will run {config.num_simulations} simulations, which will require {batches} batches of 10 simulations')
    command = ["sbatch", f'--array=0-{batches - 1}', "simulate_job.sh"]
    if input(f'Executing "{" ".join(command)}", is this okay? [y/n] ') == 'y':
        run(command)
        print("Wait until simulations have finished running before continuing with next step. Monitor progress with 'sacct -j <job id>'")
        print(f'Stats will be saved in individual files in "{config.simulation_output_dir}", the next step is to merge them together')
    else:
        print("You can run a similar command to above yourself, if you need to change it")

if __name__ == "__main__":
    main()
