import inference.config as config
from os import execv
from os.path import exists

def main():
    
    if not exists(config.parameters_file):
        print(f'Samples file "{config.parameters_file}" doesn\'t exist, aborting.')
        exit(1)

    command = ["sbatch", f'--array=0-{config.num_simulations - 1}', "simulate_job.sh"]
    if input(f'Executing "{" ".join(command)}", is this okay? [y/n] ') == 'y':
        execv(command)
        print("Wait until simulations have finished running before continuing with next step. Monitor progress with 'sacct -j <job id>'")
        print(f'Stats will be saved in individual files in "{config.simulation_output_dir}", the next step is to merge them together')
    else:
        print("You can run a similar command to above yourself, if you need to change it")

if __name__ == "__main__":
    main()