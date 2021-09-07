# Change this configuration file accordingly before running each round of inference
# Check it by running 1_check_config.py

# The proposal to use for sampling, pickled using pickle.dump()
# Any distribution with .sample() and .log_prob() should work, as long as both methods return a tensor of the correct shape.
# For the first round, this should be the prior. For subsequent rounds, this should be the posterior of the previous round.
proposal_pickle_file = "../data/inference/prior.pkl"

# The file to save the proposal's samples to, which will be used as the sets of parameters for the simulation stage.
parameters_file = "../data/inference/theta0.csv"

# The file to save the summary statistics from the simulations to.
stats_file = "../data/inference/stats0.csv"

# File containing one set of summary statistics that act as our observations
observation_file = "../data/inference/observation.csv"

# The number of simulations to run. If some fail (from timing out etc..), there will be fewer simulations than this used for inference.
num_simulations = 500

# The directory in which all the summary stats will be output to when running the simulations, and before merging them into one file
simulation_output_dir = "../output/inference/stats"
