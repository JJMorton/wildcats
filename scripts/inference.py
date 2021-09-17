# To run this, we need the following:
#  1. the pickled proposal (either the prior or a posterior)
#  2. the parameter sets in csv format
#  3. the corresponding summary statistics in csv format
#     (some may be missing if any simulations failed to run, this script deals with that)
#  4. the csv of the observation we made (a single set of summary statistics)

import inference.config as config
import inference.utils as utils
from sbi.inference import SNPE
import matplotlib.pyplot as plt
from inference.analysis import plot_samples_vs_prior
import os.path as path

def main():
    
    print("Importing data...")
    theta, x = utils.get_theta_x()
    x_o = utils.get_observation()
    prior = utils.get_prior()
    proposal = prior if config.proposal_pickle_file == config.prior_pickle_file else utils.get_proposal()

    theta, x = utils.remove_outside_prior(prior, theta, x)
    
    print(f'{theta.shape=}')
    print(f'{x.shape=}')
    print(f'{x_o.shape=}')
    print(f'{type(proposal)=}')
    print(f'{type(prior)=}')
    
    print("Running inference...")
    inference = SNPE(prior=prior, density_estimator='mdn')
    inference = inference.append_simulations(theta, x, proposal=proposal)
    density_estimator = inference.train(show_train_summary=True, use_combined_loss=True, training_batch_size=1000)
    posterior = inference.build_posterior(density_estimator, sample_with_mcmc=False)
    posterior.set_default_x(x_o)
    utils.save_posterior(posterior)
    
    print("Plotting posterior samples...")
    fig, _ = plot_samples_vs_prior(prior, posterior.sample((10_000,)), "posterior")
    plt.savefig(path.join(config.plotting_dir, utils.strip_filename(config.posterior_pickle_file) + '.jpg'))

if __name__ == "__main__":
    main()
