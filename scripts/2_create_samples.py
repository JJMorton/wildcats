# Should create file containing something like this:
#  index  bottleneck_strength_domestic  bottleneck_strength_wild  ...  pop_size_captive  pop_size_domestic_1  pop_size_domestic_2
#      0                    18377.9410                 2801.7227  ...        104.878494            7336.4956            9961.9120
#      1                     3661.8000                 9797.0740  ...        165.436390            6597.9395           12118.7300
#      2                    22694.6170                13539.2810  ...         57.784970           10612.2080            9620.7060
#      3                     3396.4937                 5320.5234  ...         62.308270            4415.3330            6757.4863
#      4                    18085.3960                10101.7300  ...         56.440113            4692.2593            9362.4280
#    ...                           ...                       ...  ...               ...                  ...                  ...
#    495                     7190.5996                 9184.7300  ...        117.080376            4437.2840            6699.8730
#    496                    13704.6260                14958.6060  ...        115.939180            3534.2222            9895.4090
#    497                     6800.1597                49753.3870  ...        128.209230            4710.8160            7421.8060
#    498                    37353.0040                11004.2660  ...         75.999504            4811.1590           11575.1640
#    499                    24142.9280                15831.2780  ...        163.544650            7827.3237            9794.4375
# [500 rows x 17 columns]

import inference.config as config
import inference.priors
import pickle
import pandas as pd
import numpy as np
from os.path import exists

def main():
    
    if exists(config.parameters_file):
        print(f'Parameters file "{config.parameters_file}" already exists, aborting.')
        exit(1)
    
    with open(config.proposal_pickle_file, 'rb') as f:
        proposal = pickle.load(f)

    samples = proposal.sample((config.num_simulations,))
    param_names = inference.priors.get_param_names()
    pd.DataFrame(np.array(samples), columns=param_names).to_csv(config.parameters_file, index=True, index_label="index")
    print(f'Saved proposal samples to "{config.parameters_file}"')


if __name__ == "__main__":
    main()