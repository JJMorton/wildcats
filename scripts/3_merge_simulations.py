import inference.config as config
import pandas as pd
import os.path as path

def main():

    df = pd.DataFrame()
    for i in range(config.num_simulations):
        filepath = path.join(config.simulation_output_dir, f'stats_{i}.csv')
        if path.exists(filepath):
            df = pd.concat([df, pd.read_csv(filepath, index_col='index')])
        else:
            print(f'File "{filepath}" does not exist, skipping.')

    df = df.dropna()
    df.to_csv(config.stats_file, index=True, index_label="index")
    print(f'{len(list(df.index))}/{config.num_simulations} simulations were successful')
    print(f'Saved summary stats to "{config.stats_file}". If this file is correct you can remove the files in "{config.simulation_output_dir}"')

if __name__ == "__main__":
    main()
