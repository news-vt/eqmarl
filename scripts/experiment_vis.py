import sys
sys.path.append('../') # Use parent dir.

import eqmarl
import tensorflow.keras as keras
from pathlib import Path
from datetime import datetime
import yaml
from importlib import import_module
import gymnasium as gym
from typing import Union
import argparse
import copy
import experiment_runner

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# sns.set()
sns.set_theme()

# import os


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('config',
        help='Experiment config file in YAML format.',
        )
    # parser.add_argument('dir',
    #     help='Session directory containing the metrics files to visualize.',
    #     )
    parser.add_argument('files',
        nargs='+',
        help='Metrics files to visualize.',
        )
    parser.add_argument('-s', '--sort',
        action='store_true',
        help='Performs a lexical sort of the filepaths prior to processing.',
        )
    args = parser.parse_args()
    return args


# method in ['std', 'minmax']
def plot_with_errorbar(ax, data, axis, error_method: str = 'std', plot_data: str = 'mean'):
    
    ###
    # Method to highlight the main plot data.
    ###
    y = data
    
    # Plots the average run value at each epoch.
    if plot_data == 'mean':
        y = np.mean(data, axis=0) # (3000,)
    
    # Plots the run with the maximum average value over the last `n` epochs.
    elif plot_data.startswith('max-'):
        n = int(plot_data.split('-')[-1])
        avg_last_n = np.mean(data[:,-n:], axis=1)
        idx_max = np.argmax(avg_last_n)
        y = data[idx_max] # Plot data is the index of the max.
        
    # Plots the run with the minimum average value over the last `n` epochs.
    elif plot_data.startswith('min-'):
        n = int(plot_data.split('-')[-1])
        avg_last_n = np.mean(data[:,-n:], axis=1)
        idx_min = np.argmin(avg_last_n)
        y = data[idx_min] # Plot data is the index of the min.
    
    else:
        raise ValueError(f"Unsupported plot highlight {plot_data}")
    
    ###
    # Method to produce the shaded error regions.
    ###
    
    # Shaded region is +/- standard deviation from the designated `y`-value.
    if error_method.startswith('std'):
        y_std = np.std(data, axis=0)# (3000,)
        n = 1 # Default is 1 std above/below the data.
        if '-' in error_method: # Pull `n` value from method type.
            n = int(error_method.split('-')[-1])
        x = np.arange(data.shape[-1])
        ax.plot(x, y, 'b-', linewidth=0.2)
        ax.fill_between(x, y - y_std, y+y_std, color='b', alpha=0.2, linewidth=0.2)
    
    # Shaded region is minimum/maximum values at each epoch.
    elif error_method == 'minmax':
        y_min = np.min(data, axis=0) # (3000,)
        y_max = np.max(data, axis=0)# (3000,)
        x = np.arange(data.shape[-1])
        ax.plot(x, y, 'b-', linewidth=0.2)
        ax.fill_between(x, y_min, y_max, color='b', alpha=0.2, linewidth=0.2)
    
    else:
        raise ValueError(f"Unsupported error method {error_method}")








def main(name, config: dict, files: list[str], flag_sort_files: bool):
    
    # session_dir = config['experiment']['session_dir']
    
    exp = experiment_runner.load_experiment(config)
    algo = exp['algorithm']
    
    # Collect filepaths and sort if necessary.
    files = [Path(f) for f in files]
    if flag_sort_files:
        files = sorted(files)
    
    
    # Load metric data.
    session_reward_history = []
    session_metrics_history = []
    for f in files:
        reward_history, metrics_history = algo.load_train_results(str(f))
        session_reward_history.append(reward_history)
        session_metrics_history.append(metrics_history)

    # Reshape to proper matrix.
    session_reward_history = session_reward_history
    session_reward_history = np.array(session_reward_history)

    # Create figure and axes using info from config file.
    config_plot = config['experiment']['plot']
    mosaic = config_plot['mosaic']
    figsize = config_plot['figsize']
    fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize)
    
    fig.suptitle(name)

    # Plot metrics designated in config file.
    df = pd.DataFrame(session_metrics_history)
    df_arr = np.array(df.values.tolist())
    for k in list(np.array(mosaic).reshape(-1)):
        i = list(df.columns).index(k)
        # plot_with_errorbar(axd[k], df_arr[:,i,:], axis=0, method='minmax')
        # plot_with_errorbar(axd[k], df_arr[:,i,:], axis=0, method='minmax', plot_highlight='max-100')
        plot_with_errorbar(axd[k], df_arr[:,i,:], axis=0, **config['experiment']['plot'].get('plotargs', {}))
        # plot_with_errorbar(axd[k], df_arr[:,i,:], axis=0, method='std', plot_highlight='max-100')
        # plot_with_errorbar(axd[k], df_arr[:,i,:], axis=0, method='std')
        
        if 'title' in config_plot['axes'][k]:
            title = config_plot['axes'][k]['title']
        else:
            title = k
        axd[k].set_title(title)
        
        if 'xlabel' in config_plot['axes'][k]:
            axd[k].set_xlabel(config_plot['axes'][k]['xlabel'])
        
        if 'ylabel' in config_plot['axes'][k]:
            axd[k].set_ylabel(config_plot['axes'][k]['ylabel'])

    fig.tight_layout()
    plt.show()

    
    
    
    
    
    
    # score_data = dict([(f"Round{r}_score", score[r]) for r in range(len(files[:5]))])
    
    # df_score = pd.DataFrame(score_data)
    # # df_score = pd.DataFrame(score)
    # # df = pd.DataFrame(session_reward_history)
    # print(df_score)
    
    # sns.lineplot(data=df_score, errorbar='sd')
    # plt.show()


if __name__ == '__main__':

    # Get program options.
    opts = get_opts()
    
    # Load the YAML config file.
    config_path = Path(opts.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.load(f, Loader=eqmarl.yaml.ConfigLoader)

    # Run the experiment.
    main(
        name=config_path.name,
        config=config,
        files=opts.files,
        flag_sort_files=opts.sort
    )