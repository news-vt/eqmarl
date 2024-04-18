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
    parser.add_argument('dir',
        help='Session directory containing the metrics files to visualize.',
        )
    args = parser.parse_args()
    return args



def plot_with_errorbar(ax, data, axis):
    
    score_mean = np.mean(data, axis=0) # (3000,)
    score_std = np.std(data, axis=0)# (3000,)
    x = np.arange(data.shape[-1])
    
    ax.plot(x, score_mean, 'b-', linewidth=0.2)
    ax.fill_between(x, score_mean - score_std, score_mean+score_std, color='b', alpha=0.2, linewidth=0.2)








def main(config: dict, dir: str):
    
    # session_dir = config['experiment']['session_dir']
    
    exp = experiment_runner.load_experiment(config)
    algo = exp['algorithm']
    
    print(dir)
    files = sorted([f for f in Path(dir).iterdir() if f.is_file()])
    print(files)
    
    
    
    session_reward_history = []
    session_metrics_history = []
    
    for f in files:
        reward_history, metrics_history = algo.load_train_results(str(f))
        
        session_reward_history.append(reward_history)
        session_metrics_history.append(metrics_history)
        # session_history.append((reward_history, metrics_history))
        
        # reward_history = np.array(reward_history)
        # score = np.sum(reward_history, axis=-1)
        
        # print(f"{reward_history.shape=}")
        # plt.plot(score, linewidth=.2)
        # plt.title('Score')
        # plt.show()
        
    session_reward_history = session_reward_history
    session_reward_history = np.array(session_reward_history)
    print(f"{session_reward_history.shape=}")

    # score = np.sum(session_reward_history, axis=-1)
    # print(f"{score.shape=}")
    
    
    # score_mean = np.mean(score, axis=0) # (3000,)
    # score_std = np.std(score, axis=0)# (3000,)
    # x = np.arange(score.shape[-1])
    
    # mosaic = [
    #     key for key in ['score'] + list(metrics_history.keys())
    # ]
    # mosaic = np.array(mosaic).reshape((3,2))
    
    config_plot = config['experiment']['plot']
    mosaic = config_plot['mosaic']
    figsize = config_plot['figsize']
    fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize)
    
    # plot_with_errorbar(axd['score'], score, axis=0)
    # axd['score'].set_title('score')
    
    
    df = pd.DataFrame(session_metrics_history)
    df_arr = np.array(df.values.tolist())
    for k in list(np.array(mosaic).reshape(-1)):
        i = list(df.columns).index(k)
        plot_with_errorbar(axd[k], df_arr[:,i,:], axis=0)
        
        if 'title' in config_plot['axes'][k]:
            title = config_plot['axes'][k]['title']
        else:
            title = k
        axd[k].set_title(title)
        
        if 'xlabel' in config_plot['axes'][k]:
            axd[k].set_xlabel(config_plot['axes'][k]['xlabel'])
        
        if 'ylabel' in config_plot['axes'][k]:
            axd[k].set_ylabel(config_plot['axes'][k]['ylabel'])

    # i = list(df.columns).index('undiscounted_reward')
    # diff = df_arr[:,i,:] - score
    # print(np.allclose(diff, np.zeros_like(diff)))

    # nl = np.array(nl)
    # nl.reshape((len(df), len(metrics_history.keys()), 3000))
    # print(df)
    # print(df_arr.shape)
    # for k in list(df.columns):
    #     print(k)
    #     print(df[k])
    #     print(df.values.ravel())
    #     break
    
    
    
    # plt.plot(x, score_mean, 'b-', linewidth=0.2)
    # plt.fill_between(x, score_mean - score_std, score_mean+score_std, color='b', alpha=0.2, linewidth=0.2)
    
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
        config=config,
        dir=opts.dir,
    )