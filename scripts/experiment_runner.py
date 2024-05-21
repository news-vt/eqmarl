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


def load_obj_from_dotpath(path: str):
    """Load object from within module. 
    
    The path should be `.` delimited and with fully specified package names (e.g., `numpy.sum`).
    """
    module, obj = path.rsplit(".", maxsplit=1)
    m = import_module(module)
    return getattr(m, obj)


def load_env(config: dict) -> gym.Env:
    env_func = load_obj_from_dotpath(config['func'])
    env_params = config['params']
    env = env_func(env_params)
    return env


def load_model(config: dict) -> keras.Model:
    init_func = load_obj_from_dotpath(config['init_func'])
    init_params = config['init_params']
    model: keras.Model = init_func(**init_params)
    model.build(config['build_shape'])
    return model


def load_optimizer(config: Union[dict,list[dict]]) -> Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]]:
    # List of optimizers, one for each trainable variable.
    if isinstance(config, list):
        optimizers: list[keras.optimizers.Optimizer] = []
        for opt_dict in config:
            opt_func = load_obj_from_dotpath(opt_dict['func'])
            optimizer = opt_func(**opt_dict['params'])
            optimizers.append(optimizer)
        return optimizers
    # One optimizer for the entire model.
    else:
        opt_dict = config
        opt_func = load_obj_from_dotpath(opt_dict['func'])
        optimizer: keras.optimizers.Optimizer = opt_func(**opt_dict['params'])
        return optimizer


def load_experiment(config: dict) -> dict:
    
    config_exp = config['experiment']
    roots = config_exp['roots']
    
    # Load the algorithm.
    config_algo = config['experiment']['algorithm']
    algo_init_func = load_obj_from_dotpath(config_algo['init_func'])
    algo_init_params = config_algo['init_params']
    if 'episode_metrics_callback' in algo_init_params:
        algo_init_params['episode_metrics_callback'] = load_obj_from_dotpath(algo_init_params['episode_metrics_callback'])

    
    # Environment.
    config_env = algo_init_params['env']
    env = load_env(config_env)
    algo_init_params['env'] = env # Overwrite the config.
    
    # Models.
    model_keys = [k for k in algo_init_params.keys() if 'model' in k]
    for key in model_keys:
        model_config = algo_init_params[key]
        model = load_model(model_config)
        algo_init_params[key] = model # Overwrite the config.

    # Optimizers.
    optimizer_keys = [k for k in algo_init_params.keys() if 'optimizer' in k]
    for key in optimizer_keys:
        optimizer_config = algo_init_params[key]
        optimizer = load_optimizer(optimizer_config)
        algo_init_params[key] = optimizer # Overwrite the config.
        
        
    algo = algo_init_func(**algo_init_params)
    
    # Load training parameters.
    train = config_exp['train']
    if 'callbacks' in train:
        cbs = []
        for cbd in train['callbacks']:
            cb_func = load_obj_from_dotpath(cbd['func'])
            cb_params = cbd['params']
            cb = cb_func(**cb_params)
            cbs.append(cb)
        train['callbacks'] = cbs # Overwrite the config.

    return dict(
        roots=roots,
        algorithm=algo,
        train=train,
        save=config_exp['save'],
    )





def main(config: str, n_train_rounds: int):

    # Time of training session start.
    datetime_session = datetime.now()
    print(f"Training session start at {datetime_session.isoformat()}")

    # Create a directory for this training session.
    session_dir = Path(config['experiment']['roots']['session_dir'].format(datetime_session=datetime_session))
    session_dir.expanduser().mkdir(parents=True, exist_ok=True)

    if n_train_rounds > 1:
        print(f'Training for {n_train_rounds} rounds')

    # Iteratively 
    for r in range(n_train_rounds):
        
        
        config_session = copy.deepcopy(config)
        exp = load_experiment(config_session)
        algo: eqmarl.Algorithm = exp['algorithm']
        train_params = exp['train']
        
        # Save some of the session and round details within the algorithm so that callbacks and other entities will have access to them.
        algo.datetime_session = datetime_session
        algo.round = r

        round_start = datetime.now()
        if n_train_rounds > 1:
            print(f'Training round {r} start: {round_start}')

        # Train models using algorithm.
        reward_history, metrics_history = algo.train(
            **train_params,
            )

        # Save results to file if a metrics file was provided.
        metrics_file = exp['save'].get('metrics_file', None)
        if metrics_file is not None:
            metrics_file = metrics_file.format(
                datetime_session=datetime_session,
                round=r,
            )
            algo.save_train_results(metrics_file, reward_history, metrics_history)
            print(f"Saved metrics file {metrics_file}")
        
        # Save models to file if filenames were provided.
        for d in exp['save'].get('model_files', []):
            model_file = d['filepath'].format(
                datetime_session=datetime_session,
                round=r,
            )
            algo.save_model(d['name'], model_file, d['save_weights_only'])
            print(f"Saved model file {model_file}")
        
        # Print the round ending time and elapsed time.
        if n_train_rounds > 1:
            round_end = datetime.now()
            print(f'Training round {r} end: {round_end}')
            print(f'Training round {r} elapsed: {round_end - round_start}')
            print()
    
    # Print the ending time and how much time has elapsed.
    datetime_session_end = datetime.now()
    print(f"Training session end at {datetime_session_end.isoformat()} (elapsed {datetime_session_end-datetime_session})")


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('config',
        help='Experiment config file in YAML format.',
        )
    parser.add_argument('-r', '--n-train-rounds',
        type=int,
        default=1,
        help='Number of times to perform training.',
        )
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # Get program options.
    opts = get_opts()
    
    # Load the YAML config file.
    print(f"Loading experiment: {opts.config}")
    config_path = Path(opts.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.load(f, Loader=eqmarl.yaml.ConfigLoader)

    # Run the experiment.
    main(
        config=config,
        n_train_rounds=opts.n_train_rounds,
    )