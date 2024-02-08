from dataclasses import dataclass
from typing import Any, Union
import gymnasium as gym
from tqdm import trange
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from ..tools import NumpyJSONEncoder


@dataclass
class Interaction:
    """Environment interaction."""
    state: tf.Tensor
    action: int
    action_probs: tf.Tensor
    reward: float
    next_state: tf.Tensor
    done: bool


@dataclass
class VectorInteraction:
    """Vectorized environment interaction."""
    states: tf.Tensor
    actions: list[int]
    action_probs: tf.Tensor
    rewards: float
    next_states: tf.Tensor
    dones: list[bool]


class Algorithm:
    """Reinforcement learning algorithm base class for use with `gym.Env` environments."""

    def __init__(self, env: gym.Env, episode_metrics_callback):
        assert isinstance(env, gym.Env), "only gymnasium environments are supported (must be instance of `gym.Env`)"
        self.env = env
        self.episode_metrics_callback = episode_metrics_callback


    def run_episode(self, 
        episode: int, # Episode number.
        total_steps: int, # Total number of steps up until the start of this episode.
        max_steps_per_episode: int, # Maximum number of steps in this episode.
        ) -> tuple[Union[float, np.ndarray], list[Interaction], int]:
        """Runs a single episode.
        
        Returns a tuple of (episode_reward, interaction_history, n_steps).
        """
        raise NotImplementedError


    def train(self,
        n_episodes: int, # Number of episodes.
        max_steps_per_episode: int = 10000,
        ) -> tuple[np.ndarray, dict[str, Any]]:
        
        print(f"Training for {n_episodes} episodes, press 'Ctrl+C' to terminate early")

        episode_reward_history = []
        episode_metrics_history = []
        total_steps = 0
        try:
            with trange(n_episodes, unit='episode') as tepisode:
                for episode in tepisode:
                    tepisode.set_description(f"Episode {episode}")

                    # Run the episode.
                    episode_reward, _, episode_steps = self.run_episode(
                        episode=episode,
                        total_steps=total_steps,
                        max_steps_per_episode=max_steps_per_episode,
                        )

                    # Increment the total number of steps.
                    total_steps += episode_steps

                    # Compute episode metrics.
                    episode_reward_history.append(episode_reward)
                    if self.episode_metrics_callback is not None:
                        episode_metrics = self.episode_metrics_callback(self.env)
                        episode_metrics_history.append(episode_metrics)
                    else:
                        episode_metrics = {}

                    tepisode.set_postfix(episode_reward=episode_reward, **episode_metrics)
                    tepisode.set_description(f"Episode {episode+1}") # Force next episode description.

        except KeyboardInterrupt:
            print(f"Terminating early at episode {episode}")
        
        # Convert 'list of dicts' to 'dict of lists'.
        if episode_metrics_history:
            episode_metrics_history = {k:[d[k] for d in episode_metrics_history] for k in episode_metrics_history[0].keys()}
        else:
            episode_metrics_history = {}
        
        return episode_reward_history, episode_metrics_history

    @staticmethod
    def save_train_results(filepath: Union[str, Path], reward_history: np.ndarray, metrics_history: dict[str, Any]):
        """Saves training results to JSON file."""
        d = dict(
            reward=reward_history,
            metrics=metrics_history,
        )
        with open(str(filepath), 'w+') as f:
            json.dump(d, f, cls=NumpyJSONEncoder)

    @staticmethod
    def load_train_results(filepath: Union[str, Path]) -> tuple[list, dict[str, Any]]:
        """Loads training results from JSON file."""
        with open(str(filepath), 'r') as f:
            d = json.load(f)
        return d['reward'], d['metrics']



class VectorAlgorithm(Algorithm):
    """Vectorized reinforcement learning algorithm base class for use with `gym.vector.VectorEnv` environments."""

    def __init__(self, env: gym.vector.VectorEnv, episode_metrics_callback):
        assert isinstance(env, gym.vector.VectorEnv), "only vectorized environments are supported (must be instance of `gym.vector.VectorEnv`)"
        self.env = env
        self.episode_metrics_callback = episode_metrics_callback
        self.n_envs = self.env.num_envs

    def run_episode(self, 
        episode: int, # Episode number.
        total_steps: int, # Total number of steps up until the start of this episode.
        max_steps_per_episode: int, # Maximum number of steps in this episode.
        ) -> tuple[Union[float, np.ndarray], list[VectorInteraction], int]:
        """Runs a single episode.
        
        Returns a tuple of (episode_reward, interaction_history, n_steps).
        """
        raise NotImplementedError