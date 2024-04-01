from dataclasses import dataclass
from typing import Any, Union
import gymnasium as gym
from tqdm import trange
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime
from ..tools import NumpyJSONEncoder
from ..callbacks import Callback, CallbackList


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
        self._episode_reward_history = []
        self._episode_metrics_history = []
        
    @property
    def episode_reward_history(self):
        # TODO may need resource locking here if multi-threading training.
        return self._episode_reward_history

    @property
    def episode_metrics_history(self):
        # TODO may need resource locking here if multi-threading training.
        return self._episode_metrics_history

    @episode_reward_history.setter
    def episode_reward_history(self, episode_reward_history):
        # TODO may need resource locking here if multi-threading training.
        self._episode_reward_history = episode_reward_history

    @episode_metrics_history.setter
    def episode_metrics_history(self, episode_metrics_history):
        # TODO may need resource locking here if multi-threading training.
        self._episode_metrics_history = episode_metrics_history

    @property
    def episode_metrics_history_dict(self):
        # TODO may need resource locking here if multi-threading training.
        if self.episode_metrics_history:
            # Convert 'list of dicts' to 'dict of lists'.
            return {
                k: [d[k] for d in self.episode_metrics_history] 
                for k in self.episode_metrics_history[0].keys()
                }
        else:
            return {}

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
        callbacks: list[Callback] = [],
        ) -> tuple[np.ndarray, dict[str, Any]]:
        
        print(f"Training for {n_episodes} episodes, press 'Ctrl+C' to terminate early")
        
        # Convert callbacks to list subclass.
        callbacks: CallbackList = CallbackList(callbacks)
        
        # Set algorithm for all callbacks.
        callbacks.algorithm = self

        # Reset history.
        self.episode_reward_history = []
        self.episode_metrics_history = []
        total_steps = 0
        
        # Callback train begin.
        callbacks.on_train_begin()
        
        # Run training.
        try:
            with trange(n_episodes, unit='episode') as tepisode:
                for episode in tepisode:
                    tepisode.set_description(f"Episode {episode}")
                    
                    # Callback episode begin.
                    callbacks.on_episode_begin(episode)

                    # Run the episode.
                    episode_reward, _, episode_steps = self.run_episode(
                        episode=episode,
                        total_steps=total_steps,
                        max_steps_per_episode=max_steps_per_episode,
                        )

                    # Increment the total number of steps.
                    total_steps += episode_steps

                    # Compute episode metrics.
                    self.episode_reward_history.append(episode_reward)
                    if self.episode_metrics_callback is not None:
                        episode_metrics = self.episode_metrics_callback(self.env)
                        self.episode_metrics_history.append(episode_metrics)
                    else:
                        episode_metrics = {}

                    # Callback episode end.
                    callbacks.on_episode_end(episode)

                    tepisode.set_postfix(episode_reward=episode_reward, **episode_metrics)
                    tepisode.set_description(f"Episode {episode+1}") # Force next episode description.

        except KeyboardInterrupt:
            print(f"Terminating early at episode {episode}")

        # Callback train end.
        callbacks.on_train_end()

        return self.episode_reward_history, self.episode_metrics_history_dict

    def save_train_results(self, filepath: Union[str, Path]):
        """Saves training results from algorithm class instance to JSON file."""
        d = dict(
            reward=self.episode_reward_history,
            metrics=self.episode_metrics_history_dict,
        )
        filepath = str(filepath).format(
            datetime=datetime.now().isoformat(),
            )
        with open(filepath, 'w+') as f:
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
        self._episode_reward_history = []
        self._episode_metrics_history = []

    def run_episode(self, 
        episode: int, # Episode number.
        total_steps: int, # Total number of steps up until the start of this episode.
        max_steps_per_episode: int, # Maximum number of steps in this episode.
        ) -> tuple[Union[float, np.ndarray], list[VectorInteraction], int]:
        """Runs a single episode.
        
        Returns a tuple of (episode_reward, interaction_history, n_steps).
        """
        raise NotImplementedError