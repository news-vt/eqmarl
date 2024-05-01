from dataclasses import dataclass
from typing import Any, Union
import gymnasium as gym
from tqdm import trange
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import sys
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
        self._models = {}

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

    @property
    def models(self) -> dict[str, keras.Model]:
        return self._models

    @models.setter
    def models(self, models: dict[str, keras.Model]):
        self._models = models


    def policy(self, state) -> tuple[int, tf.Tensor]:
        """Gets policy estimation for an input state or batched input states."""
        raise NotImplementedError


    def update(self, batch: list[Interaction]):
        """Update trained models using a batch of interactions."""
        raise NotImplementedError


    def run_episode(self, 
        episode: int, # Episode number.
        total_steps: int, # Total number of steps up until the start of this episode.
        max_steps_per_episode: int, # Maximum number of steps in this episode.
        ) -> tuple[Union[float, np.ndarray], list[Interaction], int]:
        """Runs a single episode for a standard gymnasium environment.
        
        Returns a tuple of (episode_reward, interaction_history, n_steps).
        """
        episode_reward = 0
        batch = []

        # Reset environment.
        state, _ = self.env.reset()
        
        # Iterate through environment at discrete time steps.
        for t in range(max_steps_per_episode):

            # Get policy estimation for current state.
            action, action_probs = self.policy(state)

            # Interact with environment.
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Preserve interaction.
            interaction = Interaction(
                state=state,
                action=action,
                action_probs=action_probs,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            batch.append(interaction)
            
            # Set next state.
            state = next_state

            # Modify episode reward.
            episode_reward += reward

            # Terminate episode.
            if done or truncated:
                break
        
        # Update the model after each episode.
        self.update(batch)
        
        return episode_reward, batch, t


    def train(self,
        n_episodes: int, # Number of episodes.
        max_steps_per_episode: int = 10000,
        callbacks: list[Callback] = [],
        tqdm_kwargs: dict = {},
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
            with trange(n_episodes, unit='episode', file=sys.stdout, **tqdm_kwargs) as tepisode:
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

    def save(self, filepath: Union[str, Path]):
        """Saves training results from algorithm class instance to JSON file."""
        self.save_train_results(
            filepath=filepath, 
            reward_history=self.episode_reward_history, 
            metrics_history=self.episode_metrics_history_dict,
            )

    def save_model(self, model_name: str, filepath: str, save_weights_only: bool = True):
        """Saves the model with the given name."""
        if save_weights_only:
            self.models[model_name].save_weights(filepath)
        else:
            self.models[model_name].save(filepath)

    @staticmethod
    def save_train_results(filepath: Union[str, Path], reward_history: np.ndarray, metrics_history: dict[str, Any]):
        """Saves training results to JSON file."""
        d = dict(
            reward=reward_history,
            metrics=metrics_history,
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
        self._models = {}


    def policy(self, states, batched: bool = False) -> tuple[list[int], list[tf.Tensor]]:
        """Gets vectorized policy estimation for an input state or batched input states."""
        raise NotImplementedError


    def update(self, batch: list[VectorInteraction]):
        """Update trained models using a batch of vectorized interactions."""
        raise NotImplementedError


    def run_episode(self, 
        episode: int,
        total_steps: int,
        max_steps_per_episode: int,
        ) -> tuple[Union[float, np.ndarray], list[VectorInteraction], int]:
        """Runs a single episode for a vectorized gymnasium environment.
        
        Returns a tuple of (episode_reward, interaction_history, n_steps).
        """

        episode_reward = 0
        batch = []

        # Reset environment.
        states, _ = self.env.reset()
        
        # Iterate through environment at discrete time steps.
        for t in range(max_steps_per_episode):

            # Get the joint action.
            actions, action_probs = self.policy(states)

            # Step through environment using joint action.
            next_states, rewards, dones, truncated, infos = self.env.step(actions)
            
            # Preserve interaction.
            interaction = VectorInteraction(
                states=states,
                actions=actions,
                action_probs=action_probs,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
            )
            batch.append(interaction)
            
            # Set next state.
            states = next_states

            # Modify episode reward.
            episode_reward += rewards

            # Terminate episode.
            if any(dones) or any(truncated):
                break
            
        # Update the model after each episode.
        self.update(batch)

        return episode_reward, batch, t