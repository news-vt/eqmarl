from typing import Union
import gymnasium as gym
from tqdm import trange
import numpy as np
import tensorflow as tf


class Algorithm:

    def __init__(self, env: gym.Env, episode_metrics_callback):
        super().__init__()
        self.env = env
        self.episode_metrics_callback = episode_metrics_callback


    def run_episode(self, 
        episode: int, # Episode number.
        total_steps: int, # Total number of steps up until the start of this episode.
        max_steps_per_episode: int, # Maximum number of steps in this episode.
        ) -> tuple[Union[float, np.ndarray], list, int]:
        """Runs a single episode.
        
        Returns a tuple of (episode_reward, interaction_history, n_steps).
        """
        raise NotImplementedError


    def train(self,
        n_episodes: int, # Number of episodes.
        max_steps_per_episode: int = 10000,
        ) -> dict[str, list]:
        
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