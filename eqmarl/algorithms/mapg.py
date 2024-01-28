import tensorflow as tf
import tensorflow.keras as keras
from typing import Union
import numpy as np
import gymnasium as gym

import itertools
from dataclasses import dataclass, asdict
from collections import deque

from tqdm import trange

from .algorithm import Algorithm




@dataclass
class VectorInteraction:
    """Vectorized environment interaction."""
    states: tf.Tensor
    actions: list[int]
    action_probs: tf.Tensor
    rewards: float
    next_states: tf.Tensor
    dones: list[bool]



class MAPG(Algorithm):
    """Multi-agent policy gradient with shared policy and expected discounted returns.
    """
    
    def __init__(self,
        env: gym.vector.VectorEnv, # Vectorized environment.
        model_policy: keras.Model, # One shared policy (each agent uses the same policy).
        optimizer_policy: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        # model_actor: keras.Model, # One shared policy (each agent uses the same policy).
        # model_critic: keras.Model, # One central critic.
        # model_critic_target: keras.Model, # One central critic.
        # optimizer_actor: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        # optimizer_critic: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        gamma: float,
        # steps_per_update: int, # Number of steps per model update.
        # steps_per_target_update: int = None, # Number of steps per target model update.
        # tau: float = None, # Rate at which to slowly update the target network, should be tau << 1.
        # max_memory_length: int = 10000,
        episode_metrics_callback = None, # Called at the end of each episode to report metrics.
        ):
        assert isinstance(env, gym.vector.VectorEnv), "only vectorized environments are supported (must be instance of `gym.vector.VectorEnv`)"
        assert isinstance(env.action_space, (gym.spaces.MultiDiscrete,)), "only `MultiDiscrete` action spaces are supported"
        self.env = env
        self.n_envs = self.env.num_envs
        self.episode_metrics_callback = episode_metrics_callback # cb(env)
        self.model_policy = model_policy
        self.optimizer_policy = optimizer_policy
        self.gamma = gamma

    def policy(self, states) -> tuple[list[int], list[tf.Tensor]]:
        """Get policy estimation for each agent individually."""
        joint_action, joint_action_probs = [], []
        for i, s in enumerate(states):
            # Convert to tensor.
            s = tf.convert_to_tensor(s)
            s = tf.reshape(s, (1, *s.shape))

            # pi(ai | si)
            action_probs = self.model_policy(s)

            # Sample action from estimated probability distribution.
            action = np.random.choice(action_probs.shape[-1], p=np.squeeze(action_probs))

            joint_action.append(action)
            joint_action_probs.append(action_probs)

        return joint_action, joint_action_probs

    @staticmethod
    def get_expected_returns(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True,
        ) -> tf.Tensor:
        """Computes expected discounted returns.
        
        The discount factor `gamma` denotes how much past rewards have an influence on the future. Setting =1 means no discounting.
        """
        n_rewards = rewards.shape[0]
        
        returns = []
        discounted_sum = 0
        rewards = rewards[::-1] # Reverse the rewards.
        for i in range(n_rewards):
            discounted_sum = rewards[i] + gamma * discounted_sum
            returns.append(discounted_sum)
        returns = tf.convert_to_tensor(np.array(returns[::-1], dtype='float32'))
        
        if standardize:
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - tf.math.reduce_mean(returns, axis=0)) / (tf.math.reduce_std(returns, axis=0) + eps)
        
        return returns


    def update(self, batch: list[VectorInteraction]):

        # Convert training batch to dictionary.
        batch = {k:[asdict(d)[k] for d in batch] for k in asdict(batch[0]).keys()}
        
        states_batched = np.array(batch['states']) #.squeeze()
        actions_batched = np.array(batch['actions']) #.squeeze()
        rewards_batched = np.array(batch['rewards'], dtype='float32') #.squeeze()
        next_states_batched = np.array(batch['next_states']) #.squeeze()
        
        states_batched = tf.convert_to_tensor(states_batched)
        actions_batched = tf.convert_to_tensor(actions_batched)
        rewards_batched = tf.convert_to_tensor(rewards_batched)
        next_states_batched = tf.convert_to_tensor(next_states_batched)

        # Estimate returns.
        returns = self.get_expected_returns(rewards_batched, self.gamma)
        
        
        # Update the policy network.
        with tf.GradientTape() as tape:
            tape.watch(self.model_policy.trainable_variables)
            
            # Estimate the policy for each actor individually.
            # agents_action_probs_log = []
            agents_action_probs_log = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            for i in range(self.n_envs):
                action_probs = self.model_policy(states_batched[:,i]) # pi(a|s)

                # Convert batched individual actions back to a joint action.
                id_action_pairs = np.array([(i,a) for i, a in enumerate(actions_batched[:,i])]) # (n_time_steps, 2,)
                probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
                action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)
                # agents_action_probs_log.append(action_probs_log)
                agents_action_probs_log = agents_action_probs_log.write(i, action_probs_log)
            # agents_action_probs_log = tf.stack(agents_action_probs_log, axis=-1)
            agents_action_probs_log = tf.transpose(agents_action_probs_log.stack())

            # Compute actor loss.
            loss = -tf.math.reduce_mean(returns * agents_action_probs_log)

        # Compute gradients for the actor network.
        grads = tape.gradient(loss, self.model_policy.trainable_variables)

        # Update actor network.
        if isinstance(self.optimizer_policy, (list, tuple)):
            for i in range(len(self.optimizer_policy)):
                self.optimizer_policy[i].apply_gradients([(grads[i], self.model_policy.trainable_variables[i])])
        else:
            self.optimizer_policy.apply_gradients(zip(grads, self.model_policy.trainable_variables))


    def train(self,
        n_episodes: int, # Number of episodes.
        max_steps_per_episode: int = 10000,
        ) -> dict[str, list]:
        
        print(f"Training for {n_episodes} episodes, press 'Ctrl+C' to terminate early")

        episode_reward_history = []
        episode_metrics_history = []
        steps = 0
        try:
            with trange(n_episodes, unit='episode') as tepisode:
                for episode in tepisode:
                    tepisode.set_description(f"Episode {episode}")
                    
                    episode_reward = 0
                    batch = []

                    # Reset environment.
                    states, _ = self.env.reset()
                    
                    # Iterate through environment at discrete time steps.
                    for t in range(max_steps_per_episode):
                        steps += 1

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

                    # Compute episode metrics.
                    episode_reward_history.append(episode_reward)
                    if self.episode_metrics_callback is not None:
                        episode_metrics_history.append(self.episode_metrics_callback(self.env))
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