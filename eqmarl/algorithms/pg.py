import tensorflow as tf
import tensorflow.keras as keras
from typing import Union
import numpy as np
import gymnasium as gym
from dataclasses import asdict

from .algorithm import Algorithm, Interaction

class PG(Algorithm):
    """Policy gradient using expected discounted returns.
    """

    def __init__(self,
        env: gym.Env, # Vectorized environment.
        model_policy: keras.Model, # One shared policy (each agent uses the same policy).
        optimizer_policy: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        gamma: float = 1., # Discount factor for returns, =1 means no discounting.
        episode_metrics_callback = None, # Called at the end of each episode to report metrics.
        ):
        super().__init__(env, episode_metrics_callback)
        assert isinstance(env.action_space, (gym.spaces.Discrete,)), "only `Discrete` action spaces are supported"
        self.model_policy = model_policy
        self.optimizer_policy = optimizer_policy
        self.gamma = gamma

    def policy(self, state) -> tuple[int, tf.Tensor]:
        """Interact with policy."""
        # Convert to tensor.
        s = tf.convert_to_tensor(state)
        s = tf.reshape(s, (1, *s.shape))
        
        # pi(ai | si)
        action_probs = self.model_policy(s)

        # Sample action from estimated probability distribution.
        action = np.random.choice(action_probs.shape[-1], p=np.squeeze(action_probs))

        return action, action_probs
    
    
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


    def update(self, batch: list[Interaction]):

        # Convert training batch to dictionary.
        batch = {k:[asdict(d)[k] for d in batch] for k in asdict(batch[0]).keys()}
        
        state_batched = np.array(batch['state']) #.squeeze()
        action_batched = np.array(batch['action']) #.squeeze()
        reward_batched = np.array(batch['reward'], dtype='float32') #.squeeze()
        next_state_batched = np.array(batch['next_state']) #.squeeze()
        
        state_batched = tf.convert_to_tensor(state_batched)
        action_batched = tf.convert_to_tensor(action_batched)
        reward_batched = tf.convert_to_tensor(reward_batched)
        next_state_batched = tf.convert_to_tensor(next_state_batched)

        # Estimate returns.
        returns = self.get_expected_returns(reward_batched, self.gamma)
        
        # Update the policy network.
        with tf.GradientTape() as tape:
            tape.watch(self.model_policy.trainable_variables)
            
            action_probs = self.model_policy(state_batched) # pi(a|s)

            # Convert batched individual actions back to a joint action.
            id_action_pairs = np.array([(i,a) for i, a in enumerate(action_batched)]) # (n_time_steps, 2,)
            probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
            action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)

            loss = -tf.reduce_mean(returns * action_probs_log)

        # Compute gradients for the actor network.
        grads = tape.gradient(loss, self.model_policy.trainable_variables)

        # Update actor network.
        if isinstance(self.optimizer_policy, (list, tuple)):
            for i in range(len(self.optimizer_policy)):
                self.optimizer_policy[i].apply_gradients([(grads[i], self.model_policy.trainable_variables[i])])
        else:
            self.optimizer_policy.apply_gradients(zip(grads, self.model_policy.trainable_variables))

    def run_episode(self, 
        episode: int, # Episode number.
        total_steps: int, # Total number of steps up until the start of this episode.
        max_steps_per_episode: int, # Maximum number of steps in this episode.
        ) -> tuple[Union[float, np.ndarray], list[Interaction], int]:
        """Runs a single episode.
        
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