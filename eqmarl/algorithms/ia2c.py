import tensorflow as tf
import tensorflow.keras as keras
from typing import Union
import numpy as np
import gymnasium as gym
from dataclasses import asdict

from .algorithm import VectorAlgorithm, VectorInteraction


class IA2C(VectorAlgorithm):
    """Independent advantage actor-critic (IA2C) algorithm where each has their own actor state-value critic `V(s)`.
    
    Advantage function estimates Q-value using next-state value `V(s')`.
    
    Policy loss includes entropy.
    """

    def __init__(self,
        env: gym.vector.VectorEnv, # Vectorized environment.
        model_actors: list[keras.Model], # One shared policy (each agent uses the same policy).
        model_critics: list[keras.Model], # One central critic.
        optimizer_actor: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        optimizer_critic: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        gamma: float = 1., # Discount factor for returns, =1 means no discounting.
        alpha: float = 0.001, # Entropy coefficient.
        episode_metrics_callback = None, # Called at the end of each episode to report metrics.
        ):
        super().__init__(env, episode_metrics_callback)
        assert isinstance(env.action_space, (gym.spaces.MultiDiscrete,)), "only `MultiDiscrete` action spaces are supported"
        self.model_actors = model_actors
        self.model_critics = model_critics
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.gamma = gamma
        self.alpha = alpha

    def policy(self, states, batched: bool = False) -> tuple[list[int], list[tf.Tensor]]:
        """Individual policy. Get policy estimation for each agent individually."""
        joint_action, joint_action_probs = [], []
        for i, s in enumerate(states):
            # Convert to tensor.
            s = tf.convert_to_tensor(s)
            if not batched: 
                s = tf.reshape(s, (-1, *s.shape))

            # pi(ai | si)
            action_probs = self.model_actors[i](s)

            # Sample action from estimated probability distribution.
            action = np.random.choice(action_probs.shape[-1], p=np.squeeze(action_probs))

            joint_action.append(action)
            joint_action_probs.append(action_probs)

        return joint_action, joint_action_probs


    def update(self, batch: list[VectorInteraction]):

        # Convert training batch to dictionary.
        batch = {k:[asdict(d)[k] for d in batch] for k in asdict(batch[0]).keys()}
        
        states_batched = np.array(batch['states']) #.squeeze()
        actions_batched = np.array(batch['actions']) #.squeeze()
        rewards_batched = np.array(batch['rewards'], dtype='float32') #.squeeze()
        next_states_batched = np.array(batch['next_states']) #.squeeze()
        dones_batched = np.array(batch['dones'], dtype='float32') #.squeeze()
        
        states_batched = tf.convert_to_tensor(states_batched)
        actions_batched = tf.convert_to_tensor(actions_batched)
        rewards_batched = tf.convert_to_tensor(rewards_batched)
        next_states_batched = tf.convert_to_tensor(next_states_batched)
        dones_batched = tf.convert_to_tensor(dones_batched)
        
        huber_loss = tf.keras.losses.Huber(reduction=keras.losses.Reduction.SUM)
        
        # Iterate over all agents separately.
        # TODO: This could be parallelized.
        for k in range(self.n_envs):
            
            # Isolate inputs for the current agent.
            state_batched = states_batched[:,k]
            next_state_batched = next_states_batched[:,k]
            action_batched = actions_batched[:,k]
            done_batched = dones_batched[:,k]
            reward_batched = rewards_batched[:,k]
            
            ma = self.model_actors[k]
            mc = self.model_critics[k]

            # Update the critic and actor networks simultaneously.
            with tf.GradientTape() as tape_critic, tf.GradientTape() as tape_actor:
                
                tape_critic.watch(mc.trainable_variables)
                tape_actor.watch(ma.trainable_variables)
                
                state_values = mc(state_batched) # V(s)
                next_state_values = mc(next_state_batched) # V(s')
                action_probs = ma(state_batched) # pi(a|s)
                
                # Convert batched individual actions back to a joint action.
                id_action_pairs = np.array([(i,a) for i, a in enumerate(action_batched)]) # (n_time_steps, 2,)
                probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
                action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)
                
                # Ensure shapes are correct.
                reward_batched = tf.reshape(reward_batched, shape=state_values.shape)
                done_batched = tf.reshape(done_batched, shape=state_values.shape)
                action_probs_log = tf.reshape(action_probs_log, shape=state_values.shape)
                
                # Compute advantage via Q-value estimation.
                # Q(s, a) = r(s',a) + gamma * V(s')
                q_values = reward_batched + (1. - done_batched) * self.gamma * next_state_values
                advantage = q_values - state_values # A(s, a) = Q(s, a) - V(s)
                
                # Entropy to regularize actor loss.
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=-1)
                entropy = tf.reduce_mean(entropy)

                actor_loss = tf.reduce_mean(-action_probs_log * advantage) + self.alpha * entropy
                critic_loss = huber_loss(state_values, q_values) # Huber loss of advantage.

            # Compute gradients for the actor network.
            grads_actor = tape_actor.gradient(actor_loss, ma.trainable_variables)
            grads_critic = tape_critic.gradient(critic_loss, mc.trainable_variables)

            # Update actor network.
            if isinstance(self.optimizer_actor, (list, tuple)):
                for i in range(len(self.optimizer_actor)):
                    self.optimizer_actor[i].apply_gradients([(grads_actor[i], ma.trainable_variables[i])])
            else:
                self.optimizer_actor.apply_gradients(zip(grads_actor, ma.trainable_variables))

            # Update critic network.
            if isinstance(self.optimizer_critic, (list, tuple)):
                for i in range(len(self.optimizer_critic)):
                    self.optimizer_critic[i].apply_gradients([(grads_critic[i], mc.trainable_variables[i])])
            else:
                self.optimizer_critic.apply_gradients(zip(grads_critic, mc.trainable_variables))


    def run_episode(self, 
        episode: int,
        total_steps: int,
        max_steps_per_episode: int,
        ) -> tuple[Union[float, np.ndarray], list[VectorInteraction], int]:
        """Runs a single episode.
        
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