import tensorflow as tf
import tensorflow.keras as keras
from typing import Union
import numpy as np
import gymnasium as gym
from dataclasses import dataclass, asdict
from tqdm import trange

from .algorithm import Algorithm




@dataclass
class Interaction:
    """Environment interaction."""
    state: tf.Tensor
    action: int
    action_probs: tf.Tensor
    reward: float
    next_state: tf.Tensor
    done: bool



class A2C(Algorithm):
    """Advantage actor-critic (A2C) algorithm using a state-value critic `V(s)`.
    
    Advantage function estimates Q-value using next-state value `V(s')`.
    
    Policy loss includes entropy.
    """

    def __init__(self,
        env: gym.Env, # Standard gymnasium environment.
        model_actor: keras.Model, # One shared policy (each agent uses the same policy).
        model_critic: keras.Model, # One central critic.
        optimizer_actor: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        optimizer_critic: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        gamma: float = 1., # Discount factor for returns, =1 means no discounting.
        alpha: float = 0.001, # Entropy coefficient.
        episode_metrics_callback = None, # Called at the end of each episode to report metrics.
        ):
        assert isinstance(env, gym.Env), "only gymnasium environments are supported (must be instance of `gym.Env`)"
        assert isinstance(env.action_space, (gym.spaces.Discrete,)), "only `Discrete` action spaces are supported"
        self.env = env
        self.episode_metrics_callback = episode_metrics_callback # cb(env)
        self.model_actor = model_actor
        self.model_critic = model_critic
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.gamma = gamma
        self.alpha = alpha

    def policy(self, state) -> tuple[int, tf.Tensor]:
        """Interact with policy."""
        # Convert to tensor.
        s = tf.convert_to_tensor(state)
        s = tf.reshape(s, (1, *s.shape))
        
        # pi(ai | si)
        action_probs = self.model_actor(s)

        # Sample action from estimated probability distribution.
        action = np.random.choice(action_probs.shape[-1], p=np.squeeze(action_probs))

        return action, action_probs

    def values(self, state) -> list[tf.Tensor]:
        """Get value estimate at a given state `V(s)`."""
        # Convert to tensor.
        s = tf.convert_to_tensor(state)
        s = tf.reshape(s, (1, *s.shape))

        # V(s)
        state_values = self.model_critic(s)

        return state_values
    
    @staticmethod
    def get_expected_returns(
        rewards: tf.Tensor,
        gamma: float,
        initial_value: float = 0., # Starting value for discounted sum.
        standardize: bool = True,
        ) -> tf.Tensor:
        """Computes expected discounted returns.
        
        The discount factor `gamma` denotes how much past rewards have an influence on the future. Setting =1 means no discounting.
        """
        n_rewards = rewards.shape[0]
        discounted_sum = initial_value
        returns = np.zeros_like(rewards, dtype='float32')
        for t in reversed(range(n_rewards)):
            discounted_sum = rewards[t] + gamma * discounted_sum
            returns[t] = discounted_sum
        returns = tf.convert_to_tensor(returns, dtype='float32')
        
        # print(returns)
        
        if standardize:
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - tf.math.reduce_mean(returns, axis=0)) / (tf.math.reduce_std(returns, axis=0) + eps)
            
        # print(returns)
        # print(x)
        
        return returns

    def update(self, batch: list[Interaction]):

        # Convert training batch to dictionary.
        batch = {k:[asdict(d)[k] for d in batch] for k in asdict(batch[0]).keys()}
        
        state_batched = np.array(batch['state']) #.squeeze()
        action_batched = np.array(batch['action']) #.squeeze()
        reward_batched = np.array(batch['reward'], dtype='float32') #.squeeze()
        next_state_batched = np.array(batch['next_state']) #.squeeze()
        done_batched = np.array(batch['done'], dtype='float32') #.squeeze()
        
        state_batched = tf.convert_to_tensor(state_batched)
        action_batched = tf.convert_to_tensor(action_batched)
        reward_batched = tf.convert_to_tensor(reward_batched)
        next_state_batched = tf.convert_to_tensor(next_state_batched)
        done_batched = tf.convert_to_tensor(done_batched)
        
        huber_loss = tf.keras.losses.Huber(reduction=keras.losses.Reduction.SUM)

        # Update the critic and actor networks simultaneously.
        with tf.GradientTape() as tape_critic, tf.GradientTape() as tape_actor:
            tape_critic.watch(self.model_critic.trainable_variables)
            tape_actor.watch(self.model_actor.trainable_variables)
            
            state_values = self.model_critic(state_batched) # V(s)
            next_state_values = self.model_critic(next_state_batched) # V(s')
            action_probs = self.model_actor(state_batched) # pi(a|s)
            
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
        grads_actor = tape_actor.gradient(actor_loss, self.model_actor.trainable_variables)
        grads_critic = tape_critic.gradient(critic_loss, self.model_critic.trainable_variables)

        # Update actor network.
        if isinstance(self.optimizer_actor, (list, tuple)):
            for i in range(len(self.optimizer_actor)):
                self.optimizer_actor[i].apply_gradients([(grads_actor[i], self.model_actor.trainable_variables[i])])
        else:
            self.optimizer_actor.apply_gradients(zip(grads_actor, self.model_actor.trainable_variables))

        # Update critic network.
        if isinstance(self.optimizer_critic, (list, tuple)):
            for i in range(len(self.optimizer_critic)):
                self.optimizer_critic[i].apply_gradients([(grads_critic[i], self.model_critic.trainable_variables[i])])
        else:
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.model_critic.trainable_variables))


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
                    state, _ = self.env.reset()
                    
                    # Iterate through environment at discrete time steps.
                    for t in range(max_steps_per_episode):
                        steps += 1

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