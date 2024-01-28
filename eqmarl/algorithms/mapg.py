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
    
    
    
    
    



############################################






class MAPG_shared_old(Algorithm):
    """Multi-agent policy gradient with shared policy.
    
    This algorithm assumes the all agents uses the same policy via parameter sharing.
    
    Citation: Yun et al. 2023, https://arxiv.org/abs/2301.04012
    """
    
    def __init__(self,
        env: gym.vector.VectorEnv, # Vectorized environment.
        model_actor: keras.Model, # One shared policy (each agent uses the same policy).
        model_critic: keras.Model, # One central critic.
        model_critic_target: keras.Model, # One central critic.
        optimizer_actor: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        optimizer_critic: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        gamma: float,
        steps_per_update: int, # Number of steps per model update.
        steps_per_target_update: int = None, # Number of steps per target model update.
        tau: float = None, # Rate at which to slowly update the target network, should be tau << 1.
        max_memory_length: int = 10000,
        episode_metrics_callback = None, # Called at the end of each episode to report metrics.
        ):
        assert isinstance(env, gym.vector.VectorEnv), "only vectorized environments are supported (must be instance of `gym.vector.VectorEnv`)"
        assert isinstance(env.action_space, (gym.spaces.MultiDiscrete,)), "only `MultiDiscrete` action spaces are supported"
        self.env = env
        self.n_envs = self.env.num_envs
        
        self.episode_metrics_callback = episode_metrics_callback # cb(env)

        self.model_actor = model_actor
        self.model_critic = model_critic
        self.model_critic_target = model_critic_target
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        
        self.gamma = gamma
        # self.steps_per_update = steps_per_update
        self.tau = tau
        # self.steps_per_target_update = steps_per_target_update
        # assert tau is not None or steps_per_target_update is not None, 'must provide one of `tau` or `steps_per_target_update`'
        
        # Initialize replay buffer.
        self.max_memory_length = max_memory_length
        self.replay_buffer: list[VectorInteraction] = deque(maxlen=max_memory_length)

        # Initialize the target critic network.
        self.model_critic_target.set_weights(self.model_critic.get_weights())

    def policy(self, states) -> tuple[list[int], list[tf.Tensor]]:
        """Get policy estimation for each agent individually."""
        joint_action, joint_action_probs = [], []
        for i, s in enumerate(states):
            # Convert to tensor.
            s = tf.convert_to_tensor(s)
            s = tf.reshape(s, (1, *s.shape))
            
            # pi(ai | si)
            action_probs = self.model_actor(s)
            # print(f"{action_probs.numpy()=}")
            # print(f"{s.numpy()=}")

            # Sample action from estimated probability distribution.
            action = np.random.choice(action_probs.shape[-1], p=np.squeeze(action_probs))

            joint_action.append(action)
            joint_action_probs.append(action_probs)

        return joint_action, joint_action_probs

    def values(self, states) -> list[tf.Tensor]:
        """Get value from joint state."""
        # Convert to tensor.
        s = tf.convert_to_tensor(states)
        s = tf.reshape(s, (1, *s.shape))

        # V({s0, s1, ...})
        joint_state_values = self.model_critic(s)

        return joint_state_values

    def update_target(self):
        # Using tau.
        critic_target_weights = self.model_critic_target.get_weights()
        critic_weights = self.model_critic.get_weights()
        self.model_critic_target.set_weights([self.tau * critic_weights[i] + (1. - self.tau) * critic_target_weights[i] for i in range(len(critic_weights))])

        # # Direct update.
        # self.model_critic_target.set_weights(self.model_critic.get_weights())


    def update(self, batch_size: int):

        # Randomly select interactions from replay memory and train on them.
        batch_size = min(len(self.replay_buffer), batch_size) # Ensure we do not take more than the replay buffer currently has available.
        training_batch_list: list[VectorInteraction] = np.random.choice(self.replay_buffer, size=batch_size)
        # Convert training batch to dictionary.
        training_batch_dict = {k:[asdict(d)[k] for d in training_batch_list] for k in asdict(training_batch_list[0]).keys()}
        
        states_batched = np.array(training_batch_dict['states']) #.squeeze()
        actions_batched = np.array(training_batch_dict['actions']) #.squeeze()
        # action_probs_batched = np.array(training_batch_dict['action_probs']) #.squeeze()
        rewards_batched = np.array(training_batch_dict['rewards'], dtype='float32') #.squeeze()
        next_states_batched = np.array(training_batch_dict['next_states']) #.squeeze()
        # dones_batched = np.array(training_batch_dict['dones']) #.squeeze()
        
        states_batched = tf.convert_to_tensor(states_batched)
        actions_batched = tf.convert_to_tensor(actions_batched)
        # action_probs_batched = tf.convert_to_tensor(action_probs_batched)
        rewards_batched = tf.convert_to_tensor(rewards_batched)
        next_states_batched = tf.convert_to_tensor(next_states_batched)
        # dones_batched = tf.convert_to_tensor(dones_batched)

        # Total reward is the sum of rewards across all agents at each time step.
        rewards_batched = tf.math.reduce_mean(rewards_batched, axis=-1, keepdims=True)

        # Update the critic network using the replay batch.
        with tf.GradientTape() as tape_critic, tf.GradientTape() as tape_actor:
            tape_critic.watch(self.model_critic.trainable_variables)
            tape_actor.watch(self.model_actor.trainable_variables)

            # Compute critic loss.
            state_values = self.model_critic(states_batched) # V(s)
            next_state_values = self.model_critic_target(next_states_batched) # V(s')
            y = rewards_batched + self.gamma * next_state_values - state_values # yt
            critic_loss = tf.math.reduce_mean(tf.math.square(y)) # Critic loss is |y|^2

            # Estimate the policy for each actor individually.
            # agents_action_probs_log = []
            agents_action_probs_log = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            for i in range(self.n_envs):
                action_probs = self.model_actor(states_batched[:,i]) # pi(a|s)

                # Convert batched individual actions back to a joint action.
                id_action_pairs = np.array([(i,a) for i, a in enumerate(actions_batched[:,i])]) # (n_time_steps, 2,)
                probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
                action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)
                # agents_action_probs_log.append(action_probs_log)
                agents_action_probs_log = agents_action_probs_log.write(i, action_probs_log)
            # agents_action_probs_log = tf.stack(agents_action_probs_log, axis=-1)
            agents_action_probs_log = tf.transpose(agents_action_probs_log.stack())
            # print(f"{agents_action_probs_log.shape=}")
            
            # Compute actor loss.
            actor_loss = -tf.math.reduce_mean(y * agents_action_probs_log)

        # Compute gradients for the critic network.
        critic_grads = tape_critic.gradient(critic_loss, self.model_critic.trainable_variables)
        
        # Compute gradients for the actor network.
        actor_grads = tape_actor.gradient(actor_loss, self.model_actor.trainable_variables)

        # Update actor network.
        if isinstance(self.optimizer_actor, (list, tuple)):
            for i in range(len(self.optimizer_actor)):
                self.optimizer_actor[i].apply_gradients([(actor_grads[i], self.model_actor.trainable_variables[i])])
        else:
            self.optimizer_actor.apply_gradients(zip(actor_grads, self.model_actor.trainable_variables))

        # Update critic network.
        if isinstance(self.optimizer_critic, (list, tuple)):
            for i in range(len(self.optimizer_critic)):
                self.optimizer_critic[i].apply_gradients([(critic_grads[i], self.model_critic.trainable_variables[i])])
        else:
            self.optimizer_critic.apply_gradients(zip(critic_grads, self.model_critic.trainable_variables))


    def train(self,
        n_episodes: int, # Number of episodes.
        max_steps_per_episode: int = 10000,
        batch_size: int = 150,
        # max_memory_length: int = 10000,
        # reward_termination_threshold: float = None,
        # report_interval: int = 1, # Defaults to reporting every episode.
        ) -> dict[str, list]:
        
        print(f"Training for {n_episodes} episodes, press 'Ctrl+C' to terminate early")

        # Initialize the target critic network.
        self.model_critic_target.set_weights(self.model_critic.get_weights())

        episode_reward_history = []
        episode_metrics_history = []
        steps = 0
        try:
            with trange(n_episodes, unit='episode') as tepisode:
                for episode in tepisode:
                    tepisode.set_description(f"Episode {episode}")
                    
                    episode_reward = 0

                    ##### Run episode.
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
                        self.replay_buffer.append(interaction)
                        
                        # Set next state.
                        states = next_states

                        # Modify episode reward.
                        episode_reward += rewards

                        # Terminate episode.
                        if any(dones) or any(truncated):
                            break
                    
                    
                    # Update the models.
                    # if len(self.replay_buffer) >= batch_size and (steps+1) % 100 == 0:
                    if len(self.replay_buffer) >= batch_size:
                        # print('update')
                        self.update(batch_size)
                        self.update_target()

                    # # Update target network.
                    # if (episode+1) % 10 == 0:
                    #     # print(f"update target at {episode}")
                    #     self.update_target()

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





class MAPG_joint_old(Algorithm):
    """Multi-agent policy gradient with joint policy.
    """
    
    def __init__(self,
        env: gym.vector.VectorEnv, # Vectorized environment.
        model_actor: keras.Model,
        model_critic: keras.Model, # One central critic.
        model_critic_target: keras.Model, # One central critic.
        optimizer_actor: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        optimizer_critic: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        gamma: float,
        steps_per_update: int, # Number of steps per model update.
        steps_per_target_update: int = None, # Number of steps per target model update.
        tau: float = None, # Rate at which to slowly update the target network, should be tau << 1.
        max_memory_length: int = 10000,
        episode_metrics_callback = None, # Called at the end of each episode to report metrics.
        ):
        assert isinstance(env, gym.vector.VectorEnv), "only vectorized environments are supported (must be instance of `gym.vector.VectorEnv`)"
        assert isinstance(env.action_space, (gym.spaces.MultiDiscrete,)), "only `MultiDiscrete` action spaces are supported"
        self.env = env
        self.n_envs = self.env.num_envs
        
        self.episode_metrics_callback = episode_metrics_callback # cb(env)

        self.model_actor = model_actor
        self.model_critic = model_critic
        self.model_critic_target = model_critic_target
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        
        self.gamma = gamma
        self.tau = tau
        self.steps_per_target_update = steps_per_target_update
        assert tau is not None or steps_per_target_update is not None, 'must provide one of `tau` or `steps_per_target_update`'
        
        # Initialize replay buffer.
        self.replay_buffer: list[VectorInteraction] = deque(maxlen=max_memory_length)

        # If a single actor is provided then assume it is a joint policy.
        self.is_joint_policy = isinstance(self.model_actor, keras.models.Model) and not isinstance(self.model_actor, (list, tuple))

        # Initialize the target critic network.
        self.model_critic_target.set_weights(self.model_critic.get_weights())

        # Joint action --> per-agent action
        if self.is_joint_policy:
            n_actions = self.env.action_space[0].n # Get number of actions.
            self.joint_action_map = list(itertools.product(*list(itertools.repeat(list(range(n_actions)), self.n_envs))))
            self.individual_action_map = {tup: i for i, tup in enumerate(self.joint_action_map)}

    def _policy_joint(self, states) -> tuple[list[int], list[tf.Tensor]]:
        """One joint policy."""
        # Convert to tensor.
        s = tf.convert_to_tensor(states)
        s = tf.reshape(s, (1, *s.shape))

        # pi({a0, a1, ...} | {s0, s1, ...})
        joint_action_probs = self.model_actor(s)

        # Sample action from estimated probability distribution.
        joint_action = np.random.choice(joint_action_probs.shape[-1], p=np.squeeze(joint_action_probs))
        
        # Use joint action map here.
        actions = self.joint_action_map[joint_action]

        return actions, joint_action_probs

    def _policy_individual(self, states) -> tuple[list[int], list[tf.Tensor]]:
        """Individual policy for each actor."""
        joint_action, joint_action_probs = [], []
        for i, s in enumerate(states):
            # Convert to tensor.
            s = tf.convert_to_tensor(s)
            s = tf.reshape(s, (1, *s.shape))
            
            # pi(ai | si)
            action_probs = self.model_actor[i](s)

            # Sample action from estimated probability distribution.
            action = np.random.choice(action_probs.shape[-1], p=np.squeeze(action_probs))
            
            joint_action.append(action)
            joint_action_probs.append(action_probs)

        return joint_action, joint_action_probs

    def policy(self, states) -> tuple[list[int], list[tf.Tensor]]:
        """Get policy estimation for each agent."""
        if self.is_joint_policy:
            return self._policy_joint(states)
        else:
            return self._policy_individual(states)

    def values(self, states) -> list[tf.Tensor]:
        # Convert to tensor.
        s = tf.convert_to_tensor(states)
        s = tf.reshape(s, (1, *s.shape))

        # V({s0, s1, ...})
        joint_state_values = self.model_critic(s)

        return joint_state_values

    def update_target(self,):
        critic_target_weights = self.model_critic_target.get_weights()
        critic_weights = self.model_critic.get_weights()
        self.model_critic_target.set_weights([self.tau * critic_weights[i] + (1. - self.tau) * critic_target_weights[i] for i in range(len(critic_weights))])


    def update_critic(self,
        states_batched: tf.Tensor,
        rewards_batched: tf.Tensor,
        next_states_batched: tf.Tensor,
        ):
        
        rewards_batched = tf.math.reduce_sum(rewards_batched, axis=-1, keepdims=True)

        # Update the critic network using the replay batch.
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(self.model_critic.trainable_variables)

            # Compute critic loss.
            state_values = self.model_critic(states_batched) # V(s)
            next_state_values = self.model_critic_target(next_states_batched) # V(s')
            y = rewards_batched + self.gamma * next_state_values - state_values # yt
            critic_loss = tf.math.reduce_mean(tf.math.square(y)) # Critic loss is |y|^2

        # Compute gradients for the critic network.
        critic_grads = tape_critic.gradient(critic_loss, self.model_critic.trainable_variables)

        # Update critic network.
        if isinstance(self.optimizer_critic, (list, tuple)):
            for i in range(len(self.optimizer_critic)):
                self.optimizer_critic[i].apply_gradients([(critic_grads[i], self.model_critic.trainable_variables[i])])
        else:
            self.optimizer_critic.apply_gradients(zip(critic_grads, self.model_critic.trainable_variables))

    def update_actor_joint(self,
        states_batched: tf.Tensor,
        actions_batched: tf.Tensor,
        rewards_batched: tf.Tensor,
        next_states_batched: tf.Tensor,
        ):
        """Single model as joint policy."""
        state_values = self.model_critic(states_batched) # V(s)
        next_state_values = self.model_critic_target(next_states_batched) # V(s')
        y = rewards_batched + self.gamma * next_state_values - state_values # yt
        
        with tf.GradientTape() as tape_actor:
            tape_actor.watch(self.model_actor.trainable_variables)

            action_probs = self.model_actor(states_batched) # pi(a|s)
            actions_batched_as_hashable = list(map(tuple, actions_batched.numpy()))
            joint_actions_batched = [self.individual_action_map[tup] for tup in actions_batched_as_hashable]

            # Convert batched individual actions back to a joint action.
            id_action_pairs = np.array([(i,a) for i, a in enumerate(joint_actions_batched)]) # (n_time_steps, 2,)
            probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
            action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)
            actor_loss = -tf.math.reduce_sum(y * action_probs_log)
        
        # Compute gradients for the actor network.
        actor_grads = tape_actor.gradient(actor_loss, self.model_actor.trainable_variables)

        # Update actor network.
        if isinstance(self.optimizer_actor, (list, tuple)):
            for i in range(len(self.optimizer_actor)):
                self.optimizer_actor[i].apply_gradients([(actor_grads[i], self.model_actor.trainable_variables[i])])
        else:
            self.optimizer_actor.apply_gradients(zip(actor_grads, self.model_actor.trainable_variables))

    def update_actor_shared(self,
        states_batched: tf.Tensor,
        actions_batched: tf.Tensor, # (batch, t, agent)
        rewards_batched: tf.Tensor,
        next_states_batched: tf.Tensor,
        ):
        """All agents us the same policy via parameter sharing."""
        state_values = self.model_critic(states_batched) # V(s)
        next_state_values = self.model_critic_target(next_states_batched) # V(s')
        y = rewards_batched + self.gamma * next_state_values - state_values # yt
        
        with tf.GradientTape() as tape_actor:
            tape_actor.watch(self.model_actor.trainable_variables)
            
            # Estimate the policy for each actor individually.
            agents_action_probs_log = tf.zeros_like(actions_batched, dtype='float32')
            for i in range(self.n_envs):
                action_probs = self.model_actor(states_batched[:,:,i]) # pi(a|s)

                # Convert batched individual actions back to a joint action.
                id_action_pairs = np.array([(i,a) for i, a in enumerate(actions_batched[:,:,i])]) # (n_time_steps, 2,)
                probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
                action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)
                agents_action_probs_log[:,i] = action_probs_log
            
            actor_loss = -tf.math.reduce_sum(y * action_probs_log)
        
        # Compute gradients for the actor network.
        actor_grads = tape_actor.gradient(actor_loss, self.model_actor.trainable_variables)

        # Update actor network.
        if isinstance(self.optimizer_actor, (list, tuple)):
            for i in range(len(self.optimizer_actor)):
                self.optimizer_actor[i].apply_gradients([(actor_grads[i], self.model_actor.trainable_variables[i])])
        else:
            self.optimizer_actor.apply_gradients(zip(actor_grads, self.model_actor.trainable_variables))

    def update_actor(self,
        states_batched: tf.Tensor,
        actions_batched: tf.Tensor,
        rewards_batched: tf.Tensor,
        next_states_batched: tf.Tensor,
        ):

        # Single joint policy.
        if self.is_joint_policy:
            self.update_actor_joint(states_batched, actions_batched, rewards_batched, next_states_batched)

        # Separate policies.
        else:
            self.update_actor_shared(states_batched, actions_batched, rewards_batched, next_states_batched)


    def update(self, batch_size: int):

        # Randomly select interactions from replay memory and train on them.
        training_batch_list: list[VectorInteraction] = np.random.choice(self.replay_buffer, size=batch_size)
        # Convert training batch to dictionary.
        training_batch_dict = {k:[asdict(d)[k] for d in training_batch_list] for k in asdict(training_batch_list[0]).keys()}
        
        states_batched = np.array(training_batch_dict['states']).squeeze()
        actions_batched = np.array(training_batch_dict['actions']).squeeze()
        # action_probs_batched = np.array(training_batch_dict['action_probs']).squeeze()
        rewards_batched = np.array(training_batch_dict['rewards'], dtype='float32').squeeze()
        next_states_batched = np.array(training_batch_dict['next_states']).squeeze()
        # dones_batched = np.array(training_batch_dict['dones']).squeeze()
        
        states_batched = tf.convert_to_tensor(states_batched)
        actions_batched = tf.convert_to_tensor(actions_batched)
        # action_probs_batched = tf.convert_to_tensor(action_probs_batched)
        rewards_batched = tf.convert_to_tensor(rewards_batched)
        next_states_batched = tf.convert_to_tensor(next_states_batched)
        # dones_batched = tf.convert_to_tensor(dones_batched)


        ###
        # Update critic network.
        ###
        self.update_critic(states_batched, rewards_batched, next_states_batched)

        ###
        # Update policy network(s).
        ###
        self.update_actor(states_batched, actions_batched, rewards_batched, next_states_batched)

        # Update target network.
        self.update_target()


    def train(self,
        n_episodes: int, # Number of episodes.
        max_steps_per_episode: int = 10000,
        batch_size: int = 150,
        # max_memory_length: int = 10000,
        # reward_termination_threshold: float = None,
        # report_interval: int = 1, # Defaults to reporting every episode.
        ) -> dict[str, list]:
        
        print(f"Training for {n_episodes} episodes, press 'Ctrl+C' to terminate early")

        # Initialize the target critic network.
        self.model_critic_target.set_weights(self.model_critic.get_weights())

        episode_reward_history = []
        episode_metrics_history = []
        steps = 0
        try:
            with trange(n_episodes, unit='episode') as tepisode:
                for episode in tepisode:
                    tepisode.set_description(f"Episode {episode}")
                    
                    episode_reward = 0

                    ##### Run episode.
                    # Reset environment.
                    states, _ = self.env.reset()
                    
                    # Iterate through environment at discrete time steps.
                    for _ in range(max_steps_per_episode):
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
                        self.replay_buffer.append(interaction)
                        
                        # Update the models.
                        if (steps + 1) % self.steps_per_target_update == 0:
                            self.update(batch_size)

                        # Modify episode reward.
                        episode_reward += rewards

                        # Terminate episode.
                        if any(dones) or any(truncated):
                            break
                    
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