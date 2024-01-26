import numpy as np
import tensorflow as tf
from tensorflow import keras


class Agent:
    """Reinforcement learning (RL) agent.
    
    Handles training a single RL agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, 
        batched_states: tf.Tensor,
        batched_actions: tf.Tensor,
        batched_action_probs: tf.Tensor,
        batched_next_states: tf.Tensor,
        batched_rewards: tf.Tensor,
        ):
        raise NotImplementedError

    def policy(self, state):
        raise NotImplementedError


class MultiAgent(Agent):
    """Multi-agent reinforcement learning (MARL) training.
    
    Handles training multiple RL agents in tandem.
    """

    def update(self, 
        batched_states: tf.Tensor,
        batched_joint_actions: tf.Tensor,
        batched_joint_action_probs: tf.Tensor,
        batched_next_states: tf.Tensor,
        batched_rewards: tf.Tensor,
        ):
        raise NotImplementedError

    def policy(self, states):
        raise NotImplementedError



class AC(Agent):
    """Actor-critic agent."""
    def __init__(self,
        model_actor: keras.Model,
        model_critic: keras.Model,
        optimizer_actor,
        optimizer_critic,
        n_actions: int,
        gamma: float,
        ):
        super().__init__()
        self.model_actor = model_actor
        self.model_critic = model_critic
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.n_actions = n_actions
        self.gamma = gamma
        
    def policy(self, state) -> tuple[int, tf.Tensor]:
        
        # Convert to tensor.
        s = tf.convert_to_tensor(state)
        s = tf.reshape(s, (1, *s.shape))

        # pi(a | s)
        action_probs = self.model_actor(s)

        # Sample action from estimated probability distribution.
        action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
        
        return action, action_probs
    
    def value(self, state) -> tf.Tensor:
        # Convert to tensor.
        s = tf.convert_to_tensor(state)
        s = tf.reshape(s, (1, *s.shape))

        # V(s)
        state_values = self.model_critic(s)
        return state_values
    
    def get_expected_returns(self,
        rewards: tf.Tensor,
        standardize: bool = True,
        ) -> tf.Tensor:
        n_rewards = rewards.shape[0]
        
        returns = []
        discounted_sum = 0
        rewards = rewards[::-1] # Reverse the rewards.
        for i in range(n_rewards):
            discounted_sum = rewards[i] + self.gamma * discounted_sum
            returns.append(discounted_sum)
        returns = tf.convert_to_tensor(np.array(returns[::-1], dtype='float32'))
        
        if standardize:
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - tf.math.reduce_mean(returns, axis=0)) / (tf.math.reduce_std(returns, axis=0) + eps)

        return returns
    
    def update(self, 
        batched_states: tf.Tensor, # (n_time_steps, n_agents, 36,)
        batched_actions: tf.Tensor,
        batched_action_probs: tf.Tensor,
        batched_next_states: tf.Tensor,
        batched_rewards: tf.Tensor, # (n_time_steps, n_agents,)
        ):
        """Update using expected discount returns."""

        huber_loss = tf.keras.losses.Huber(reduction=keras.losses.Reduction.SUM)
        
        # Pre-compute returns since we already know rewards.
        returns = self.get_expected_returns(batched_rewards, standardize=True)
        
        # Train individual models separately.
        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            tape_actor.watch(self.model_actor.trainable_variables)
            tape_critic.watch(self.model_critic.trainable_variables)
            
            # pi(a | s)
            action_probs = self.model_actor(batched_states) # (n_time_steps,n_actions,)
            
            # V(s)
            state_values = self.model_critic(batched_states)
            
            # Actor loss.
            advantage = returns - state_values
            id_action_pairs = np.array([(i,a) for i, a in enumerate(batched_actions)]) # (n_time_steps, 2,)
            probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
            action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)
            actor_loss = -tf.math.reduce_sum(action_probs_log * advantage)
            
            # Critic loss.
            critic_loss = huber_loss(state_values, returns)

        # Compute gradients for both actor and critic.
        actor_grads = tape_actor.gradient(actor_loss, self.model_actor.trainable_variables)
        critic_grads = tape_critic.gradient(critic_loss, self.model_critic.trainable_variables)

        # Apply gradients to each model's parameters.
        if isinstance(self.optimizer_actor, (list, tuple)):
            for i in range(len(self.optimizer_actor)):
                self.optimizer_actor[i].apply_gradients([(actor_grads[i], self.model_actor.trainable_variables[i])])
        else:
            self.optimizer_actor.apply_gradients(zip(actor_grads, self.model_actor.trainable_variables))
            
        if isinstance(self.optimizer_critic, (list, tuple)):
            for i in range(len(self.optimizer_critic)):
                self.optimizer_critic[i].apply_gradients([(critic_grads[i], self.model_critic.trainable_variables[i])])
        else:
            self.optimizer_critic.apply_gradients(zip(critic_grads, self.model_critic.trainable_variables))


class IAC(MultiAgent):
    """Independent actor-critic (IAC) agents for MARL environments.
    
    In IAC, every agent has their own actor and critic models.
    """

    def __init__(self,
        model_actors: list[keras.Model],
        model_critics: list[keras.Model],
        gamma: float,
        optimizer_actor,
        optimizer_critic,
        n_actions: int,
        ):
        super().__init__()
        self.model_actors = model_actors
        self.model_critics = model_critics
        self.gamma = gamma
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.n_actions = n_actions
        
    def agent_policy(self, index: int, state) -> tuple[int, tf.Tensor]:
        
        # Convert to tensor.
        s = tf.convert_to_tensor(state)
        s = tf.reshape(s, (1, *s.shape))

        # pi(a | s)
        action_probs = self.model_actors[index](s)

        # Sample action from estimated probability distribution.
        action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
        
        return action, action_probs
    
    def agent_value(self, index: int, state) -> tf.Tensor:
        # Convert to tensor.
        s = tf.convert_to_tensor(state)
        s = tf.reshape(s, (1, *s.shape))

        # V(s)
        state_values = self.model_critics[index](s)
        return state_values

    def policy(self, states) -> tuple[list[int], list[tf.Tensor]]:
        joint_action, joint_action_probs = tuple(zip(*[self.agent_policy(i, s) for i, s in enumerate(states)]))
        return joint_action, joint_action_probs

    def values(self, states) -> list[tf.Tensor]:
        joint_state_values = [self.agent_value(i, s) for i, s in enumerate(states)]
        return joint_state_values
    
    def get_expected_returns(self,
        rewards: tf.Tensor,
        standardize: bool = True,
        ) -> tf.Tensor:
        n_rewards = rewards.shape[0]
        
        returns = []
        discounted_sum = 0
        rewards = rewards[::-1] # Reverse the rewards.
        for i in range(n_rewards):
            discounted_sum = rewards[i] + self.gamma * discounted_sum
            returns.append(discounted_sum)
        returns = tf.convert_to_tensor(np.array(returns[::-1], dtype='float32'))
        
        if standardize:
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - tf.math.reduce_mean(returns, axis=0)) / (tf.math.reduce_std(returns, axis=0) + eps)
        
        return returns
    
    def update(self, 
        batched_states: tf.Tensor, # (n_time_steps, n_agents, 36,)
        batched_joint_actions: tf.Tensor,
        batched_joint_action_probs: tf.Tensor,
        batched_next_states: tf.Tensor,
        batched_rewards: tf.Tensor, # (n_time_steps, n_agents,)
        ):
        """Update using expected discount returns."""

        huber_loss = tf.keras.losses.Huber(reduction=keras.losses.Reduction.SUM)
        
        # Pre-compute returns since we already know rewards.
        returns = self.get_expected_returns(batched_rewards, standardize=True)
        
        # Train individual models separately.
        for aidx, (ma, mc) in enumerate(zip(self.model_actors, self.model_critics)):
            with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                tape_actor.watch(ma.trainable_variables)
                tape_critic.watch(mc.trainable_variables)
                
                # pi(a | s)
                action_probs = ma(batched_states[:,aidx,:]) # (n_time_steps,n_actions,)
                
                # V(s)
                state_values = mc(batched_states[:,aidx,:])
                
                # Actor loss.
                advantage = returns[:,aidx] - state_values
                id_action_pairs = np.array([(i,a) for i, a in enumerate(batched_joint_actions[:,aidx])]) # (n_time_steps, 2,)
                probs_of_chosen_actions = tf.gather_nd(action_probs, id_action_pairs) # (n_time_steps,)
                action_probs_log = tf.math.log(probs_of_chosen_actions) # (n_time_steps,)
                actor_loss = -tf.math.reduce_sum(action_probs_log * advantage)
                
                # Critic loss.
                critic_loss = huber_loss(state_values, returns[:,aidx])

            # Compute gradients for both actor and critic.
            actor_grads = tape_actor.gradient(actor_loss, ma.trainable_variables)
            critic_grads = tape_critic.gradient(critic_loss, mc.trainable_variables)

            # Apply gradients to each model's parameters.
            if isinstance(self.optimizer_actor, (list, tuple)):
                for i in range(len(self.optimizer_actor)):
                    self.optimizer_actor[i].apply_gradients([(actor_grads[i], ma.trainable_variables[i])])
            else:
                self.optimizer_actor.apply_gradients(zip(actor_grads, ma.trainable_variables))
                
            if isinstance(self.optimizer_critic, (list, tuple)):
                for i in range(len(self.optimizer_critic)):
                    self.optimizer_critic[i].apply_gradients([(critic_grads[i], mc.trainable_variables[i])])
            else:
                self.optimizer_critic.apply_gradients(zip(critic_grads, mc.trainable_variables))