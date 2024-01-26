import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange
import gymnasium as gym
from dataclasses import dataclass

from . import environments
from . import agents

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
class JointInteraction:
    """Environment interaction."""
    states: tf.Tensor
    joint_action: int
    joint_action_probs: tf.Tensor
    rewards: float
    next_states: tf.Tensor
    done: bool


class EnvTrainer:
    """Model training for specific environments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_episode(self, agent: agents.Agent) -> tuple[list[dict], dict]:
        raise NotImplementedError

    def train(self,
        n_episodes: int, # Number of episodes.
        agent: agents.Agent,
        ) -> dict[str, list]:
        raise NotImplementedError


class GymTrainer(EnvTrainer):
    
    def __init__(self,
        env: gym.Env,
        ):
        self.env = env

    def compute_episode_reward(self, interaction_history: list[Interaction]) -> float:
        """Episode reward is the sum of all rewards per time step."""
        rewards = np.array([ei.reward for ei in interaction_history], dtype='float32') # Gather all rewards.
        return np.sum(rewards)

    def run_episode(self, agent: agents.Agent) -> tuple[list[Interaction], dict]:
        """Runs a single episode in the training environment."""
        
        # Reset environment.
        state, _ = self.env.reset()

        interaction_history: list[dict] = []
        while True:

            ####
            ### Interact with environment
            ####

            # Get the 
            action, action_probs = agent.policy(state)

            # Step through environment using joint action.
            next_state, reward, done, _, _ = self.env.step(action)

            # Preserve interaction.
            interaction = Interaction(
                state=state,
                action=action,
                action_probs=action_probs,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            interaction_history.append(interaction)
            
            # Set next state.
            state = next_state

            if done:
                break

        # Preserve metrics as dictionary.
        metrics = dict(
            episode_reward=self.compute_episode_reward(interaction_history),
        )

        return interaction_history, metrics


    def train(self,
        n_episodes: int, # Number of episodes.
        agent: agents.Agent,
        reward_termination_threshold: float = None,
        report_interval: int = 1, # Defaults to reporting every episode.
        ) -> dict[str, list]:
        
        episode_metrics_history = []
        episode_reward_history = []
        with trange(n_episodes, unit='episode') as tepisode:
            for episode in tepisode:
                tepisode.set_description(f"Episode {episode}")
                
                episode_interaction_history, episode_metrics = self.run_episode(agent=agent)
                
                tepisode.set_postfix(**episode_metrics)
                
                # Update the models using the controller.
                batched_reward = np.array([ei.reward for ei in episode_interaction_history], dtype='float32').squeeze()
                batched_state = np.array([ei.state for ei in episode_interaction_history]).squeeze()
                batched_next_state = np.array([ei.next_state for ei in episode_interaction_history]).squeeze()
                batched_action = np.array([ei.action for ei in episode_interaction_history]).squeeze()
                batched_action_probs = np.array([ei.action_probs for ei in episode_interaction_history], dtype='float32').squeeze()

                batched_reward = tf.convert_to_tensor(batched_reward)
                batched_state = tf.convert_to_tensor(batched_state)
                batched_next_state = tf.convert_to_tensor(batched_next_state)
                batched_action = tf.convert_to_tensor(batched_action)
                batched_action_probs = tf.convert_to_tensor(batched_action_probs)

                # Update controller.
                agent.update(
                    batched_state,
                    batched_action,
                    batched_action_probs,
                    batched_next_state,
                    batched_reward,
                )
                
                episode_metrics_history.append(episode_metrics)
                episode_reward_history.append(episode_metrics['episode_reward'])
                
                # Report status at regular episodic intervals.
                if report_interval is not None and (episode+1) % report_interval == 0:
                    avg_rewards = np.mean(episode_reward_history[-10:])
                    msg = "Episode {}/{}, average last {} rewards {}".format(episode+1, n_episodes, report_interval, avg_rewards)
                    tepisode.set_description(f"Episode {episode+1}") # Force next episode description.
                    print(msg, flush=True) # Print status message.
                    
                    # Terminate training if score reaches above threshold.
                    # This is to prevent over-training.
                    if reward_termination_threshold is not None and avg_rewards >= reward_termination_threshold:
                        break
                
                tepisode.set_description(f"Episode {episode+1}") # Force next episode description.
        
        # Convert 'list of dicts' to 'dict of lists'.
        episode_metrics_history = {k:[d[k] for d in episode_metrics_history] for k in episode_metrics_history[0].keys()}
        
        return episode_metrics_history


class CoinGame2Trainer(EnvTrainer):
    """Train MARL models using CoinGame2 environment."""

    def __init__(self, env_params):
        self.env = environments.coin_game.coin_game_make(env_params)

    def compute_episode_reward(self, interaction_history: list[JointInteraction]) -> float:
        """Episode reward is the sum of all rewards per time step."""
        rewards = np.array([ei.rewards for ei in interaction_history], dtype='float32') # Gather all rewards.
        return np.sum(rewards)

    def run_episode(self, multiagent: agents.MultiAgent) -> tuple[list[JointInteraction], dict]:
        """Runs a single episode in the training environment."""
        
        # Reset environment.
        states = self.env.reset()

        interaction_history: list[dict] = []
        while True:

            ####
            ### Interact with environment
            ####

            # Get the 
            joint_action, joint_action_probs = multiagent.policy(states)

            # Step through environment using joint action.
            next_states, rewards, done, _ = self.env.step(joint_action)

            # Preserve interaction.
            interaction = JointInteraction(
                states=states,
                joint_action=joint_action,
                joint_action_probs=joint_action_probs,
                rewards=rewards,
                next_states=next_states,
                done=done,
            )
            interaction_history.append(interaction)
            
            # Set next state.
            states = next_states

            if done:
                break
            
        # Compute environment performance metrics.
        coins_collected = self.env.domain_values()[self.env.get_index('coins_collected')]
        own_coins_collected = self.env.domain_values()[self.env.get_index('own_coins_collected')]
        episode_coins_collected = coins_collected
        episode_own_coins_collected = own_coins_collected
        episode_undiscounted_reward = np.sum(self.env.undiscounted_returns)
        episode_discounted_reward = np.sum(self.env.discounted_returns)
        
        # Compute "own coin rate", and treat all divide by zeros errors as zero.
        if coins_collected != 0:
            episode_own_coin_rate = own_coins_collected/coins_collected
        else:
            episode_own_coin_rate = 0

        # Preserve metrics as dictionary.
        metrics = dict(
            episode_reward=self.compute_episode_reward(interaction_history),
            episode_undiscounted_reward=episode_undiscounted_reward,
            episode_discounted_reward=episode_discounted_reward,
            episode_coins_collected=episode_coins_collected,
            episode_own_coins_collected=episode_own_coins_collected,
            episode_own_coin_rate=episode_own_coin_rate,
        )

        return interaction_history, metrics


    def train(self,
        n_episodes: int, # Number of episodes.
        multiagent: agents.MultiAgent,
        reward_termination_threshold: float = None,
        report_interval: int = 1, # Defaults to reporting every episode.
        ) -> dict[str, list]:
        
        episode_metrics_history = []
        episode_reward_history = []
        episode_discounted_reward_history = []
        episode_undiscounted_reward_history = []
        episode_coins_collected_history = []
        episode_own_coins_collected_history = []
        episode_own_coin_rate_history = []
        with trange(n_episodes, unit='episode') as tepisode:
            for episode in tepisode:
                tepisode.set_description(f"Episode {episode}")
                
                episode_interaction_history, episode_metrics = self.run_episode(multiagent=multiagent)
                
                tepisode.set_postfix(**episode_metrics)
                
                # Update the models using the controller.
                batched_rewards = np.array([ei.rewards for ei in episode_interaction_history], dtype='float32').squeeze()
                batched_states = np.array([ei.states for ei in episode_interaction_history]).squeeze()
                batched_next_states = np.array([ei.next_states for ei in episode_interaction_history]).squeeze()
                batched_joint_actions = np.array([ei.joint_action for ei in episode_interaction_history]).squeeze()
                batched_joint_action_probs = np.array([ei.joint_action_probs for ei in episode_interaction_history], dtype='float32').squeeze()

                batched_rewards = tf.convert_to_tensor(batched_rewards)
                batched_states = tf.convert_to_tensor(batched_states)
                batched_next_states = tf.convert_to_tensor(batched_next_states)
                batched_joint_actions = tf.convert_to_tensor(batched_joint_actions)
                batched_joint_action_probs = tf.convert_to_tensor(batched_joint_action_probs)

                # Update controller.
                multiagent.update(
                    batched_states,
                    batched_joint_actions,
                    batched_joint_action_probs,
                    batched_next_states,
                    batched_rewards,
                )
                
                episode_metrics_history.append(episode_metrics)
                episode_reward_history.append(episode_metrics['episode_reward'])
                episode_discounted_reward_history.append(episode_metrics['episode_discounted_reward'])
                episode_undiscounted_reward_history.append(episode_metrics['episode_undiscounted_reward'])
                episode_coins_collected_history.append(episode_metrics['episode_coins_collected'])
                episode_own_coins_collected_history.append(episode_metrics['episode_own_coins_collected'])
                episode_own_coin_rate_history.append(episode_metrics['episode_own_coin_rate'])
                
                # Report status at regular episodic intervals.
                if report_interval is not None and (episode+1) % report_interval == 0:
                    avg_rewards = np.mean(episode_reward_history[-10:])
                    avg_discounted_rewards = np.mean(episode_discounted_reward_history[-10:])
                    avg_undiscounted_rewards = np.mean(episode_undiscounted_reward_history[-10:])
                    avg_coins_collected = np.mean(episode_coins_collected_history[-10:])
                    avg_own_coins_collected = np.mean(episode_own_coins_collected_history[-10:])
                    avg_own_coin_rate = np.mean(episode_own_coin_rate_history[-10:])
                    msg = "Episode {}/{}, average last {} rewards {}".format(episode+1, n_episodes, report_interval, avg_rewards)
                    msg = f"{msg}: {avg_undiscounted_rewards=}, {avg_discounted_rewards=}, {avg_coins_collected=}, {avg_own_coins_collected=}, {avg_own_coin_rate=}"
                    tepisode.set_description(f"Episode {episode+1}") # Force next episode description.
                    print(msg, flush=True) # Print status message.
                    
                    # Terminate training if score reaches above threshold.
                    # This is to prevent over-training.
                    if reward_termination_threshold is not None and avg_rewards >= reward_termination_threshold:
                        break
                
                tepisode.set_description(f"Episode {episode+1}") # Force next episode description.
        
        # Convert 'list of dicts' to 'dict of lists'.
        episode_metrics_history = {k:[d[k] for d in episode_metrics_history] for k in episode_metrics_history[0].keys()}
        
        return episode_metrics_history