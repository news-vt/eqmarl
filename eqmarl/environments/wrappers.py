import numpy as np
import gymnasium as gym
from minigrid.core.world_object import Goal

class StepRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_pos = None
        self.goal_position: tuple = None
        
    def step(self, action):
        # Take the given action.
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure the environment agent position is a tuple type.
        current_pos = tuple(self.unwrapped.agent_pos)

        # Give a negative reward for standing still.
        if not terminated and self.last_pos is not None and current_pos == self.last_pos:
            # print(f"matched pos")
            reward = -2.
        # Give a negative reward for each step.
        elif not terminated and reward == 0:
                reward = -1.
        # If we're terminating, then scale the reward by 100.
        elif terminated:
            reward = reward * 100.
        
        # Update agent position.
        self.last_pos = current_pos

        return obs, reward, terminated, truncated, info