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
        
        # # Compute slope to goal.
        # slope = np.divide(
        #     self.goal_position[1] - self.unwrapped.agent_pos[1],
        #     self.goal_position[0] - self.unwrapped.agent_pos[0],
        # )
        # slope = np.arctan(slope) # Angle.
        # # print(f"{slope=}, {np.arctan(slope)=}")

        # Give a negative reward for standing still.
        if not terminated and self.last_pos is not None and self.unwrapped.agent_pos == self.last_pos:
            # print(f"matched pos")
            reward = -2.
        # Give a negative reward for each step.
        elif not terminated and reward == 0:
                # reward = +1. * 1./slope
                # reward = +1.
                reward = -1.
        # If we're terminating, then scale the reward by 100.
        elif terminated:
            reward = reward * 100.
            
        self.last_pos = self.unwrapped.agent_pos
        # print(f"{self.unwrapped.agent_pos=}")

        return obs, reward, terminated, truncated, info
    
    # def reset(self, *args, **kwargs):
    #     obs, info = self.env.reset(*args, **kwargs)

    #     if not self.goal_position:
    #         self.goal_position = [
    #             x for x, y in enumerate(self.unwrapped.grid.grid) if isinstance(y, Goal)
    #         ]
    #         # in case there are multiple goals , needs to be handled for other env types
    #         if len(self.goal_position) >= 1:
    #             self.goal_position = (
    #                 int(self.goal_position[0] / self.unwrapped.height),
    #                 self.goal_position[0] % self.unwrapped.width,
    #             )
        
    #     # print(f"{self.goal_position=}, {self.unwrapped.agent_pos=}")

    #     return obs, info