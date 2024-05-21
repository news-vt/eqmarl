"""
This environment was adapted from the work of Phan et al. 2022 (https://dl.acm.org/doi/10.5555/3535850.3535967).

The original source for this code can be found on GitHub at: https://github.com/thomyphan/emergent-cooperation/blob/main/mate/environments/coin_game.py
"""

from typing import Union
import numpy
import random
import gymnasium as gym

class Environment:

    def __init__(self, params) -> None:
        self.domain_value_labels = params["domain_value_labels"]
        self.observation_dim = params["observation_dim"]
        self.nr_agents = params["nr_agents"]
        self.nr_actions = params["nr_actions"]
        self.time_limit = params["time_limit"]
        self.gamma = params["gamma"]
        self.time_step = 0
        self.sent_gifts = numpy.zeros(self.nr_agents)
        self.discounted_returns = numpy.zeros(self.nr_agents)
        self.undiscounted_returns = numpy.zeros(self.nr_agents)
        self.domain_counts = numpy.zeros(len(self.domain_value_labels))
        self.last_joint_action = -numpy.ones(self.nr_agents, dtype='int')

    """
     Performs the joint action in order to change the environment.
     Returns the reward for each agent in a list sorted by agent ID.
    """
    def perform_step(self, joint_action):
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        return numpy.zeros(self.nr_agents), {}

    """
     Indicates if an episode is done and the environments needs
     to be reset.
    """
    def is_done(self):
        return self.time_step >= self.time_limit

    def action_as_vector(self, action):
        if action < self.nr_actions:
            vector = numpy.zeros(self.nr_actions)
            if action >= 0:
                vector[action] = 1
        else:
            vector = numpy.ones(self.nr_actions)
        return vector

    """
     Performs a joint action to change the state of the environment.
     Returns the joint observation, the joint reward, a done flag,
     and other optional information (e.g., logged data).
     Note: The joint action must be a list ordered according to the agent ID!.
    """
    def step(self, joint_action):
        assert len(joint_action) == self.nr_agents, "Length of 'joint_action' is {}, expected {}"\
            .format(len(joint_action), self.nr_agents)
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        rewards, infos = self.perform_step(joint_action)
        for i, a in enumerate(joint_action):
            self.last_joint_action[i] = a
            if a >= self.nr_actions:
                self.sent_gifts[i] += 1
        assert len(rewards) == self.nr_agents, "Length of 'rewards' is {}, expected {}"\
            .format(len(rewards), self.nr_agents)
        observations = self.joint_observation()
        assert len(observations) == self.nr_agents, "Length of 'observations' is {}, expected {}"\
            .format(len(observations), self.nr_agents)
        self.time_step += 1
        self.domain_counts[0] += 1.0
        self.undiscounted_returns += rewards
        self.discounted_returns += (self.gamma**self.time_step)*rewards
        if "neighbor_agents" not in infos:
            infos["neighbor_agents"] = [[j for j in range(self.nr_agents) if j != i] for i in range(self.nr_agents)]
        return observations, rewards, self.is_done(), infos

    def get_index(self, label):
        return self.domain_value_labels.index(label)

    """
     The local observation for a specific agent. Only visible for
     the corresponding agent and private to others.
    """
    def local_observation(self, agent_id):
        pass

    """
     Returns the observations of all agents in a listed sorted by agent ids.
    """
    def joint_observation(self):
        return [numpy.array(self.local_observation(i)).reshape(self.observation_dim) for i in range(self.nr_agents)]

    """
     Returns a high-level value which is domain-specific.
    """
    def domain_values(self):
        return self.domain_counts

    def domain_value_debugging_indices(self):
        return 0,1

    """
     Re-Setup of the environment for a new episode.
    """
    def reset(self):
        self.time_step = 0
        self.discounted_returns[:] = 0
        self.undiscounted_returns[:] = 0
        self.last_joint_action[:] = -1
        self.domain_counts[:] = 0
        self.sent_gifts[:] = 0
        return self.joint_observation()
    


MOVE_NORTH = 0
MOVE_SOUTH = 1
MOVE_WEST = 2
MOVE_EAST = 3

COIN_GAME_ACTIONS = [MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST]

class MovableAgent:

    def __init__(self, agent_id, width, height, view_range):
        self.agent_id = agent_id
        self.position = None
        self.width = width
        self.height = height
        self.view_range = view_range

    def move(self, action):
        x, y = self.position
        if action == MOVE_NORTH and y + 1 < self.height:
            self.position = (x, y + 1)
        if action == MOVE_SOUTH and y - 1 >= 0:
            self.position = (x, y - 1)
        if action == MOVE_EAST and x + 1 < self.width:
            self.position = (x + 1, y)
        if action == MOVE_WEST and x - 1 >= 0:
            self.position = (x - 1, y)

    def reset(self, position):
        self.position = position

    def visible_positions(self):
        x0, y0 = self.position
        x_center = int(self.view_range/2)
        y_center = int(self.view_range/2)
        positions = [(x,y) for x in range(-x_center+x0, x_center+1+x0)\
            for y in range(-y_center+y0, y_center+1+y0)]
        return positions

    def relative_position_to(self, other_position):
        dx = other_position[0] - self.position[0]
        dy = other_position[1] - self.position[1]
        return dx, dy

class Coin:

    def __init__(self, nr_agents):
        self.agent_ids = list(range(nr_agents))
        self.agent_id = None # Indicates color of coin
        self.position = None

    def reset(self, position):
        self.position = position
        self.agent_id = random.choice(self.agent_ids)


# Observation shape for "CoinGame-2" is (4,3,3) which means:
# - index=0: 3x3 grid world with a `1` where the current agent is.
# - index=1: 3x3 grid world where a `1` is added to every cell that has other agents.
# - index=2: 3x3 grid world with a `1` for location of coin that matches the focused agent's color.
# - index=3: 3x3 grid world with a `1` for location of coin that matches other agent's color.

class CoinGameEnvironment(Environment):

    def __init__(self, params):
        params["domain_value_labels"] = ["time_steps", "coins_collected", "own_coins_collected", "coin_1_generated"]
        super(CoinGameEnvironment, self).__init__(params)
        self.width = params["width"]
        self.height = params["height"]
        self.view_range = params["view_range"]
        self.observation_shape = (4, self.width, self.height)
        self.agents = [MovableAgent(i, self.width, self.height, self.view_range) for i in range(self.nr_agents)]
        self.positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.coin = Coin(self.nr_agents)

    def perform_step(self, joint_action):
        rewards, infos = super(CoinGameEnvironment, self).perform_step(joint_action)
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        coin_collected = False
        agent_actions = list(zip(self.agents, joint_action))
        random.shuffle(agent_actions)
        for agent, action in agent_actions:
            agent.move(action)
            if agent.position == self.coin.position:
                self.domain_counts[1] += 1
                coin_collected = True
                rewards[agent.agent_id] += 1
                if agent.agent_id != self.coin.agent_id:
                    rewards[self.coin.agent_id] -= 2
                else:
                    self.domain_counts[2] += 1
        if coin_collected:
            old_position = self.coin.position
            new_position = random.choice([pos for pos in self.positions if pos != old_position])
            self.coin.reset(new_position)
        return rewards, infos

    def get_metric_indices(self, metric):
        if metric == "own_coin_prob":
            return self.get_index("own_coins_collected"), self.get_index("coins_collected")
        return None, self.get_index("time_steps")
    
    def domain_value_debugging_indices(self):
        return self.get_index("own_coins_collected"), self.get_index("coins_collected")

    def local_observation(self, agent_id):
        observation = numpy.zeros(self.observation_shape)
        focus_agent = self.agents[agent_id]
        x, y = focus_agent.position
        observation[0][x][y] = 1
        for agent in self.agents:
            if agent.agent_id != focus_agent.agent_id:
                x, y = agent.position
                observation[1][x][y] += 1
        index = 2
        if self.coin.agent_id != agent_id:
            index = 3
        x, y = self.coin.position
        observation[index][x][y] = 1
        return observation.reshape(-1)

    def reset(self):
        positions = random.sample(self.positions, k=(self.nr_agents+1))
        for i, agent in enumerate(self.agents):
            agent.reset(positions[i])
        self.coin.reset(positions[-1])
        return super(CoinGameEnvironment, self).reset()

def coin_game_make(params):
    domain_name = params["domain_name"]
    
    default_params = {}
    default_params["gamma"] = 0.95
    default_params["time_limit"] = 150
    default_params["nr_actions"] = len(COIN_GAME_ACTIONS)
    default_params["history_length"] = 1
    default_params["view_range"] = 5
    if domain_name == "CoinGame-2":
        default_params["nr_agents"] = 2
        default_params["width"] = 3
        default_params["height"] = 3
    if domain_name == "CoinGame-4":
        default_params["nr_agents"] = 4
        default_params["width"] = 5
        default_params["height"] = 5
    default_params["observation_dim"] = int(default_params["width"]*default_params["height"]*4)
    
    kwargs = {**default_params, **params}
    
    return CoinGameEnvironment(kwargs)



def vector_coin_game_make(params: dict):
    """Generates a vectorized CoinGame instance using a partial parameter dictionary."""
    env = coin_game_make(params)
    return VectorCoinGameEnvironment(env)


class VectorCoinGameEnvironment(gym.vector.VectorEnv):
    """Vectorized CoinGame environment wrapper that complies with `gym.vector.VectorEnv` API.
    """
    
    def __init__(self, params_or_env: Union[dict, CoinGameEnvironment]):
        assert isinstance(params_or_env, (dict, CoinGameEnvironment)), 'must supply either a parameter dictionary or an existing CoinGame environment instance'
        if isinstance(params_or_env, dict):
            self.env = CoinGameEnvironment(params_or_env)
        elif isinstance(params_or_env, CoinGameEnvironment):
            self.env = params_or_env
        else:
            raise NotImplementedError
        self.num_envs = self.env.nr_agents
        self.single_action_space = gym.spaces.Discrete(self.env.nr_actions)
        self.action_space = gym.spaces.MultiDiscrete([self.env.nr_actions] * self.env.nr_agents)
        self.observation_space = gym.spaces.Box(low=0, high=self.env.nr_agents, shape=(self.env.nr_agents, self.env.observation_dim), dtype='int32')
    
    @staticmethod
    def _vectorize_list(l: list[numpy.ndarray]):
        """Converts a list of numpy arrays to a single numpy array with the same data type."""
        dtype = l[0].dtype
        return numpy.array(l, dtype=dtype)

    # Explicitly wrap `reset()` to fill in missing info dictionary.
    def reset(self, *args, **kwargs):
        o = self.env.reset()
        o = self._vectorize_list(o)
        return o, {} # Fill in missing info as empty dict.

    # Explicitly wrap `step()` to fill in missing truncated flag.
    def step(self, *args, **kwargs):
        observations, rewards, done, infos = self.env.step(*args, **kwargs)
        
        # Vectorize.
        observations = self._vectorize_list(observations)
        rewards = self._vectorize_list(rewards)
        done = numpy.array([done] * self.env.nr_agents)
        truncated = numpy.array([False] * self.env.nr_agents) # Fill in missing truncated flag as static False.
        return observations, rewards, done, truncated, infos
    
    # Implicitly forward all other methods to `self.env`.
    def __getattr__(self, name):
        return getattr(self.env, name)



def episode_metrics_callback(env: CoinGameEnvironment) -> dict:
    """Computes metrics at the end of each episode for the given CoinGame environment.
    
    Returns a dictionary with the following keys:
        - `coins_collected`
        - `own_coins_collected`
        - `own_coin_rate`
        - `undiscounted_reward`
        - `discounted_reward`
    """
    
    coins_collected = env.domain_values()[env.get_index('coins_collected')]
    own_coins_collected = env.domain_values()[env.get_index('own_coins_collected')]
    undiscounted_reward = numpy.sum(env.undiscounted_returns)
    discounted_reward = numpy.sum(env.discounted_returns)
    own_coin_rate = own_coins_collected/coins_collected if coins_collected != 0 else 0
    
    return dict(
        coins_collected=coins_collected,
        own_coins_collected=own_coins_collected,
        own_coin_rate=own_coin_rate,
        undiscounted_reward=undiscounted_reward,
        discounted_reward=discounted_reward,
    )