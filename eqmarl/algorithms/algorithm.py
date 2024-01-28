import gymnasium as gym


class Algorithm:

    def __init__(self, env: gym.Env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env

    def train(self, *args, **kwargs) -> dict[list, dict]:
        raise NotImplementedError