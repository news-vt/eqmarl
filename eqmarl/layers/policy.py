import tensorflow.keras as keras
from .observable import *


class WeightedAlternatingSoftmaxPolicy(keras.layers.Layer):
    """Softmax policy acting on weighted alternating observables using inverse temperature parameter $\\beta$.
    """
    def __init__(self, beta: float, n_actions: int):
        super().__init__()
        self.beta = beta
        self.n_actions = n_actions
        self.weighted_alternating_obs_layer = WeightedAlternatingObservables(
            beta=beta,
            n_actions=n_actions,
        )
        self.softmax_layer = keras.layers.Softmax()

    def call(self, inputs):
        x = self.weighted_alternating_obs_layer(inputs)
        return self.softmax_layer(x)