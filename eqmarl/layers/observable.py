import tensorflow as tf
import tensorflow.keras as keras


class AlternatingObservables(keras.layers.Layer):
    """Applies alternating $(-1)^{i}$ weights to a series of observables (typically one per action in reinforcement learning, where $i$ is the action index).
    
    Inspired by: https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning#2_policy-gradient_rl_with_pqc_policies
    """
    
    def __init__(self, n_actions: int):
        super().__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(n_actions)]]),
            dtype='float32',
            trainable=True,
            name='observables-weights',
        )
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) # Multiply inputs by alternating weights.


class WeightedAlternatingObservables(AlternatingObservables):
    """Applies inverse temperature parameter $\\beta$ to alternating observables.
    
    Inspired by: https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning#2_policy-gradient_rl_with_pqc_policies
    """
    
    def __init__(self, beta: float, n_actions: int):
        super().__init__(n_actions=n_actions)
        self.beta_layer = keras.layers.Lambda(lambda x: x * beta)
        
    def call(self, inputs):
        x = super().call(inputs) # Get result from alternating observables.
        return self.beta_layer(x) # Apply inverse temperature parameter.