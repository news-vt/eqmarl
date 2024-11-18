"""General purpose Keras layers."""

import tensorflow as tf
import tensorflow.keras as keras


class Weighted(keras.layers.Layer):
    """Learnable weighting.
    
    Applies a learned weighting to input observables in range [-1, 1].
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)),
            dtype='float32',
            trainable=True,
            name='obs-weights',
            )

    def call(self, inputs):
        return tf.math.multiply(
            inputs,
            tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0),
            )


class RescaleWeighted(keras.layers.Layer):
    """Learnable rescaling from range [-1, 1] to range [0, 1]."""
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)),
            dtype='float32',
            trainable=True,
            name='obs-weights',
            )

    def call(self, inputs):
        return tf.math.multiply(
            (1+inputs)/2., # Rescale from [-1, 1] to range [0, 1].
            tf.repeat(self.w, repeats=tf.shape(inputs)[0], axis=0),
            )
