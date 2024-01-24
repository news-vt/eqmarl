import tensorflow as tf
import tensorflow.keras as keras
import functools as ft

from .layers import *



def generate_model_CoinGame2_actor_classical(n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model


def generate_model_CoinGame2_critic_classical(units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model


def generate_model_CoinGame2_actor_quantum(
    qubits,
    n_agents,
    d_qubits,
    obs_shape, # i.e., (4,3,3)
    n_layers,
    observables,
    beta = 1.0,
    name=None,
    ):
    
    def map_observable_to_vector(obs: tf.Tensor) -> tf.Tensor:
        """Converts an observable with shape `(...,x,y)` into `(...,x)` where the final dimension `y` is represented as a `sum({grid_y_val} * 2^{grid_y_length - grid_y_index})` for every column in all rows."""
        b = 2**tf.range(obs.shape[-1], dtype=obs.dtype) # Power of 2 that represents the column within the grid.
        return tf.tensordot(obs[...,::-1], b[::-1], 1)

    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        # squash_activation='tanh',
        squash_activation='arctan',
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            # keras.Input(shape=(n_agents, d_qubits, n_feat), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            # keras.Input(shape=(n_agents, *obs_shape), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: map_observable_to_vector(x)), # converts (n_agents,4,3,3) into (n_agents,4,3)
                ], name="input-preprocess"),
            qlayer,
            # CustomQuantumLayer(n_wires, n_layers, observables),
            # keras.layers.Lambda(lambda x: tf.math.abs(x)), # Convert complex to float via abs.
            # keras.layers.Activation('tanh'), # Ensure outputs of PQC are in range [-1, 1].
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
            # keras.Sequential([Rescaling(len(observables))], name=is_target*'Target'+'Q-values'),
        ], name=name)
    return model


def generate_model_CoinGame2_qlearning_quantum(
    qubits,
    n_agents,
    d_qubits,
    n_feat, # 1D Feature dimension of observable.
    n_layers,
    observables,
    is_target,
    ):
    
    # n_wires = len(qubits)
    
    # qubits = cirq.LineQubit.range(n_wires)
    
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        # squash_activation='tanh',
        squash_activation='arctan',
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        )

    model = keras.Sequential([
            keras.Input(shape=(n_agents, d_qubits, n_feat), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            qlayer,
            # CustomQuantumLayer(n_wires, n_layers, observables),
            # keras.layers.Lambda(lambda x: tf.math.abs(x)), # Convert complex to float via abs.
            # keras.layers.Activation('tanh'), # Ensure outputs of PQC are in range [-1, 1].
            keras.Sequential([RescaleWeighted(len(observables))], name=is_target*'Target'+'Q-values'),
        ],
        )
    return model