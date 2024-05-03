import tensorflow as tf
import tensorflow.keras as keras
import functools as ft

from .layers import *
from .tools import *
from .observables import *

###
# General environment classical models.
###

def generate_model_actor_classical(n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model


def generate_model_critic_classical(units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model



###
# CartPole models.
###

def generate_model_CartPole_actor_quantum(
    n_layers,
    beta = 1.0,
    squash_activation = 'linear', # linear, arctan/atan, tanh
    observables_type = 'normal', # None/normal, alternating/alt
    name = None,
    ):
    """Single-agent variant of hybrid quantum actor for CartPole.
    """
    # State boundaries for input normalization.
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole..
    obs_shape = (4,1)

    # Qubit dimension is pre-determined for CartPole environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    if observables_type is not None and observables_type.startswith('alt'):
        observables = make_observables_CartPole_alternating(qubits)
    else:
        observables = make_observables_CartPole(qubits)

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_Rx,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Lambda(lambda x: x/state_bounds), # Normalizes input states.
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                ], name="input-preprocess"),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CartPole_critic_quantum(
    n_layers,
    beta = 1.0,
    squash_activation = 'arctan', # linear, arctan/atan, tanh
    name = None,
    ):
    """Single-agent variant of hybrid quantum actor for CartPole.
    """
    # State boundaries for input normalization.
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole..
    obs_shape = (4,1)

    # Qubit dimension is pre-determined for CartPole environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    observables = [
        cirq.Z(qubits[0]) * cirq.Z(qubits[1]) * cirq.Z(qubits[2]) * cirq.Z(qubits[3]),
        ]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_Rx,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Lambda(lambda x: x/state_bounds), # Normalizes input states.
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                ], name="input-preprocess"),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model


def generate_model_CartPole_actor_quantum_partite(
    n_agents,
    n_layers,
    beta = 1.0,
    squash_activation = 'linear', # linear, arctan/atan, tanh
    observables_type = 'normal', # None/normal, alternating/alt
    name=None,
    ):
    """eQMARL variant of hybrid quantum actor for CartPole.
    """
    # State boundaries for input normalization.
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole..
    obs_shape = (4,1)

    # Qubit dimension is pre-determined for CartPole environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)
    
    # Set observable generation function.
    if observables_type is not None and observables_type.startswith('alt'):
        observables_func = make_observables_CartPole_alternating
    else:
        observables_func = make_observables_CartPole

    # Generate observables.
    agent_obs = []
    for aidx in range(n_agents):
        qidx = aidx * d_qubits # Starting qubit index for the current partition.
        obs = observables_func(qubits[qidx:qidx+d_qubits])
        agent_obs.append(obs)
    observables = permute_observables(agent_obs) # Permute all combinations of agent observables.

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        # squash_activation='tanh',
        # squash_activation='arctan',
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_Rx,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Lambda(lambda x: x/state_bounds), # Normalizes input states.
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to agent observations.
                ], name="input-preprocess"),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


# MARK: CoinGame2 models.
###
# CoinGame2 models.
###


# Observation shape for "CoinGame-2" is (4,3,3) which means:
# - index=0: 3x3 grid world with a `1` where the current agent is.
# - index=1: 3x3 grid world where a `1` is added to every cell that has other agents.
# - index=2: 3x3 grid world with a `1` for location of coin that matches the focused agent's color.
# - index=3: 3x3 grid world with a `1` for location of coin that matches other agent's color.
def filter_CoinGame2_obs_feature_dims(obs: tf.Tensor, keepdims: list[int]) -> tf.Tensor:
    """Removes CoinGame2 observation feature dimension(s).
    
    This is useful for converting the default MDP state into a POMDP.
    
    Assumes `obs.shape` is either (4,3,3) or (n_agents,4,3,3).
    
    Observation shape for "CoinGame-2" is (4,3,3) which means:
    - index=0: 3x3 grid world with a `1` where the current agent is.
    - index=1: 3x3 grid world where a `1` is added to every cell that has other agents.
    - index=2: 3x3 grid world with a `1` for location of coin that matches the focused agent's color.
    - index=3: 3x3 grid world with a `1` for location of coin that matches other agent's color.
    
    Converts (n_agents,4,3,3) to (n_agents,3,3,3) by removing the index `i=1` from the second feature dimension.
    """
    assert len(obs.shape) >=3, 'observation shape must either be (4,3,3) or (n_agents,4,3,3)'
    splits = tf.split(obs, obs.shape[-3], axis=-3)
    t = tf.stack([splits[dim] for dim in keepdims], axis=-4)
    t = tf.squeeze(t, axis=-3) # Only keep indexes [0,2,3].
    return t


def generate_model_CoinGame2_actor_classical_shared_mdp(n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame2_critic_classical_joint_mdp(n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,-1))]
    layers += [keras.layers.LocallyConnected1D(u, kernel_size=1, activation=activation) for u in units]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame2_critic_classical_joint_mdp_central(n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()] # Flatten all inputs.
    layers += [keras.layers.Dense(u, activation=activation) for u in units] # Central branch dense layers.
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model


def generate_model_CoinGame2_actor_classical_shared_pomdp(keepdims: list[int], n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((4,3,3))]
    layers += [keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame2_critic_classical_joint_pomdp(keepdims: list[int], n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,4,3,3))]
    layers += [keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Reshape((n_agents,-1))]
    layers += [keras.layers.LocallyConnected1D(u, kernel_size=1, activation=activation) for u in units]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame2_critic_classical_joint_pomdp_central(keepdims: list[int], n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,4,3,3))]
    layers += [keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Flatten()] # Flatten all inputs.
    layers += [keras.layers.Dense(u, activation=activation) for u in units] # Central branch dense layers.
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model



def map_CoinGame2_obs_to_encoded_vector(obs: tf.Tensor) -> tf.Tensor:
    """Reduces last dimension of CoinGame2 observation into single number, where the value is the sum of fractional powers of 2 that represents the column within the game grid.
    
    For example, converts an observation of shape (A,B,C) to (A,B) where the last dimension is reduced using the function $\sum_{i=0}^{C-1} obs[...,i] 2^{-i}$.
    """
    i = -tf.range(obs.shape[-1], dtype=obs.dtype) # Fractional power of 2 that represents the column within the grid.
    return tf.math.reduce_sum(obs * (2**i), axis=-1)


def generate_model_CoinGame2_actor_quantum_shared_mdp(
    n_layers: int,
    squash_activation: str = 'arctan',
    beta: float = 1.0,
    name: str = None,
    ):
    """Single-agent variant of hybrid quantum actor for CoinGame.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Qubit dimension is pre-determined for CoinGame2 environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    observables = [
        cirq.Z(qubits[0]),
        cirq.Z(qubits[1]),
        cirq.Z(qubits[2]),
        cirq.Z(qubits[3]),
    ]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: map_CoinGame2_obs_to_encoded_vector(x)), # converts (4,3,3) into (4,3)
                ], name="input-preprocess"),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CoinGame2_critic_quantum_partite_mdp(
    n_agents: int,
    n_layers: int,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    beta: float = 1.0,
    name: str = None,
    input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
    input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
    ):
    """eQMARL variant of hybrid joint quantum critic for CoinGame.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Qubit dimension is pre-determined for CoinGame2 environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: map_CoinGame2_obs_to_encoded_vector(x)), # converts (n_agents,4,3,3) into (n_agents,4,3)
                ], name="input-preprocess"),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model


###################
def generate_model_CoinGame2_critic_quantum_central_mdp(
    n_agents: int,
    n_layers: int,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    beta: float = 1.0,
    name: str = None,
    ):
    """Centralized variant of hybrid joint quantum critic for CoinGame.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Qubit dimension is pre-determined for CoinGame2 environment.
    # Using `4` to match observable dimension.
    d_qubits = 4 * n_agents

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents*obs_shape[0], *obs_shape[1:])), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: map_CoinGame2_obs_to_encoded_vector(x)), # converts (n_agents*4,3,3) into (n_agents*4,3)
                ], name="input-preprocess"),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model
###################



def generate_model_CoinGame2_actor_quantum_shared_pomdp(
    keepdims: list[int],
    n_layers: int,
    squash_activation: str = 'arctan',
    beta: float = 1.0,
    name: str = None,
    ):
    """Single-agent variant of hybrid quantum actor for CoinGame2.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)
    d_qubits = 3

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    observables = [cirq.Z(q) for q in qubits]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
                keras.layers.Lambda(lambda x: map_CoinGame2_obs_to_encoded_vector(x)), # Fractional power of 2.
                ], name="input-preprocess"),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CoinGame2_critic_quantum_partite_pomdp(
    keepdims: list[int],
    n_agents: int,
    n_layers: int,
    beta: float = 1.0,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    name: str = None,
    input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
    input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
    ):
    """eQMARL variant of hybrid joint quantum critic for CoinGame2.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)
    d_qubits = 3

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
                keras.layers.Lambda(lambda x: map_CoinGame2_obs_to_encoded_vector(x)), # Fractional power of 2.
                ], name="input-preprocess"),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model



def generate_model_CoinGame2_actor_quantum_nnreduce_shared_pomdp(
    d_qubits: int,
    keepdims: list[int],
    n_layers: int,
    squash_activation: str = 'arctan',
    beta: float = 1.0,
    name: str = None,
    nn_activation: str = 'linear',
    trainable_w_enc: bool = True,
    ):
    """Single-agent variant of hybrid quantum actor for CoinGame2.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    observables = [cirq.Z(q) for q in qubits]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
                ], name="input-preprocess"),
            keras.layers.Flatten(), # Convert to batched 1D.
            keras.layers.Dense(d_qubits * 3, activation=nn_activation), # Reduce last feature dimension to (3,3).
            keras.layers.Reshape((d_qubits,3)),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CoinGame2_critic_quantum_nnreduce_partite_pomdp(
    d_qubits: int,
    keepdims: list[int],
    n_agents: int,
    n_layers: int,
    beta: float = 1.0,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    name: str = None,
    nn_activation: str = 'linear',
    trainable_w_enc: bool = True,
    input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
    input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
    ):
    """eQMARL variant of hybrid joint quantum critic for CoinGame2.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
                ], name="input-preprocess"),
            keras.layers.Reshape((n_agents,-1)), # Convert (n_agents,3,3,3) to (n_agents,3,9).
            keras.layers.LocallyConnected1D(d_qubits * 3, kernel_size=1, activation=nn_activation), # Reduce last feature dimension to (n_agents,3,3), use local dense units only to separate agents.
            keras.layers.Reshape((n_agents,d_qubits,3)),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model


#######################
def generate_model_CoinGame2_critic_quantum_nnreduce_central_pomdp(
    d_qubits: int,
    keepdims: list[int],
    n_agents: int,
    n_layers: int,
    beta: float = 1.0,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    name: str = None,
    nn_activation: str = 'linear',
    trainable_w_enc: bool = True,
    ):
    """Central variant of hybrid joint quantum critic for CoinGame2.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=n_agents * d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: filter_CoinGame2_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
                ], name="input-preprocess"),
            keras.layers.Reshape((n_agents,-1)),
            keras.layers.LocallyConnected1D(d_qubits * 3, kernel_size=1, activation=nn_activation), # Reduce last feature dimension to (n_agents,3,3), use local dense units only to separate agents.
            keras.layers.Reshape((n_agents*d_qubits,3)),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model
#######################


# MARK: CoinGame4 models.
###
# CoinGame4 models.
###


# Observation shape for "CoinGame-4" is (4,5,5) which means:
# - index=0: 5x5 grid world with a `1` where the current agent is.
# - index=1: 5x5 grid world where a `1` is added to every cell that has other agents.
# - index=2: 5x5 grid world with a `1` for location of coin that matches the focused agent's color.
# - index=3: 5x5 grid world with a `1` for location of coin that matches other agent's color.
def filter_CoinGame4_obs_feature_dims(obs: tf.Tensor, keepdims: list[int]) -> tf.Tensor:
    """Removes CoinGame4 observation feature dimension(s).
    
    This is useful for converting the default MDP state into a POMDP.
    
    Assumes `obs.shape` is either (4,5,5) or (n_agents,4,5,5).
    
    Observation shape for "CoinGame-2" is (4,5,5) which means:
    - index=0: 5x5 grid world with a `1` where the current agent is.
    - index=1: 5x5 grid world where a `1` is added to every cell that has other agents.
    - index=2: 5x5 grid world with a `1` for location of coin that matches the focused agent's color.
    - index=3: 5x5 grid world with a `1` for location of coin that matches other agent's color.
    
    Converts (n_agents,4,5,5) to (n_agents,3,5,5) by removing the index `i=1` from the second feature dimension.
    """
    assert len(obs.shape) >=3, 'observation shape must either be (4,5,5) or (n_agents,4,5,5)'
    splits = tf.split(obs, obs.shape[-3], axis=-3)
    t = tf.stack([splits[dim] for dim in keepdims], axis=-4)
    t = tf.squeeze(t, axis=-3) # Only keep indexes [0,2,3].
    return t



def generate_model_CoinGame4_actor_quantum_nnreduce_shared_pomdp(
    d_qubits: int,
    keepdims: list[int],
    n_layers: int,
    squash_activation: str = 'arctan',
    beta: float = 1.0,
    name: str = None,
    nn_activation: str = 'linear',
    trainable_w_enc: bool = True,
    ):
    """Single-agent variant of hybrid quantum actor for CoinGame4.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,5,5)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    observables = [cirq.Z(q) for q in qubits]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: filter_CoinGame4_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
                ], name="input-preprocess"),
            keras.layers.Flatten(), # Convert to batched 1D.
            keras.layers.Dense(d_qubits * 3, activation=nn_activation), # Reduce last feature dimension to (d_qubits,3), where `3` is the number of rotations per qubit.
            keras.layers.Reshape((d_qubits,3)),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CoinGame4_critic_quantum_nnreduce_partite_pomdp(
    d_qubits: int,
    keepdims: list[int],
    n_agents: int,
    n_layers: int,
    beta: float = 1.0,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    name: str = None,
    nn_activation: str = 'linear',
    trainable_w_enc: bool = True,
    input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
    input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
    ):
    """eQMARL variant of hybrid joint quantum critic for CoinGame4.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,5,5)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: filter_CoinGame4_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
                ], name="input-preprocess"),
            keras.layers.Reshape((n_agents,-1)), # Convert (n_agents,3,5,5) to (n_agents,75).
            keras.layers.LocallyConnected1D(d_qubits * 3, kernel_size=1, activation=nn_activation), # Reduce last feature dimension to (d_qubits,3), where `3` is the number of rotations per qubit.
            keras.layers.Reshape((n_agents,d_qubits,3)),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model


def generate_model_CoinGame4_actor_quantum_shared_mdp(
    d_qubits: int,
    n_layers: int,
    squash_activation: str = 'arctan',
    beta: float = 1.0,
    name: str = None,
    trainable_w_enc: bool = True,
    nn_activation: str = 'linear',
    ):
    """Single-agent variant of hybrid quantum actor for CoinGame4.
    """

    # Shape of observables is already known for CoinGame4.
    obs_shape = (4,5,5)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    observables = [cirq.Z(q) for q in qubits]

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                ], name="input-preprocess"),
            keras.layers.Flatten(), # Convert to batched 1D.
            keras.layers.Dense(d_qubits * 3, activation=nn_activation), # Reduce last feature dimension to (d_qubits,3), where `3` is the number of rotations per qubit.
            keras.layers.Reshape((d_qubits,3)),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CoinGame4_critic_quantum_partite_mdp(
    d_qubits: int,
    n_agents: int,
    n_layers: int,
    beta: float = 1.0,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    name: str = None,
    nn_activation: str = 'linear',
    trainable_w_enc: bool = True,
    input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
    input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
    ):
    """eQMARL variant of hybrid joint quantum critic for CoinGame4.
    """

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,5,5)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                ], name="input-preprocess"),
            keras.layers.Reshape((n_agents,-1)), # Convert (n_agents,4,5,5) to (n_agents,100).
            keras.layers.LocallyConnected1D(d_qubits * 3, kernel_size=1, activation=nn_activation), # Reduce last feature dimension to (d_qubits,3), where `3` is the number of rotations per qubit.
            keras.layers.Reshape((n_agents,d_qubits,3)),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model



def generate_model_CoinGame4_actor_classical_shared_mdp(n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame4_critic_classical_joint_mdp(n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,-1))]
    layers += [keras.layers.LocallyConnected1D(u, kernel_size=1, activation=activation) for u in units]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame4_critic_classical_joint_mdp_central(n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Flatten()] # Flatten all inputs.
    layers += [keras.layers.Dense(u, activation=activation) for u in units] # Central branch dense layers.
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model






def generate_model_CoinGame4_actor_classical_shared_pomdp(keepdims: list[int], n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((4,5,5))]
    layers += [keras.layers.Lambda(lambda x: filter_CoinGame4_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame4_critic_classical_joint_pomdp(keepdims: list[int], n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,4,5,5))]
    layers += [keras.layers.Lambda(lambda x: filter_CoinGame4_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Reshape((n_agents,-1))]
    layers += [keras.layers.LocallyConnected1D(u, kernel_size=1, activation=activation) for u in units]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CoinGame4_critic_classical_joint_pomdp_central(keepdims: list[int], n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,4,5,5))]
    layers += [keras.layers.Lambda(lambda x: filter_CoinGame4_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Flatten()] # Flatten all inputs.
    layers += [keras.layers.Dense(u, activation=activation) for u in units] # Central branch dense layers.
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model





# MARK: OLD METHODS
########## OLD METHODS


def generate_model_CoinGame2_actor_quantum(
    n_layers,
    beta = 1.0,
    name=None,
    ):
    """Single-agent variant of hybrid quantum actor for CoinGame.
    """
    
    def map_observable_to_vector(obs: tf.Tensor) -> tf.Tensor:
        """Converts an observable with shape `(...,x,y)` into `(...,x)` where the final dimension `y` is represented as a `sum({grid_y_val} * 2^{grid_y_length - grid_y_index})` for every column in all rows."""
        b = 2**tf.range(obs.shape[-1], dtype=obs.dtype) # Power of 2 that represents the column within the grid.
        return tf.tensordot(obs[...,::-1], b[::-1], 1)

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Qubit dimension is pre-determined for CoinGame2 environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Generate observables.
    observables = make_observables_CoinGame2(qubits)

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        qubits=qubits, 
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
            keras.Input(shape=(input_size,), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((*obs_shape,)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: map_observable_to_vector(x)), # converts (4,3,3) into (4,3)
                ], name="input-preprocess"),
            qlayer, # Hybrid quantum layer.
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CoinGame2_actor_quantum_partite(
    n_agents,
    n_layers,
    beta = 1.0,
    name=None,
    ):
    """eQMARL variant of hybrid quantum actor for CoinGame.
    """
    
    def map_observable_to_vector(obs: tf.Tensor) -> tf.Tensor:
        """Converts an observable with shape `(...,x,y)` into `(...,x)` where the final dimension `y` is represented as a `sum({grid_y_val} * 2^{grid_y_length - grid_y_index})` for every column in all rows."""
        b = 2**tf.range(obs.shape[-1], dtype=obs.dtype) # Power of 2 that represents the column within the grid.
        return tf.tensordot(obs[...,::-1], b[::-1], 1)

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Qubit dimension is pre-determined for CoinGame2 environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Generate observables.
    agent_obs = []
    for aidx in range(n_agents):
        qidx = aidx * d_qubits # Starting qubit index for the current partition.
        obs = make_observables_CoinGame2(qubits[qidx:qidx+d_qubits])
        agent_obs.append(obs)
    observables = permute_observables(agent_obs) # Permute all combinations of agent observables.

    # Define quantum layer.
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

def generate_model_CoinGame2_critic_quantum_partite(
    n_agents,
    n_layers,
    beta = 1.0,
    squash_activation = 'linear', # linear, arctan/atan, tanh
    name=None,
    ):
    """eQMARL variant of hybrid joint quantum critic for CoinGame.
    """
    
    def map_observable_to_vector(obs: tf.Tensor) -> tf.Tensor:
        """Converts an observable with shape `(...,x,y)` into `(...,x)` where the final dimension `y` is represented as a `sum({grid_y_val} * 2^{grid_y_length - grid_y_index})` for every column in all rows."""
        b = 2**tf.range(obs.shape[-1], dtype=obs.dtype) # Power of 2 that represents the column within the grid.
        return tf.tensordot(obs[...,::-1], b[::-1], 1)

    # Shape of observables is already known for CoinGame2.
    obs_shape = (4,3,3)

    # Qubit dimension is pre-determined for CoinGame2 environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(n_agents * d_qubits)

    # Observables is joint Pauli product across all qubits.
    observables = [ft.reduce(lambda x,y: x*y, [cirq.Z(q) for q in qubits])]

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: map_observable_to_vector(x)), # converts (n_agents,4,3,3) into (n_agents,4,3)
                ], name="input-preprocess"),
            qlayer,
            keras.Sequential([
                RescaleWeighted(len(observables)),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model


def generate_model_CoinGame2_qlearning_quantum_partite(
    qubits,
    n_agents,
    d_qubits,
    n_feat, # 1D Feature dimension of observable.
    n_layers,
    observables,
    is_target,
    ):
    """eQMARL variant of Q-learning agent for CoinGame.
    """
    
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