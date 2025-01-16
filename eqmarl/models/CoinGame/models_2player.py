import tensorflow as tf
import tensorflow.keras as keras
import functools as ft
import pennylane as qml

from ...layers.keras_layers import *
# from ...layers.tfq_layers import *
from ...layers.pennylane_layers import *
from ...ops.pennylane_ops import *
from ...tools import *
# from ...observables import *


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
    wires = list(range(d_qubits))
    
    # Observables is Pauli Z of all qubits.
    # Z1, Z2, Z3, ...
    def observables_func(wires: list[int]):
        return [qml.PauliZ(wires=w) for w in wires]
    
    # Get total number of observables.
    n_observables = len(observables_func(wires=wires))

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        wires=wires,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables_func=observables_func,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        trainable_w_enc=True,
        pennylane_device='default.qubit',
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
                RescaleWeighted(n_observables),
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
    wires = list(range(n_agents * d_qubits))
    
    # Observables is joint Pauli product across all qubits.
    # Z1 @ Z2 @ Z3 @ ...
    def observables_func(wires: list[int]):
        return [ft.reduce(lambda x,y: x @ y, [qml.PauliZ(wires=w) for w in wires])]
    
    # Get total number of observables.
    n_observables = len(observables_func(wires=wires))

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        wires=wires, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables_func=observables_func,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        trainable_w_enc=True,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        pennylane_device='default.qubit',
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
                RescaleWeighted(n_observables),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model



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
    wires = list(range(d_qubits))

    # Observables is joint Pauli product across all qubits.
    # Z1 @ Z2 @ Z3 @ ...
    def observables_func(wires: list[int]):
        return [ft.reduce(lambda x,y: x @ y, [qml.PauliZ(wires=w) for w in wires])]
    
    # Get total number of observables.
    n_observables = len(observables_func(wires=wires))

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        wires=wires, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables_func=observables_func,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        trainable_w_enc=True,
        pennylane_device='default.qubit',
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
                RescaleWeighted(n_observables),
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
    wires = list(range(d_qubits))
    
    # Observables is Pauli Z of all qubits.
    # Z1, Z2, Z3, ...
    def observables_func(wires: list[int]):
        return [qml.PauliZ(wires=w) for w in wires]
    
    # Get total number of observables.
    n_observables = len(observables_func(wires=wires))

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        wires=wires, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables_func=observables_func,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        pennylane_device='default.qubit',
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
                RescaleWeighted(n_observables),
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
    wires = list(range(n_agents * d_qubits))

    # Observables is joint Pauli product across all qubits.
    # Z1 @ Z2 @ Z3 @ ...
    def observables_func(wires: list[int]):
        return [ft.reduce(lambda x,y: x @ y, [qml.PauliZ(wires=w) for w in wires])]
    
    # Get total number of observables.
    n_observables = len(observables_func(wires=wires))

    # Define quantum layer.
    qlayer = HybridPartiteVariationalEncodingPQC(
        wires=wires, 
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables_func=observables_func,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        pennylane_device='default.qubit',
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
                RescaleWeighted(n_observables),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model



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
    wires = list(range(n_agents * d_qubits))
    
    # Observables is joint Pauli product across all qubits.
    # Z1 @ Z2 @ Z3 @ ...
    def observables_func(wires: list[int]):
        return [ft.reduce(lambda x,y: x @ y, [qml.PauliZ(wires=w) for w in wires])]
    
    # Get total number of observables.
    n_observables = len(observables_func(wires=wires))

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        wires=wires, 
        d_qubits=n_agents * d_qubits,
        n_layers=n_layers,
        observables_func=observables_func,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz, # Encoder uses 3 weights per qubit.
        trainable_w_enc=trainable_w_enc,
        pennylane_device='default.qubit',
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
                RescaleWeighted(n_observables),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model
