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



# MARK: CartPole
###
# Vectorized CartPole models.
###

def filter_CartPole_obs_feature_dims(obs: tf.Tensor, keepdims: list[int]) -> tf.Tensor:
    """Removes CartPole observation feature dimension(s).
    
    This is useful for converting the default MDP state into a POMDP.
    
    Assumes `obs.shape` is either (4,1) or (n_agents,4,1).
    """
    assert len(obs.shape) >= 2, 'observation shape must either be (4,1) or (n_agents,4,1)'
    splits = tf.split(obs, obs.shape[-2], axis=-2)
    t = tf.stack([splits[dim] for dim in keepdims], axis=-3)
    t = tf.squeeze(t, axis=-2) # Only keep indexes [0,2,3].
    return t

def generate_model_CartPole_actor_classical_shared_mdp(n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Lambda(lambda x: x/state_bounds)] # Normalizes input states.]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model


def generate_model_CartPole_critic_classical_joint_mdp_central(n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Lambda(lambda x: x/state_bounds)] # Normalizes input states.]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CartPole_critic_classical_joint_mdp(n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Lambda(lambda x: x/state_bounds)] # Normalizes input states.]
    layers += [keras.layers.Reshape((n_agents,-1))]
    layers += [keras.layers.LocallyConnected1D(u, kernel_size=1, activation=activation) for u in units]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CartPole_actor_quantum_shared_mdp(
    n_layers: int,
    squash_activation: str = 'arctan',
    beta: float = 1.0,
    name: str = None,
    ):
    """Single-agent variant of hybrid quantum actor for CoinGame.
    """
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole.
    obs_shape = (4,1)

    # Qubit dimension is pre-determined for CartPole environment.
    # Using `4` to match observable dimension.
    d_qubits = 4

    # Create qubit list using qubit dimensions.
    wires = list(range(d_qubits))
    
    # Generate observables.
    # Alternating observable (one is negative of the other).
    # Observables are:
    # 1. Z0 * Z1 * Z2 * Z3
    # 2. -(Z0 * Z1 * Z2 * Z3)
    def observables_func(wires: list[int]):
        return [
            qml.PauliZ(wires=wires[0]) @ qml.PauliZ(wires=wires[1]) @ qml.PauliZ(wires=wires[2]) @ qml.PauliZ(wires=wires[3]),
            -qml.PauliZ(wires=wires[0]) @ qml.PauliZ(wires=wires[1]) @ qml.PauliZ(wires=wires[2]) @ qml.PauliZ(wires=wires[3]),
        ]
    
    # Get total number of observables.
    n_observables = len(observables_func(wires=wires))

    # Define quantum layer.
    qlayer = HybridVariationalEncodingPQC(
        wires=wires, 
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables_func=observables_func,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_Rx, # only 1 feature per dimension, so use only a single rotation.
        trainable_w_enc=True,
        pennylane_device='default.qubit',
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
                RescaleWeighted(n_observables),
                keras.layers.Lambda(lambda x: x * beta),
                keras.layers.Softmax(),
                ], name='observables-policy')
        ], name=name)
    return model


def generate_model_CartPole_critic_quantum_central_mdp(
    n_agents: int,
    n_layers: int,
    squash_activation: str = 'arctan', # linear, arctan/atan, tanh
    beta: float = 1.0,
    name: str = None,
    ):
    """Centralized variant of hybrid joint quantum critic for CoinGame.
    """
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole.
    obs_shape = (4,1)

    # Qubit dimension is pre-determined for CartPole environment.
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
        encoding_layer_cls=ParameterizedRotationLayer_Rx,
        trainable_w_enc=True,
        pennylane_device='default.qubit',
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Lambda(lambda x: x/state_bounds), # Normalizes input states.
                keras.layers.Reshape((n_agents*obs_shape[0], *obs_shape[1:])), # Reshape to matrix grid (n_agents*4,1)
                ], name="input-preprocess"),
            qlayer,
            keras.Sequential([
                RescaleWeighted(n_observables),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model

def generate_model_CartPole_critic_quantum_partite_mdp(
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
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole.
    obs_shape = (4,1)

    # Qubit dimension is pre-determined for CartPole environment.
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
        encoding_layer_cls=ParameterizedRotationLayer_Rx,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
        trainable_w_enc=True,
        pennylane_device='default.qubit',
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Lambda(lambda x: x/state_bounds), # Normalizes input states.
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                ], name="input-preprocess"),
            qlayer,
            keras.Sequential([
                RescaleWeighted(n_observables),
                keras.layers.Lambda(lambda x: x * beta),
                ], name='observables-value')
        ], name=name)
    return model


def generate_model_CartPole_actor_classical_shared_pomdp(keepdims: list[int], n_actions: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((4,1))]
    layers += [keras.layers.Lambda(lambda x: x/state_bounds)] # Normalizes input states.]
    layers += [keras.layers.Lambda(lambda x: filter_CartPole_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(n_actions, activation='softmax', name='policy')] # Policy estimation pi(a|s)
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CartPole_critic_classical_joint_pomdp_central(keepdims: list[int], n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,4,1))]
    layers += [keras.layers.Lambda(lambda x: x/state_bounds)] # Normalizes input states.]
    layers += [keras.layers.Lambda(lambda x: filter_CartPole_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Flatten()] # Flatten all inputs.
    layers += [keras.layers.Dense(u, activation=activation) for u in units] # Central branch dense layers.
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model

def generate_model_CartPole_critic_classical_joint_pomdp(keepdims: list[int], n_agents: int, units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))
    assert type(units) == list, 'units must be a list of integers'
    layers = []
    layers += [keras.layers.Reshape((n_agents,4,1))]
    layers += [keras.layers.Lambda(lambda x: x/state_bounds)] # Normalizes input states.]
    layers += [keras.layers.Lambda(lambda x: filter_CartPole_obs_feature_dims(x, keepdims=keepdims))]
    layers += [keras.layers.Reshape((n_agents,-1))]
    layers += [keras.layers.LocallyConnected1D(u, kernel_size=1, activation=activation) for u in units]
    layers += [keras.layers.Flatten()]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model


def generate_model_CartPole_actor_quantum_nnreduce_shared_pomdp(
    d_qubits: int,
    keepdims: list[int],
    n_layers: int,
    squash_activation: str = 'arctan',
    beta: float = 1.0,
    name: str = None,
    nn_activation: str = 'linear',
    trainable_w_enc: bool = True,
    ):
    """Single-agent variant of hybrid quantum actor for CartPole.
    """
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole.
    obs_shape = (4,1)

    # Create qubit list using qubit dimensions.
    qubits = cirq.LineQubit.range(d_qubits)
    
    # Create qubit list using qubit dimensions.
    wires = list(range(d_qubits))
    
    # Generate observables.
    # Alternating observable (one is negative of the other).
    # Observables are:
    # 1. Z0 * Z1 * Z2 * Z3
    # 2. -(Z0 * Z1 * Z2 * Z3)
    def observables_func(wires: list[int]):
        return [
            qml.PauliZ(wires=wires[0]) @ qml.PauliZ(wires=wires[1]) @ qml.PauliZ(wires=wires[2]) @ qml.PauliZ(wires=wires[3]),
            -qml.PauliZ(wires=wires[0]) @ qml.PauliZ(wires=wires[1]) @ qml.PauliZ(wires=wires[2]) @ qml.PauliZ(wires=wires[3]),
        ]
    
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
                keras.layers.Lambda(lambda x: x/state_bounds),
                keras.layers.Lambda(lambda x: filter_CartPole_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
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


def generate_model_CartPole_critic_quantum_nnreduce_central_pomdp(
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
    """Central variant of hybrid joint quantum critic for CartPole.
    """
    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole.
    obs_shape = (4,1)

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
                keras.layers.Lambda(lambda x: x/state_bounds),
                keras.layers.Lambda(lambda x: filter_CartPole_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
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


def generate_model_CartPole_critic_quantum_nnreduce_partite_pomdp(
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
    """eQMARL variant of hybrid joint quantum critic for CartPole.
    """

    state_bounds = tf.convert_to_tensor(np.array([2.4, 2.5, 0.21, 2.5], dtype='float32'))

    # Shape of observables is already known for CartPole.
    obs_shape = (4,1)

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
        pennylane_device='default.qubit'
        )
    
    # Raw observations are given as a 1D list, so convert matrix shape into list size.
    input_size = ft.reduce(lambda x, y: x*y, obs_shape)

    model = keras.Sequential([
            keras.Input(shape=(n_agents, input_size), dtype=tf.dtypes.float32, name='input'), # Shape of model input, which should match the observation vector shape.
            keras.Sequential([
                keras.layers.Reshape((n_agents, *obs_shape)), # Reshape to matrix grid.
                keras.layers.Lambda(lambda x: x/state_bounds),
                keras.layers.Lambda(lambda x: filter_CartPole_obs_feature_dims(x, keepdims=keepdims)), # Convert observable to POMDP.
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

