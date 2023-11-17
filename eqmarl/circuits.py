from typing import Optional
import pennylane as qml
from pennylane import numpy as np
from numpy.typing import NDArray

from .observables import *
from .ops import *
from .types import WireListType

# Attempt to import TensorFlow.
try:
    import tensorflow as tf
except ImportError:
    pass


class QuantumCircuit:
    """Base class for quantum circuits intended to be used with PennyLane API.
    
    This class allows for custom circuits to be created with support for state variables (such as `self.wires` and `self.n_wires`).
    The addition of state allows for reduced parameters when calling the class instance like a function.
    
    Example usage is:
    >>> wires = list(range(4))
    >>> circuit = Circuit(wires=wires)
    >>> dev = qml.device('default.qubit', wires=len(wires))
    >>> qnode = qml.QNode(func=circuit, device=dev)
    >>> qnode()
    
    Or can create a qnode directly using the class methods:
    >>> wires = list(range(4))
    >>> circuit = Circuit(wires=wires)
    >>> dev = circuit.device('default.qubit')
    >>> qnode = circuit.qnode(device=dev)
    >>> qnode()
    """
    
    def __init__(self, 
        wires: int | list[int],
        observables: list | Callable[[list], list] | None = None,
        ):
        if isinstance(wires, int):
            self.wires = list(range(wires))
        elif isinstance(wires, (list, tuple)):
            self.wires = wires
        else:
            raise ValueError(f"Wires must either be an integer, list, or tuple; got {wires}")
        self.n_wires = len(self.wires)
        
        self.set_observables(observables=observables)
    
    def __call__(self, inputs = None):
        """Construct the circuit and pass parameters.
        
        This function must abide by the constraints for all PennyLane quantum functions (see https://docs.pennylane.ai/en/stable/introduction/circuits.html#quantum-functions).
        
        The argument `inputs` is required for compatibility with with `KerasLayer` or `TorchLayer`. The default `None` value allows this class to be used without specifying inputs (if designed with proper error handling). The index of `inputs` within the argument sequence does not matter, just as long as the name is reserved.
        """
        raise NotImplementedError()
    
    def measure(self) -> list | None:
        """Returns list of expectations across all observables."""
        if self.observables is not None:
            return [qml.expval(o) for o in self.observables]

    def device(self, *args, **kwargs):
        """Helper for creating a `qml.device` object from the current circuit."""
        return qml.device(*args, wires=self.wires, **kwargs)
    
    def qnode(self, *args, device: qml.Device | str | dict = None, **kwargs):
        """Helper for creating a `qml.QNode` object from the current circuit.
        
        The `device` argument can either be a dictionary
        """
        # Create a basic default device if none was specified.
        if device is None:
            device = self.device(name='default.qubit')
        # Create device from name string.
        elif isinstance(device, str):
            device = self.device(name=device)
        # Create device from keyword arguments.
        elif isinstance(device, dict):
            device = self.device(**device)
        # Default case is to treat device as a `qml.Device` object.
        assert list(device.wires) == list(self.wires), 'Circuit wires must match device wires'

        # Create quantum node using the current circuit as a callable.
        kwargs = dict(**kwargs, device=device) # Update kwargs to include device.
        func: Callable = self.__call__
        func.__func__.__name__ = self.__class__.__name__ # Duck type the class name as the function name.
        return qml.QNode(func, *args, **kwargs)

    @property
    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        """Returns a dictionary of shapes corresponding to trainable weights, the keys in the dictionary match the names of the parameters in the `__call__` function.
        
        The format of the dictionary should match that accepted by `qml.qnn.KerasLayer`.
        """
        return {} # Default returns an empty dictionary (i.e., no trainable weights).

    @property
    def weight_specs(self) -> dict[str, dict]:
        """Returns a nested dictionary of arguments to initialize trainable weights, the keys in the dictionary match the names of the parameters in the `__call__` function.
        
        The format of the dictionary should match that accepted by `qml.qnn.KerasLayer.weight_specs` argument, and the Keras layers `add_weight()` function (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_weight).
        """
        return {}

    @property
    def output_shape(self) -> tuple[int,...]:
        """Returns number of observables at output.
        
        This is useful in combination with `qml.KerasLayer`.
        
        Note 1: The returned shape does not include the batch dimension.
        """
        if self.observables is not None:
            return (len(self.observables),)

    @property
    def input_shape(self) -> tuple[int,...]:
        """Returns required shape for `inputs` argument.

        Note 1: The returned shape does not include the batch dimension.
        Note 2: PennyLane will compress inputs to 2D when batching for usage with `KerasLayer` or `TorchLayer`.
        """
        raise NotImplementedError()
    
    @property
    def shape(self) -> tuple[int,...] | tuple[tuple[int,...],...]:
        """Returns shape of parameters for circuit, or tuple of shapes if multiple parameters exist."""
        raise NotImplementedError()

    @staticmethod
    def get_shape(*args, **kwargs) -> tuple[int,...] | tuple[tuple[int,...],...]:
        """Returns shape of parameters for circuit, or tuple of shapes if multiple parameters exist."""
        raise NotImplementedError()

    def set_observables(self, observables: list | Callable[[list], list] | None = None):
        """Allows setting observables after circuit has been created."""
        if observables is None:
            self.observables = None
        elif hasattr(observables, '__iter__') and not isinstance(observables, str):
            self.observables = np.asarray(observables).tolist() # Ensure type is a Python list.
        elif isinstance(observables, Callable):
            self.observables = np.asarray(observables(self.wires)).tolist() # Ensure type is a Python list.
        else:
            raise ValueError(f"observables must either be a list or function; got `{observables}`")
    
    def get_keras_layer(self, **kwargs):
        """Constructs a `qml.qnn.KerasLayer` instance from the circuit using existing parameter shapes, specs, and output dimension.
        
        Keyword arguments are passed directly to `qml.qnn.KerasLayer.__init__`.
        """
        # Create a default QNode if none was given.
        qnode = kwargs.pop('qnode', self.qnode())
        
        # Wrap keyword arguments into single dictionary to prevent duplicate keys (uses last key provided if duplicates exist).
        kwargs = dict(
            qnode=qnode,
            weight_shapes=self.weight_shapes,
            output_dim=self.output_shape,
            weight_specs=self.weight_specs,
            **kwargs,
        )
        layer = qml.qnn.KerasLayer(**kwargs)

        # Force set the input and output shapes of the layer.
        layer.input_shape: tf.TensorShape = tf.TensorShape((None, *self.input_shape))
        layer.output_shape: tf.TensorShape = tf.TensorShape((None, *self.output_shape))

        return layer


class AgentCircuit(QuantumCircuit):
    
    def __init__(self,
        wires: int | list[int],
        n_layers: int,
        observables: list | Callable[[list], list] | None = None,
        initial_state: list | Callable[[list], None] = None,
        ):
        super().__init__(
            wires=wires,
            observables=observables,
            )
        self.n_layers = n_layers
        self.initial_state = initial_state

    def __call__(self,
        weights_var,
        weights_enc,
        inputs=None,
        ) -> NDArray | None:

        # Prepare initial state from function.
        if isinstance(self.initial_state, Callable):
            self.initial_state(self.wires)

        # Prepare initial state via state vector.
        elif hasattr(self.initial_state, '__iter__'):
            qml.QubitStateVector(self.initial_state, wires=self.wires)
        
        # Encoding parameters.
        # If inputs were provided then do the following:
        # - Treat `enc_inputs` as lambda values which are multiplied by `inputs`.
        # - `agents_enc_inputs` will NOT have a batch dimension
        # - `inputs` will be 2D with shape (batch, n_agents * d_qubits).
        if inputs is not None:
            inputs = np.reshape(inputs, (-1, self.n_wires)) # Ensure shape is 2D with (batch, d_qubits)
            weights_enc = np.einsum("lqf,bq->blqf", weights_enc, inputs) # For each agent, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
        
        VariationalEncodingPQC(
            weights_var=weights_var,
            weights_enc=weights_enc,
            n_layers=self.n_layers,
            wires=self.wires,
            )
        
        # Return measurements.
        return self.measure()
    
    @property
    def weight_shapes(self):
        shape_var, shape_enc = self.shape
        return {
            'weights_var': shape_var,
            'weights_enc': shape_enc,
        }

    @property
    def weight_specs(self):
        return {
            'weights_var': dict(
                trainable=True,
                dtype='float32',
                initializer=tf.random_uniform_initializer(
                    minval=0.,
                    maxval=np.pi,
                    ),
                ),
            'weights_enc': dict(
                trainable=True,
                dtype='float32',
                initializer=tf.ones,
                ),
        }

    @property
    def input_shape(self):
        """Returns required shape for `inputs` argument.
        
        Note 1: The returned shape does not include the batch dimension.
        Note 2: PennyLane will compress inputs to 2D when batching for usage with `KerasLayer` or `TorchLayer`.
        """
        return (self.n_wires,)

    @property
    def shape(self):
        return self.get_shape(self.wires, self.n_layers)

    @staticmethod
    def get_shape(wires: int | list[int], n_layers: int):
        if isinstance(wires, int): wires = list(range(wires)) # Ensure wires is a list.
        return VariationalEncodingPQC.shape(
            n_layers=n_layers,
            wires=wires,
            )
        
        

class MARLCircuit(QuantumCircuit):
    
    def __init__(self,
        n_agents: int,
        d_qubits: int,
        n_layers: int,
        observables: list | Callable[[list], list] | None = None,
        initial_state: list | Callable[[list], None] = None,
        ):
        super().__init__(
            wires=n_agents * d_qubits,
            observables=observables,
            )
        self.n_agents = n_agents
        self.d_qubits = d_qubits
        self.n_layers = n_layers
        self.initial_state = initial_state

    def __call__(self, 
        agents_var_thetas: NDArray,
        agents_enc_inputs: NDArray,
        inputs = None, # Required for keras layer support; must have shape (batch, n_agents, d_qubits).
        ) -> NDArray | None:

        # Prepare initial state from function.
        if isinstance(self.initial_state, Callable):
            self.initial_state(self.wires)

        # OR, prepare initial state via state vector/list.
        elif hasattr(self.initial_state, '__iter__'):
            qml.QubitStateVector(self.initial_state, wires=self.wires)

        # Create sub-circuit for each agent.
        for aidx in range(self.n_agents):
            qidx = aidx * self.d_qubits # Starting qubit index for the specified agent.
            
            # Variational parameters (batching is optional).
            weights_var = agents_var_thetas[..., aidx, :, :, :]
            
            # Encoding parameters.
            # If inputs were provided then do the following:
            # - Treat `enc_inputs` as lambda values which are multiplied by `inputs`.
            # - `agents_enc_inputs` will NOT have a batch dimension
            # - `inputs` will be 2D with shape (batch, n_agents * d_qubits).
            if inputs is not None:
                inputs = np.reshape(inputs, (-1, self.n_agents, self.d_qubits)) # Ensure shape is 3D with (batch, n_agents, d_qubits)
                weights_enc = np.einsum("alqf,baq->baqf", agents_enc_inputs, inputs) # For each agent, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
            else:
                weights_enc = agents_enc_inputs[..., aidx, :, :, :]
            
            # Add PQC for the wires corresponding to the current agent.
            VariationalEncodingPQC(
                weights_var=weights_var,
                weights_enc=weights_enc,
                n_layers=self.n_layers,
                wires=self.wires[qidx:qidx + self.d_qubits],
            )

        # Return measurements.
        return self.measure()
    
    @property
    def weight_shapes(self):
        shape_var, shape_enc = self.shape
        return {
            'agents_var_thetas': shape_var,
            'agents_enc_inputs': shape_enc,
        }

    @property
    def weight_specs(self):
        return {
            'agents_var_thetas': dict(
                trainable=True,
                dtype='float32',
                initializer=tf.random_uniform_initializer(
                    minval=0.,
                    maxval=np.pi,
                    ),
                ),
            'agents_enc_inputs': dict(
                trainable=True,
                dtype='float32',
                initializer=tf.ones,
                ),
        }

    @property
    def input_shape(self):
        """Returns required shape for `inputs` argument.
        
        Note 1: The returned shape does not include the batch dimension.
        Note 2: PennyLane will compress inputs to 2D when batching for usage with `KerasLayer` or `TorchLayer`.
        """
        return (self.n_agents * self.d_qubits,)

    @property
    def shape(self):
        return self.get_shape(self.n_agents, self.d_qubits, self.n_layers)

    @staticmethod
    def get_shape(n_agents, d_qubits, n_layers):
        wires = list(range(n_agents * d_qubits))
        shape_var, shape_enc = VariationalEncodingPQC.shape(
            n_layers=n_layers,
            wires=wires[:d_qubits], # All agents are identical, so only need shape of first agent (wires 0, 1, ..., d-1).
            )
        shape_var = (n_agents, *shape_var,)
        shape_enc = (n_agents, *shape_enc,)
        return shape_var, shape_enc