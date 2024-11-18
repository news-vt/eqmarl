# https://github.com/news-vt/eqmarl/blob/4f8765d8a9a1d02d29a6ebcff370229daf724cfe/measure_observables.ipynb
# https://github.com/news-vt/eqmarl/blob/4f8765d8a9a1d02d29a6ebcff370229daf724cfe/eqmarl/circuits/pqc.py

from typing import (
    Any,
    Callable,
    Iterable,
    Type,
)
import pennylane as qml
from pennylane import numpy as np # Must import Numpy from PennyLane.
import sympy
from enum import Enum
import tensorflow as tf
import functools as ft

from ..ops import pennylane_ops as ops

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x)) # For numerical stability
    return exp_x / np.sum(exp_x, axis=0)


class OLD_VariationalEncodingCircuit:
    """Parameterized quantum circuit with variational and encoding layers."""
    
    class WeightManifest(str, Enum):
        WEIGHT_VAR = 'weights_var' # Matches names of `__call__` arguments.
        WEIGHT_ENC = 'weights_enc' # Matches names of `__call__` arguments.
    
    def __init__(self,
        wires: list[int],
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables_func: Callable,
        variational_layer_cls: Type[ops.ParameterizedOperation] = ops.VariationalRotationLayer,
        encoding_layer_cls: Type[ops.ParameterizedOperation] = ops.EncodingLayer,
        squash_activation: str = 'linear',
        ):
        self.wires = wires
        self.d_qubits = d_qubits
        self.n_layers = n_layers
        self.observables_func = observables_func
        self.squash_activation = squash_activation
        self.variational_layer_cls = variational_layer_cls
        self.encoding_layer_cls = encoding_layer_cls
        print(f"{encoding_layer_cls=}")

    def __call__(self, weights_var, weights_enc, inputs=None):
        """Allows calling an instance of this circuit object like a function.
        
        The `inputs` argument is required for compatibility with the `qml.qnn.KerasLayer` class.
        """

        # Encoding parameters.
        # If inputs were provided then do the following:
        # - Treat `enc_inputs` as lambda values which are multiplied by `inputs`.
        # - `agents_enc_inputs` will NOT have a batch dimension
        # - `inputs` will be 2D with shape (batch, n_agents * d_qubits).
        # Multiply input vectors by the encoding weights.
        # Preserves batching.
        # Einsum dimension labels:
        #   b = batch
        #   l = layer
        #   q = qubit
        #   f = feature
        if inputs is not None:
            # inputs = np.reshape(inputs, (-1, self.d_qubits)) # Ensure shape is 2D with (batch, d_qubits)
            inputs = tf.reshape(inputs, (-1, *self.input_shape)) # Ensure shape is 2D with (batch, d_qubits, n_features)
            print(f"BEFORE: {inputs.shape=}")
            print(f"BEFORE: {weights_enc.shape=}")
            # angles_enc = np.einsum("lqf,bqf->blqf", weights_enc, inputs) # For each agent, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
            angles_enc = tf.einsum("lqf,bqf->blqf", weights_enc, inputs) # For each agent, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
            print(f"AFTER: {angles_enc.shape=}")
        else:
            angles_enc = weights_enc
            
        # Squash the encoding input angles using the provided activation function.
        if self.squash_activation in ('linear',): # Do nothing.
            pass
        elif self.squash_activation in ('relu',): # Do nothing.
            angles_enc = relu(angles_enc)
        elif self.squash_activation in ('arctan', 'atan'):
            angles_enc = tf.atan(angles_enc)
        elif self.squash_activation in ('tanh',):
            angles_enc = tf.tanh(angles_enc)
        elif self.squash_activation in ('sigmoid',):
            angles_enc = sigmoid(angles_enc)
        elif self.squash_activation in ('softmax',):
            angles_enc = softmax(angles_enc)
        
        # Run the circuit using the batched variational and encoding weights.
        return self.circuit(weights_var=weights_var, weights_enc=angles_enc)

    def circuit(self, weights_var, weights_enc):
        
        # Apply hadamard to all qubits.
        for w in self.wires:
            qml.Hadamard(wires=w)

        # Run the circuit using the batched variational and encoding weights.
        ops.VariationalEncodingParameterizedOperation(
            weights_var=weights_var,
            weights_enc=weights_enc,
            n_layers=self.n_layers,
            wires=self.wires,
            encoding_layer=self.encoding_layer_cls,
            variational_layer=self.variational_layer_cls,
        )
        
        # Measure.
        measurements = [qml.expval(obs) for obs in self.observables_func(self.wires)]
        return measurements

    @property
    def shape(self):
        return self.get_shape(self.wires, self.n_layers, self.variational_layer_cls, self.encoding_layer_cls)
    
    @property
    def output_shape(self):
        """Returns number of observables at output.
        
        This is useful in combination with `qml.KerasLayer`.
        """
        return (len(self.observables_func(self.wires)),)

    @property
    def input_shape(self):
        """Returns required shape for `inputs` argument.
        
        Note 1: The returned shape does not include the batch dimension.
        Note 2: PennyLane will compress inputs to 2D when batching for usage with `KerasLayer` or `TorchLayer`.
        """
        # return (self.d_qubits,)
        return self.encoding_layer_cls.get_shape(len(self.wires))
        # return ft.reduce(lambda x, y: x * y, self.encoding_layer_cls.get_shape(len(self.wires)))
    
    @property
    def weight_shapes(self) -> dict:
        shape_var, shape_enc = self.shape
        weight_shapes = {
            self.WeightManifest.WEIGHT_VAR: shape_var,
            self.WeightManifest.WEIGHT_ENC: shape_enc,
        }
        return weight_shapes

    @staticmethod
    def get_shape(wires, n_layers, variational_layer, encoding_layer):
        return ops.VariationalEncodingParameterizedOperation.get_shape(
            n_layers=n_layers,
            wires=wires,
            variational_layer=variational_layer,
            encoding_layer=encoding_layer,
            )







class VariationalEncodingCircuit:
    """Parameterized quantum circuit with variational and encoding layers."""
    
    class WeightManifest(str, Enum):
        WEIGHT_VAR = 'weights_var' # Matches names of `__call__` arguments.
        # WEIGHT_ENC = 'weights_enc' # Matches names of `__call__` arguments.
    
    def __init__(self,
        wires: list[int],
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables_func: Callable,
        variational_layer_cls: Type[ops.ParameterizedOperation] = ops.VariationalRotationLayer,
        encoding_layer_cls: Type[ops.ParameterizedOperation] = ops.EncodingLayer,
        # squash_activation: str = 'linear',
        ):
        self.wires = wires
        self.d_qubits = d_qubits
        self.n_layers = n_layers
        self.observables_func = observables_func
        # self.squash_activation = squash_activation
        self.variational_layer_cls = variational_layer_cls
        self.encoding_layer_cls = encoding_layer_cls

    def __call__(self, weights_var, inputs):
        """Allows calling an instance of this circuit object like a function.
        
        The `inputs` argument is required for compatibility with the `qml.qnn.KerasLayer` class.
        """

        # # Encoding parameters.
        # # If inputs were provided then do the following:
        # # - Treat `enc_inputs` as lambda values which are multiplied by `inputs`.
        # # - `agents_enc_inputs` will NOT have a batch dimension
        # # - `inputs` will be 2D with shape (batch, n_agents * d_qubits).
        # # Multiply input vectors by the encoding weights.
        # # Preserves batching.
        # # Einsum dimension labels:
        # #   b = batch
        # #   l = layer
        # #   q = qubit
        # #   f = feature
        # if inputs is not None:
        #     # inputs = np.reshape(inputs, (-1, self.d_qubits)) # Ensure shape is 2D with (batch, d_qubits)
        #     inputs = tf.reshape(inputs, (-1, *self.input_shape)) # Ensure shape is 2D with (batch, d_qubits, n_features)
        #     print(f"BEFORE: {inputs.shape=}")
        #     print(f"BEFORE: {weights_enc.shape=}")
        #     # angles_enc = np.einsum("lqf,bqf->blqf", weights_enc, inputs) # For each agent, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
        #     angles_enc = tf.einsum("lqf,bqf->blqf", weights_enc, inputs) # For each agent, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
        #     print(f"AFTER: {angles_enc.shape=}")
        # else:
        #     angles_enc = weights_enc
            
        # # Squash the encoding input angles using the provided activation function.
        # if self.squash_activation in ('linear',): # Do nothing.
        #     pass
        # elif self.squash_activation in ('relu',): # Do nothing.
        #     angles_enc = relu(angles_enc)
        # elif self.squash_activation in ('arctan', 'atan'):
        #     angles_enc = tf.atan(angles_enc)
        # elif self.squash_activation in ('tanh',):
        #     angles_enc = tf.tanh(angles_enc)
        # elif self.squash_activation in ('sigmoid',):
        #     angles_enc = sigmoid(angles_enc)
        # elif self.squash_activation in ('softmax',):
        #     angles_enc = softmax(angles_enc)
        
        
        print(f"[VariationalEncodingCircuit_NEW.call] {inputs.shape=}")
        _, shape_enc = self.shape
        print(f"[PartiteVariationalEncodingCircuit.call] {shape_enc=}")
        inputs = tf.reshape(inputs, shape=(-1, *shape_enc))
        print(f"[VariationalEncodingCircuit_NEW.call] {inputs.shape=}")
        
        # Run the circuit using the batched variational and encoding weights.
        return self.circuit(weights_var=weights_var, weights_enc=inputs)

    def circuit(self, weights_var, weights_enc):
        print(f"[VariationalEncodingCircuit_NEW.circuit] {weights_var.shape=}")
        print(f"[VariationalEncodingCircuit_NEW.circuit] {weights_enc.shape=}")
        
        # Apply hadamard to all qubits.
        for w in self.wires:
            qml.Hadamard(wires=w)

        # Run the circuit using the batched variational and encoding weights.
        ops.VariationalEncodingParameterizedOperation(
            weights_var=weights_var,
            weights_enc=weights_enc,
            n_layers=self.n_layers,
            wires=self.wires,
            encoding_layer=self.encoding_layer_cls,
            variational_layer=self.variational_layer_cls,
        )
        
        # Measure.
        measurements = [qml.expval(obs) for obs in self.observables_func(self.wires)]
        return measurements

    @property
    def shape(self):
        return self.get_shape(self.wires, self.n_layers, self.variational_layer_cls, self.encoding_layer_cls)
    
    @property
    def output_shape(self):
        """Returns number of observables at output.
        
        This is useful in combination with `qml.KerasLayer`.
        """
        return (len(self.observables_func(self.wires)),)

    @property
    def input_shape(self):
        """Returns required shape for `inputs` argument.
        
        Note 1: The returned shape does not include the batch dimension.
        Note 2: PennyLane will compress inputs to 2D when batching for usage with `KerasLayer` or `TorchLayer`.
        """
        # return (self.d_qubits,)
        return self.encoding_layer_cls.get_shape(len(self.wires))
        # return ft.reduce(lambda x, y: x * y, self.encoding_layer_cls.get_shape(len(self.wires)))
    
    @property
    def weight_shapes(self) -> dict:
        shape_var, shape_enc = self.shape
        weight_shapes = {
            self.WeightManifest.WEIGHT_VAR: shape_var,
            # self.WeightManifest.WEIGHT_ENC: shape_enc,
        }
        return weight_shapes

    @staticmethod
    def get_shape(wires, n_layers, variational_layer, encoding_layer):
        return ops.VariationalEncodingParameterizedOperation.get_shape(
            n_layers=n_layers,
            wires=wires,
            variational_layer=variational_layer,
            encoding_layer=encoding_layer,
            )





class PartiteVariationalEncodingCircuit:
    """Parameterized quantum circuit with variational and encoding layers."""
    
    class WeightManifest(str, Enum):
        WEIGHT_VAR = 'weights_var' # Matches names of `__call__` arguments.
        # WEIGHT_ENC = 'weights_enc' # Matches names of `__call__` arguments.
    
    def __init__(self,
        wires: list[int],
        n_parts: int, # Number of partitions.
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables_func: Callable,
        variational_layer_cls: Type[ops.ParameterizedOperation] = ops.VariationalRotationLayer,
        encoding_layer_cls: Type[ops.ParameterizedOperation] = ops.EncodingLayer,
        # squash_activation: str = 'linear',
        input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
        input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
        ):
        self.wires = wires
        self.n_parts = n_parts
        self.d_qubits = d_qubits
        self.n_layers = n_layers
        self.observables_func = observables_func
        # self.squash_activation = squash_activation
        self.variational_layer_cls = variational_layer_cls
        self.encoding_layer_cls = encoding_layer_cls
        self.input_entanglement = input_entanglement
        self.input_entanglement_type = input_entanglement_type

    def __call__(self, weights_var, inputs):
        """Allows calling an instance of this circuit object like a function.
        
        The `inputs` argument is required for compatibility with the `qml.qnn.KerasLayer` class.
        """
        
        print(f"[PartiteVariationalEncodingCircuit.call] {inputs.shape=}")
        _, shape_enc = self.shape
        print(f"[PartiteVariationalEncodingCircuit.call] {shape_enc=}")
        inputs = tf.reshape(inputs, shape=(-1, *shape_enc))
        print(f"[PartiteVariationalEncodingCircuit.call] {inputs.shape=}")
        
        # Run the circuit using the batched variational and encoding weights.
        return self.circuit(weights_var=weights_var, weights_enc=inputs)

    def circuit(self, weights_var, weights_enc):
        print(f"[PartiteVariationalEncodingCircuit.circuit] {weights_var.shape=}")
        print(f"[PartiteVariationalEncodingCircuit.circuit] {weights_enc.shape=}")
        
        # Add GHZ entangling layer at the start if necessary.
        if self.input_entanglement:
            # ['phi+', 'phi-', 'psi+', 'psi-']
            if self.input_entanglement_type == 'phi+':
                ops.entangle_agents_phi_plus(self.wires, self.d_qubits, self.n_parts)
            elif self.input_entanglement_type == 'phi-':
                ops.entangle_agents_phi_minus(self.wires, self.d_qubits, self.n_parts)
            elif self.input_entanglement_type == 'psi+':
                ops.entangle_agents_psi_plus(self.wires, self.d_qubits, self.n_parts)
            elif self.input_entanglement_type == 'psi-':
                ops.entangle_agents_psi_minus(self.wires, self.d_qubits, self.n_parts)
            else:
                raise ValueError(f"unsupported input entanglement type {self.input_entanglement_type}")

        # Build circuit in partitions.
        for pidx in range(self.n_parts):
            qidx = pidx * self.d_qubits # Starting qubit index for the current partition.
            
            # Run the circuit using the batched variational and encoding weights.
            ops.VariationalEncodingParameterizedOperation(
                weights_var=weights_var[pidx],
                weights_enc=weights_enc[pidx],
                n_layers=self.n_layers,
                wires=self.wires[qidx:qidx + self.d_qubits],
                encoding_layer=self.encoding_layer_cls,
                variational_layer=self.variational_layer_cls,
            )
        
            # for l in range(self.n_layers):
            #     # Variational layer.
            #     ops.append(
            #         self.variational_layer_cls(theta_var[pidx, l], name=f'{pidx}-v{l}')(*qubits[qidx:qidx + self.d_qubits])
            #     )
                
            #     # Entangling layer.
            #     ops.append(
            #         circular_entangling_layer(qubits[qidx:qidx + self.d_qubits])
            #     )
                
            #     # Encoding layer.
            #     ops.append(
            #         self.encoding_layer_cls(theta_enc[pidx, l], name=f'{pidx}-e{l}')(*qubits[qidx:qidx + self.d_qubits])
            #     )
            # # Last variational layer at the end.
            # ops.append(
            #     self.variational_layer_cls(theta_var[pidx, l+1], name=f'v{l+1}')(*qubits[qidx:qidx + self.d_qubits])
            # )

        # Measure.
        measurements = [qml.expval(obs) for obs in self.observables_func(self.wires)]
        return measurements

    @property
    def shape(self):
        return self.get_shape(self.d_qubits, self.n_parts, self.n_layers, self.variational_layer_cls, self.encoding_layer_cls)
    
    @property
    def output_shape(self):
        """Returns number of observables at output.
        
        This is useful in combination with `qml.KerasLayer`.
        """
        return (len(self.observables_func(self.wires)),)

    @property
    def input_shape(self):
        """Returns required shape for `inputs` argument.
        
        Note 1: The returned shape does not include the batch dimension.
        Note 2: PennyLane will compress inputs to 2D when batching for usage with `KerasLayer` or `TorchLayer`.
        """
        _, shape_enc = self.shape
        return shape_enc
        # return (self.d_qubits,)
        # return self.encoding_layer_cls.get_shape(len(self.wires))
        # return ft.reduce(lambda x, y: x * y, self.encoding_layer_cls.get_shape(len(self.wires)))
    
    @property
    def weight_shapes(self) -> dict:
        shape_var, _ = self.shape
        weight_shapes = {
            self.WeightManifest.WEIGHT_VAR: shape_var,
        }
        return weight_shapes

    @staticmethod
    def get_shape(d_qubits, n_parts, n_layers, variational_layer, encoding_layer):
        wires = list(range(d_qubits))
        # Get shapes for a single partition.
        shape_var, shape_enc = ops.VariationalEncodingParameterizedOperation.get_shape(
            n_layers=n_layers,
            wires=wires,
            variational_layer=variational_layer,
            encoding_layer=encoding_layer,
            )
        # Modify existing shapes to include number of partitions at first index.
        shape_var = (n_parts, *shape_var)
        shape_enc = (n_parts, *shape_enc)
        return (shape_var, shape_enc)












# CircuitType = Callable[[], Iterable[Any]]


# def generate_variational_encoding_circuit(
#     wires: list[int],
#     d_qubits: int, # Number of qubits per partition.
#     n_layers: int,
#     decompose: bool = False,
#     variational_layer_cls:  ops.ParameterizedOperation = ops.VariationalRotationLayer,
#     encoding_layer_cls:  ops.ParameterizedOperation = ops.EncodingLayer,
#     ) -> tuple[CircuitType, tuple[np.tensor]]:
#     """Parameterized variational and encoding circuit.
#     """
    
#     shape_var = variational_layer_cls.get_shape(d_qubits)
#     shape_enc = encoding_layer_cls.get_shape(d_qubits)
    
#     ### Define weights for circuit.
#     #
#     ## Variational shape
#     theta_var = sympy.symbols(f'var(0:{n_layers+1})_' + '_'.join(f'(0:{x})' for x in shape_var))
#     theta_var = np.asarray(theta_var).reshape((n_layers+1, *shape_var))
#     ## Encoding shape
#     theta_enc = sympy.symbols(f'enc(0:{n_layers})_' + '_'.join(f'(0:{x})' for x in shape_enc))
#     theta_enc = np.asarray(theta_enc).reshape((n_layers, *shape_enc))
    

#     def gen_circuit() -> Iterable[Any]:

#         # Apply hadamard to all qubits.
#         for w in wires:
#             yield qml.Hadamard(w)

#         # Build circuit in layers.
#         for l in range(n_layers):