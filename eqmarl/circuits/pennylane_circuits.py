# https://github.com/news-vt/eqmarl/blob/4f8765d8a9a1d02d29a6ebcff370229daf724cfe/measure_observables.ipynb
# https://github.com/news-vt/eqmarl/blob/4f8765d8a9a1d02d29a6ebcff370229daf724cfe/eqmarl/circuits/pqc.py

from typing import (
    Any,
    Callable,
    Type,
)
import pennylane as qml
from pennylane import numpy as np # Must import Numpy from PennyLane.
from enum import Enum
import tensorflow as tf

from ..ops import pennylane_ops as ops


class VariationalEncodingCircuit:
    """Parameterized quantum circuit with variational and encoding layers."""
    
    class WeightManifest(str, Enum):
        WEIGHT_VAR = 'weights_var' # Matches names of `__call__` arguments.
    
    def __init__(self,
        wires: list[int],
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables_func: Callable,
        variational_layer_cls: Type[ops.ParameterizedOperation] = ops.VariationalRotationLayer,
        encoding_layer_cls: Type[ops.ParameterizedOperation] = ops.EncodingLayer,
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
