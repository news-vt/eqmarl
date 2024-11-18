from typing import (
    Type,
    Callable,
)

import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from ..ops import pennylane_ops as ops
from ..circuits import pennylane_circuits as circuits


class HybridVariationalEncodingPQC(keras.layers.Layer):
    """Hybrid quantum-classical PQC layer."""
    
    def __init__(self, 
        wires: list[int],
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables_func: Callable,
        name: str = None,
        squash_activation: str = 'linear',
        variational_layer_cls: Type[ops.ParameterizedOperation] = ops.VariationalRotationLayer,
        encoding_layer_cls: Type[ops.ParameterizedOperation] = ops.EncodingLayer,
        trainable_w_enc: bool = True,
        pennylane_device: str = 'default.qubit',
        ):
        name = name or self.__class__.__name__
        super().__init__(name=name)
        
        self.wires = wires
        self.n_layers = n_layers
        self.n_qubits = len(wires)
        self.squash_activation = squash_activation
        self.trainable_w_enc = trainable_w_enc
        self.pennylane_device = pennylane_device
        
        # Build the circuit.
        self.circuit = self.generate_circuit(
            wires=wires,
            d_qubits=d_qubits,
            n_layers=n_layers,
            observables_func=observables_func,
            variational_layer_cls=variational_layer_cls,
            encoding_layer_cls=encoding_layer_cls,
        )
        
        _, shape_enc = self.circuit.shape
        self.w_enc = tf.Variable(
            initial_value=tf.ones(shape=shape_enc, dtype='float32'),
            # trainable=True,
            trainable=trainable_w_enc,
            name='w_enc',
        )

    def build(self, input_shape):
        print(f"[HybridVariationalEncodingPQC_NEW.build] {input_shape=}")
        
        # Attach circuit to PennyLane device and QNode.
        device = qml.device(self.pennylane_device, wires=self.n_qubits)
        self.qnode = qml.QNode(func=self.circuit, device=device, interface='tf', diff_method='best')
        
        weight_shapes = self.circuit.weight_shapes
        weight_specs = {
            self.circuit.WeightManifest.WEIGHT_VAR: {
                "initializer": keras.initializers.random_uniform(minval=0.0, maxval=np.pi),
                "trainable": True, # Variational weights are always trainable.
                "dtype": "float32",
            },
            # self.circuit.WeightManifest.WEIGHT_ENC: {
            #     "initializer": keras.initializers.Ones(),
            #     "trainable": self.trainable_w_enc, # Encoding weights are trainable on-demand.
            #     "dtype": "float32",
            # },
        }

        # Build the Keras quantum layer.
        self.qlayer = qml.qnn.KerasLayer(
            qnode=self.qnode,
            weight_shapes=weight_shapes,
            weight_specs=weight_specs,
            output_dim=self.circuit.output_shape,
            )
        return super().build(input_shape)
    
    def call(self, inputs):
        
        print(f"[HybridVariationalEncodingPQC_NEW.call] {inputs.shape=}")
        
        # Multiply input vectors by the encoding weights.
        # Preserves batching.
        # Einsum dimension labels:
        #   b = batch
        #   l = layer
        #   q = qubit
        #   f = feature
        # angles_enc = tf.einsum("lqf,bq->blqf", self.w_enc, inputs)
        angles_enc = tf.einsum("lqf,bqf->blqf", self.w_enc, inputs) # For each partition `p`, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.

        # Squash the encoding input angles using the provided activation function.
        if self.squash_activation in ('arctan', 'atan'):
            angles_enc = tf.math.atan(angles_enc)
        else:
            angles_enc = keras.layers.Activation(self.squash_activation)(angles_enc)
        
        print(f"[HybridVariationalEncodingPQC_NEW.call] {angles_enc.shape=}")
        
        angles_enc = keras.layers.Flatten()(angles_enc)
        print(f"[HybridVariationalEncodingPQC_NEW.call] {angles_enc.shape=}")
        
        # Pass the encoding angles into the quantum cicuit.
        return self.qlayer(angles_enc)
    
    def compute_output_shape(self, input_shape):
        return self.circuit.output_shape
    
    @staticmethod
    def generate_circuit(*args, **kwargs):
        return circuits.VariationalEncodingCircuit(*args, **kwargs)











class HybridPartiteVariationalEncodingPQC(keras.layers.Layer):
    """eQMARL variant of `HybridVariationalEncodingPQC`."""
    
    def __init__(self, 
        wires: list[int],
        n_parts: int, # Number of partitions.
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables_func: Callable,
        name: str = None,
        squash_activation: str = 'linear',
        variational_layer_cls: Type[ops.ParameterizedOperation] = ops.VariationalRotationLayer,
        encoding_layer_cls: Type[ops.ParameterizedOperation] = ops.EncodingLayer,
        trainable_w_enc: bool = True,
        input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
        input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
        pennylane_device: str = 'default.qubit',
        ):
        name = name or self.__class__.__name__
        super().__init__(name=name)
        
        self.wires = wires
        self.n_layers = n_layers
        self.n_qubits = len(wires)
        self.squash_activation = squash_activation
        self.trainable_w_enc = trainable_w_enc
        self.pennylane_device = pennylane_device
        
        # Build the circuit.
        self.circuit = self.generate_circuit(
            wires=wires,
            n_parts=n_parts,
            d_qubits=d_qubits,
            n_layers=n_layers,
            observables_func=observables_func,
            variational_layer_cls=variational_layer_cls,
            encoding_layer_cls=encoding_layer_cls,
            input_entanglement=input_entanglement,
            input_entanglement_type=input_entanglement_type,
        )
        
        _, shape_enc = self.circuit.shape
        self.w_enc = tf.Variable(
            initial_value=tf.ones(shape=shape_enc, dtype='float32'),
            # trainable=True,
            trainable=trainable_w_enc,
            name='w_enc',
        )
        
    def build(self, input_shape):
        print(f"[HybridPartiteVariationalEncodingPQC] {input_shape=}")
        
        # Attach circuit to PennyLane device and QNode.
        device = qml.device(self.pennylane_device, wires=self.n_qubits)
        self.qnode = qml.QNode(func=self.circuit, device=device, interface='tf')
        
        weight_shapes = self.circuit.weight_shapes
        weight_specs = {
            self.circuit.WeightManifest.WEIGHT_VAR: {
                "initializer": keras.initializers.random_uniform(minval=0.0, maxval=np.pi),
                "trainable": True, # Variational weights are always trainable.
                "dtype": "float32",
            },
        }

        # Build the Keras quantum layer.
        self.qlayer = qml.qnn.KerasLayer(
            qnode=self.qnode,
            weight_shapes=weight_shapes,
            weight_specs=weight_specs,
            output_dim=self.circuit.output_shape,
            )
        return super().build(input_shape)
        
    def call(self, inputs):
        
        
        print(f"[HybridPartiteVariationalEncodingPQC.call] {self.w_enc.shape=}")
        print(f"[HybridPartiteVariationalEncodingPQC.call] {inputs.shape=}")
        
        # Multiply input vectors by the encoding weights.
        # Preserves batching.
        # Einsum dimension labels:
        #   p = partition index
        #   b = batch
        #   l = layer
        #   q = qubit
        #   f = feature
        # angles_enc = tf.einsum("lqf,bq->blqf", self.w_enc, inputs)
        angles_enc = tf.einsum("plqf,bpqf->bplqf", self.w_enc, inputs) # For each partition `p`, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.

        # Squash the encoding input angles using the provided activation function.
        if self.squash_activation in ('arctan', 'atan'):
            angles_enc = tf.math.atan(angles_enc)
        else:
            angles_enc = keras.layers.Activation(self.squash_activation)(angles_enc)
        
        print(f"[HybridPartiteVariationalEncodingPQC.call] {angles_enc.shape=}")
        
        angles_enc = keras.layers.Flatten()(angles_enc)
        print(f"[HybridPartiteVariationalEncodingPQC.call] {angles_enc.shape=}")
        
        # Pass the encoding angles into the quantum cicuit.
        return self.qlayer(angles_enc)
        
        ##################
        
        
        # batch_size = tf.gather(tf.shape(inputs), 0)
        
        # # Since input is batched, we must batch the TFQ circuits.
        # batched_circuits = tf.repeat(self.empty_circuit_tensor, repeats=batch_size)
        
        # # Batch the variational weight angles.
        # angles_var = tf.reshape(tf.tile(self.w_var, multiples=[batch_size, *([1]*(len(self.w_var.shape)-1))]), shape=(-1, *self.w_var.shape))
        
        # # Multiply input vectors by the encoding weights.
        # # Preserves batching.
        # # Einsum dimension labels:
        # #   p = partition index
        # #   b = batch
        # #   l = layer
        # #   q = qubit
        # #   f = feature
        # # angles_enc = tf.einsum("lqf,bq->blqf", self.w_enc, inputs)
        # angles_enc = tf.einsum("plqf,bpqf->bplqf", self.w_enc, inputs) # For each partition `p`, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
        # # _,p,q,f = inputs.shape
        # # angles_enc = tf.reshape(tf.tile(inputs, multiples=[1, 1, self.n_layers, 1]), shape=(-1,p,self.n_layers,q,f))
        
        # # Squash the encoding input angles using the provided activation function.
        # if self.squash_activation in ('arctan', 'atan'):
        #     angles_enc = tf.math.atan(angles_enc)
        # else:
        #     angles_enc = keras.layers.Activation(self.squash_activation)(angles_enc)
        
        
        # # Combine all angles into a single batched tensor.
        # # This is necessary because TensorFlowQuantum requires parameters to be in 1D list format. Since the circuits are also batched, this turns into 2D with shape (batch_size, num_symbols).
        # #
        # # Since all angles are different shapes, compress each down to batched 2D and then concatenate along the feature dimension.
        # joined_angles = tf.concat([
        #     tf.reshape(angles_var, (batch_size, -1)),
        #     tf.reshape(angles_enc, (batch_size, -1)),
        # ], axis=1)
        # #
        # # Now reorder angles based on explicit symbol ordering.
        # joined_angles = tf.gather(joined_angles, self.symbols_idx, axis=1)
        
        # # Run batched angles.
        # # Result will be 2D with shape (batch_size, num_observables).
        # out = self.computation_layer([batched_circuits, joined_angles])
        
        # return out
    
    def compute_output_shape(self, input_shape):
        return self.circuit.output_shape
    
    @staticmethod
    def generate_circuit(*args, **kwargs):
        return circuits.PartiteVariationalEncodingCircuit(*args, **kwargs)
    
    # @staticmethod
    # def generate_circuit(*args, **kwargs):
    #     return generate_partite_variational_encoding_circuit(*args, **kwargs)