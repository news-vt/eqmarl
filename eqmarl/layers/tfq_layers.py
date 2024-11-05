import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_quantum as tfq
import cirq

from ..ops.cirq_ops import *
from ..circuits.cirq_circuits import *



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


class HybridVariationalEncodingPQC(keras.layers.Layer):
    """Hybrid quantum-classical PQC layer."""
    
    def __init__(self, 
        qubits: list,
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables: list,
        name: str = None,
        squash_activation: str = 'linear',
        variational_layer_cls: ParameterizedOperationGate = VariationalRotationLayer,
        encoding_layer_cls: ParameterizedOperationGate = EncodingLayer,
        trainable_w_enc: bool = True,
        ):
        name = name or self.__class__.__name__
        super().__init__(name=name)
        
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.squash_activation = squash_activation
        
        # Build circuit.
        circuit, (symbols_var, symbols_enc) = self.generate_circuit(qubits, d_qubits, n_layers, decompose=True, variational_layer_cls=variational_layer_cls, encoding_layer_cls=encoding_layer_cls)
        self.circuit = circuit
        
        # Define trainable variables for TensorFlow layer.
        self.w_var = tf.Variable(
            initial_value=tf.random_uniform_initializer(minval=0.0, maxval=np.pi)(shape=symbols_var.shape, dtype='float32'),
            trainable=True,
            name='w_var',
        )
        self.w_enc = tf.Variable(
            initial_value=tf.ones(shape=symbols_enc.shape, dtype='float32'),
            # trainable=True,
            trainable=trainable_w_enc,
            name='w_enc',
        )

        # Explicit symbol ordering.
        self.symbols = [str(s) for s in np.concatenate((symbols_var.flatten(), symbols_enc.flatten()))]
        self.symbols_idx = tf.constant([self.symbols.index(s) for s in sorted(self.symbols)]) # Cross-ref symbol with its index in the explicit ordering.
        
        # Empty circuit for batching.
        self.empty_circuit_tensor = tfq.convert_to_tensor([cirq.Circuit()])
        
        # The variational+encoding circuit for computation.
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)
        
    def call(self, inputs):
        batch_size = tf.gather(tf.shape(inputs), 0)
        
        # Since input is batched, we must batch the TFQ circuits.
        batched_circuits = tf.repeat(self.empty_circuit_tensor, repeats=batch_size)
        
        # Batch the variational weight angles.
        angles_var = tf.reshape(tf.tile(self.w_var, multiples=[batch_size, *([1]*(len(self.w_var.shape)-1))]), shape=(-1, *self.w_var.shape))
        
        # Multiply input vectors by the encoding weights.
        # Preserves batching.
        # Einsum dimension labels:
        #   b = batch
        #   l = layer
        #   q = qubit
        #   f = feature
        # angles_enc = tf.einsum("lqf,bq->blqf", self.w_enc, inputs)
        angles_enc = tf.einsum("lqf,bqf->blqf", self.w_enc, inputs) # For each partition `p`, encode each `input` state feature `q` on the `q-th` qubit and repeat encoding on same qubit for every layer `l`. Number of input features must match number of qubits.
        # _,p,q,f = inputs.shape
        # angles_enc = tf.reshape(tf.tile(inputs, multiples=[1, 1, self.n_layers, 1]), shape=(-1,p,self.n_layers,q,f))
        
        # Squash the encoding input angles using the provided activation function.
        if self.squash_activation in ('arctan', 'atan'):
            angles_enc = tf.math.atan(angles_enc)
        else:
            angles_enc = keras.layers.Activation(self.squash_activation)(angles_enc)
        
        
        # Combine all angles into a single batched tensor.
        # This is necessary because TensorFlowQuantum requires parameters to be in 1D list format. Since the circuits are also batched, this turns into 2D with shape (batch_size, num_symbols).
        #
        # Since all angles are different shapes, compress each down to batched 2D and then concatenate along the feature dimension.
        joined_angles = tf.concat([
            tf.reshape(angles_var, (batch_size, -1)),
            tf.reshape(angles_enc, (batch_size, -1)),
        ], axis=1)
        #
        # Now reorder angles based on explicit symbol ordering.
        joined_angles = tf.gather(joined_angles, self.symbols_idx, axis=1)
        
        # Run batched angles.
        # Result will be 2D with shape (batch_size, num_observables).
        out = self.computation_layer([batched_circuits, joined_angles])
        
        return out
    
    @staticmethod
    def generate_circuit(*args, **kwargs):
        return generate_variational_encoding_circuit(*args, **kwargs)




class HybridPartiteVariationalEncodingPQC(keras.layers.Layer):
    """eQMARL variant of `HybridVariationalEncodingPQC`."""
    
    def __init__(self, 
        qubits: list,
        n_parts: int, # Number of partitions.
        d_qubits: int, # Number of qubits per partition.
        n_layers: int,
        observables: list,
        name: str = None,
        squash_activation: str = 'linear',
        variational_layer_cls: ParameterizedOperationGate = VariationalRotationLayer,
        encoding_layer_cls: ParameterizedOperationGate = EncodingLayer,
        trainable_w_enc: bool = True,
        input_entanglement: bool = True, # Flag to enable input entanglement (defaults to True).
        input_entanglement_type: bool = 'phi+', # ['phi+', 'phi-', 'psi+', 'psi-']
        ):
        name = name or self.__class__.__name__
        super().__init__(name=name)
        
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.squash_activation = squash_activation
        
        # Build circuit.
        circuit, (symbols_var, symbols_enc) = self.generate_circuit(qubits, n_parts, d_qubits, n_layers, decompose=True, variational_layer_cls=variational_layer_cls, encoding_layer_cls=encoding_layer_cls, input_entanglement=input_entanglement, input_entanglement_type=input_entanglement_type)
        self.circuit = circuit
        
        # Define trainable variables for TensorFlow layer.
        self.w_var = tf.Variable(
            initial_value=tf.random_uniform_initializer(minval=0.0, maxval=np.pi)(shape=symbols_var.shape, dtype='float32'),
            trainable=True,
            name='w_var',
        )
        self.w_enc = tf.Variable(
            initial_value=tf.ones(shape=symbols_enc.shape, dtype='float32'),
            # trainable=True,
            trainable=trainable_w_enc,
            name='w_enc',
        )
        
        # print(f"{self.w_var.shape=}")
        # print(f"{self.w_enc.shape=}")
        
        # Explicit symbol ordering.
        self.symbols = [str(s) for s in np.concatenate((symbols_var.flatten(), symbols_enc.flatten()))]
        self.symbols_idx = tf.constant([self.symbols.index(s) for s in sorted(self.symbols)]) # Cross-ref symbol with its index in the explicit ordering.
        
        # Empty circuit for batching.
        self.empty_circuit_tensor = tfq.convert_to_tensor([cirq.Circuit()])
        
        # The variational+encoding circuit for computation.
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)
        
    def call(self, inputs):
        batch_size = tf.gather(tf.shape(inputs), 0)
        
        # Since input is batched, we must batch the TFQ circuits.
        batched_circuits = tf.repeat(self.empty_circuit_tensor, repeats=batch_size)
        
        # Batch the variational weight angles.
        angles_var = tf.reshape(tf.tile(self.w_var, multiples=[batch_size, *([1]*(len(self.w_var.shape)-1))]), shape=(-1, *self.w_var.shape))
        
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
        # _,p,q,f = inputs.shape
        # angles_enc = tf.reshape(tf.tile(inputs, multiples=[1, 1, self.n_layers, 1]), shape=(-1,p,self.n_layers,q,f))
        
        # Squash the encoding input angles using the provided activation function.
        if self.squash_activation in ('arctan', 'atan'):
            angles_enc = tf.math.atan(angles_enc)
        else:
            angles_enc = keras.layers.Activation(self.squash_activation)(angles_enc)
        
        
        # Combine all angles into a single batched tensor.
        # This is necessary because TensorFlowQuantum requires parameters to be in 1D list format. Since the circuits are also batched, this turns into 2D with shape (batch_size, num_symbols).
        #
        # Since all angles are different shapes, compress each down to batched 2D and then concatenate along the feature dimension.
        joined_angles = tf.concat([
            tf.reshape(angles_var, (batch_size, -1)),
            tf.reshape(angles_enc, (batch_size, -1)),
        ], axis=1)
        #
        # Now reorder angles based on explicit symbol ordering.
        joined_angles = tf.gather(joined_angles, self.symbols_idx, axis=1)
        
        # Run batched angles.
        # Result will be 2D with shape (batch_size, num_observables).
        out = self.computation_layer([batched_circuits, joined_angles])
        
        return out
    
    @staticmethod
    def generate_circuit(*args, **kwargs):
        return generate_partite_variational_encoding_circuit(*args, **kwargs)