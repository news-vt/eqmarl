from __future__ import annotations

from ..circuits.pqc import *
from ..types import *
import cirq
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_quantum as tfq

# Classical A3C tutorial in TensorFlow (2016):
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2#.hg13tn9zw


class PQCBase(keras.layers.Layer):

    @staticmethod
    def build_circuit(*args, **kwargs) -> ParameterizedCircuitFunctionReturnType:
        """Returns generator to construct a PQC along with a tuple of parameter symbols."""
        raise NotImplementedError()


class VariationalEncodingPQC(PQCBase):
    """More complex parameterized variational + encoding policy circuit.
    
    Tensorflow layer representation of `variational_encoding_pqc`.
    
    Contains variational layers with Rx, Ry, Rz rotations parameterized by $\theta$, followed by a next-neighbor entanglement layer, followed by an encoding layer to encode the state $s$. The final layer in the circuit is a variational layer.
    
    Implementation inspired by: https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning#12_reuploadingpqc_layer_using_controlledpqc
    """

    # Alias for circuit builder function.
    build_circuit = variational_encoding_pqc

    def __init__(self,
        observables: cirq.PauliSum | list[cirq.PauliSum],
        qubits: QubitListType,
        n_layers: int,
        n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
        variational_layer_fn: VariationalCircuitFunctionType = variational_rotation_layer,
        entangling_layer_fn: EntanglingCircuitFunctionType = lambda qubits: neighbor_entangling_layer(qubits, gate=cirq.CNOT),
        symbol_superscript_index: int = None,
        quantum_data_circuit_fn: Callable[[QubitListType], Iterable[Any]] = None,
        activation: str = 'linear',
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.activation = activation

        # Get circuit generator function and symbol tuple.
        circuit_gen_fn, (symbols_theta, symbols_enc) = VariationalEncodingPQC.build_circuit(
            qubits=qubits,
            n_layers=n_layers,
            n_var_rotations=n_var_rotations,
            variational_layer_fn=variational_layer_fn,
            entangling_layer_fn=entangling_layer_fn,
            symbol_superscript_index=symbol_superscript_index,
        )
        
        # Build circuit and add as TFQ layer.
        circuit = cirq.Circuit(circuit_gen_fn())
        self.pqc = tfq.layers.ControlledPQC(circuit, observables)

        # Define circuit for quantum data.
        # In most cases there will only be classical data, so the `quantum_data_circuit_fn` will be an empty circuit.
        self.quantum_data = tfq.convert_to_tensor([
            cirq.Circuit(),
            cirq.Circuit(quantum_data_circuit_fn(qubits) if quantum_data_circuit_fn is not None else []),
        ])

        # Define explicit symbol order.
        symbols = [str(symbol) for symbol in (symbols_theta + symbols_enc)]
        self.n_symbols = len(symbols)
        self.indices = tf.constant([symbols.index(name) for name in sorted(symbols)]) # Define explicit symbol ordering.

        # Define TensorFlow variables for symbols.
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(symbols_theta))),
            dtype='float32',
            trainable=True,
            name='theta',
        )
        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init,
            dtype='float32',
            trainable=True,
            name='lambda',
        )

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs[0]), 0) # Collect batch dimension.
        tiled_up_circuits = tf.repeat(self.quantum_data, repeats=batch_dim) # Repeat quantum data circuit for each batch.
        tiled_up_theta = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        
        scaled_inputs = tf.einsum('i,ji->ji', self.lmbd, tiled_up_inputs)
        squashed_inputs = keras.layers.Activation(self.activation)(scaled_inputs)
        
        joined_vars = tf.concat([tiled_up_theta, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        
        return self.pqc([tiled_up_circuits, joined_vars])