from __future__ import annotations
from ..circuits.policy import *
from ..types import *
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow.keras as keras


class VariationalPolicyPQC(keras.layers.Layer):
    """Simple parameterized variational policy.
    
    Tensorflow layer representation of `parameterized_variational_policy_circuit`.
    
    Contains variational layer with Rx, Ry, Rz rotations parameterized by $\theta$, followed by a next-neighbor entanglement layer.
    """

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
        name: str = 'VariationalPolicyPQC',
        ):
        super().__init__(name=name)
        self.n_qubits = len(qubits)
        self.activation = activation

        # Get circuit generator function and symbol tuple.
        circuit_gen_fn, (symbols_theta,) = parameterized_variational_policy_circuit(
            qubits=qubits,
            n_layers=n_layers,
            n_var_rotations=n_var_rotations,
            variational_layer_fn=variational_layer_fn,
            entangling_layer_fn=entangling_layer_fn,
            symbol_superscript_index=symbol_superscript_index,
        )
        
        # Build circuit and add as TFQ layer.
        circuit = cirq.Circuit(circuit_gen_fn(qubits))
        self.pqc = tfq.layers.ControlledPQC(circuit, observables)

        # Define circuit for quantum data.
        # In most cases there will only be classical data, so the `quantum_data_circuit_fn` will be an empty circuit.
        self.quantum_data = tfq.convert_to_tensor([
            cirq.Circuit(),
            quantum_data_circuit_fn(qubits) if quantum_data_circuit_fn is not None else [],
        ])

        # Define explicit symbol order.
        symbols = [str(symbol) for symbol_list in symbols_theta for symbol in symbol_list]
        self.n_symbols = len(symbols)

        # Define TensorFlow variables for symbols.
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(symbols_theta))),
            dtype='float32',
            trainable=True,
            name='theta',
        )

        # Define explicit symbol ordering.
        self.indices = tf.constant([symbols.index(name) for name in sorted(symbols)])

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs[0]), 0) # Collect batch dimension.
        tiled_up_circuits = tf.repeat(self.quantum_data, repeats=batch_dim) # Repeat quantum data circuit for each batch.
        tiled_up_theta = tf.tile(self.theta, multiples=[batch_dim, 1])
        
        joined_vars = tiled_up_theta
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        
        return self.pqc([tiled_up_circuits, joined_vars])

# class PolicyPQC(keras.layers.Layer):

#     def __init__(self,
#         qubits: QubitListType,
#         # n_layers: int,
#         # n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
#         # variational_layer_fn: VariationalCircuitFunctionType = variational_rotation_layer,
#         # entangling_layer_fn: EntanglingCircuitFunctionType = lambda qubits: neighbor_entangling_layer(qubits, gate=cirq.CNOT),
#         # symbol_superscript_index: int = None,
#         policy_circuit_fn: ParameterizedPolicyCircuitFunctionType, # User specifies simple function that builds policy circuit.
#         observables: cirq.PauliSum | list[cirq.PauliSum],
#         quantum_data_circuit_fn: Callable[[QubitListType], Iterable[Any]] = None,
#         activation: str = 'linear',
#         name: str = 'VariationalPolicyPQC',
#         ):
#         super().__init__(name=name)
#         self.n_qubits = len(qubits)
#         self.activation = activation

#         # Get circuit generator function and symbol tuple.
#         circuit_gen_fn, symbol_tuple = policy_circuit_fn(qubits)
        
#         # Build circuit and add as TFQ layer.
#         circuit = cirq.Circuit(circuit_gen_fn(qubits))
#         self.pqc = tfq.layers.ControlledPQC(circuit, observables)

#         # Define circuit for quantum data.
#         # In most cases there will only be classical data, so the `quantum_data_circuit_fn` will be an empty circuit.
#         self.quantum_data = tfq.convert_to_tensor([
#             cirq.Circuit(),
#             quantum_data_circuit_fn(qubits) if quantum_data_circuit_fn is not None else [],
#         ])

#         # Define explicit symbol order.
#         symbols = [str(symbol) for symbol_list in symbol_tuple for symbol in symbol_list]
#         self.n_symbols = len(symbols)

#         # Define TensorFlow variables for symbols.
#         self.variables = tf.Variable(
#             initial_value=tf.zeros(shape=(self.n_symbols,)),
#             dtype='float32',
#             trainable=True,
#             name='variables',
#         )

#         # Define explicit symbol ordering.
#         self.indices = tf.constant([symbols.index(name) for name in sorted(symbols)])

#     def call(self, inputs):
#         batch_dim = tf.gather(tf.shape(inputs[0]), 0) # Collect batch dimension.
#         tiled_up_circuits = tf.repeat(self.quantum_data, repeats=batch_dim) # Repeat quantum data circuit for each batch.
#         tiled_up_variables = tf.tile(self.variables, multiples=[batch_dim, 1])
#         ## ...
        
#         return self.pqc([tiled_up_circuits, self.variables])