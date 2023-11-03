from .types import *
from .layers import *
from functools import reduce
import cirq
import tensorflow.keras as keras



def ClassicalCriticDense(units: list[int], activation: str = 'relu', **kwargs) -> keras.Model:
    assert type(units) == list, 'units must be a list of integers'
    layers = [keras.layers.Dense(u, activation=activation) for u in units]
    layers += [keras.layers.Dense(1, activation=None, name='v')] # Value function estimator V(s).
    model = keras.Sequential(layers=layers, **kwargs)
    return model


def QuantumActorVariationalEncoding(
    qubits: QubitListType,
    n_actions: int,
    n_layers: int,
    beta: float,
    **kwargs, # All extra args flow into model init.
    ) -> keras.Model:
    
    # Observables define the measurement basis for the PQC.
    # In this case, use a global Pauli product of $Z0 * Z1 * Z2 * \dots$.
    ops = [cirq.Z(q) for q in qubits] # Z0 Z1 Z2 ...
    observables = [reduce((lambda x, y: x * y), ops)] # Z0 * Z1 * Z2 * ...

    # Define layers.
    input_tensor = tf.keras.Input(shape=(len(qubits),), dtype='float32', name='input')
    pqc = VariationalEncodingPQC(
        observables=observables,
        qubits=qubits,
        n_layers=n_layers,
        name='pqc',
    )
    policy = WeightedAlternatingSoftmaxPolicy(
        beta=beta,
        n_actions=n_actions,
        name='policy',
    )

    # Create model.
    x = input_tensor
    x = pqc([x])
    x = policy(x)
    model = keras.Model(inputs=input_tensor, outputs=x, **kwargs)
    return model