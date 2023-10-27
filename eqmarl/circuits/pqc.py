from __future__ import annotations
from ..types import *
from typing import Any, Callable, Generator, Iterable
import cirq
import numpy as np
import sympy


## Adapted from: https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning

def variational_rotation_3(qubit: QubitType, symbols: tuple[float, float, float]) -> tuple[Any, Any, Any]:
    """Applies 3 rotation gates (Rx, Ry, Rz) to a single qubit using provided symbols for parameterization."""
    return [
        cirq.rx(symbols[0])(qubit),
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit),
    ]

def variational_rotation_layer(
    qubits: QubitListType,
    symbols: SymbolMatrixType,
    variational_rotation_fn: Callable[[QubitType, SymbolListType], Any] = variational_rotation_3,
    ):
    return [variational_rotation_fn(qubit, symbols[i]) for i, qubit in enumerate(qubits)]


def circular_entangling_layer(
    qubits: QubitListType,
    gate: TwoQubitGateFunctionType = cirq.CZ,
    ) -> Generator:
    """Entangles a list of qubits with their next-neighbor in circular fashion (i.e., ensures first and last qubit are also entangled)."""
    yield [gate(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    if len(qubits) != 2:
        yield gate(qubits[0], qubits[-1]) # Entangle the first and last qubit.


def neighbor_entangling_layer(
    qubits: QubitListType,
    gate: TwoQubitGateFunctionType = cirq.CNOT,
    ) -> Generator:
    """Entangles a list of qubits with their next-neighbor (does not entangle first and last qubit)."""
    yield [gate(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]


def single_rotation_encoding_layer(
    qubits: QubitListType,
    symbols: SymbolListType,
    gate: QubitGateFunctionType = cirq.rx,
    ) -> Any:
    yield [gate(symbols[i])(qubit) for i, qubit in enumerate(qubits)]


def variational_pqc(
    qubits: QubitListType,
    n_layers: int,
    n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
    variational_layer_fn: VariationalCircuitFunctionType = variational_rotation_layer,
    entangling_layer_fn: EntanglingCircuitFunctionType = lambda qubits: neighbor_entangling_layer(qubits, gate=cirq.CNOT),
    symbol_superscript_index: int = None,
    ) -> ParameterizedCircuitFunctionReturnType:
    """Simple parameterized variational circuit.

    Contains variational layer with Rx, Ry, Rz rotations parameterized by $\theta$, followed by a next-neighbor entanglement layer.
    """
    d = len(qubits) # Dimension of qubits.
    
    # Variational parameters.
    var_thetas = sympy.symbols(f"theta{f'^{{({symbol_superscript_index})}}' if symbol_superscript_index is not None else ''}(0:{n_var_rotations*(n_layers+1)*d})") # Add +1 here because there will be a final variational layer at the end.
    var_thetas = np.asarray(var_thetas).reshape((n_layers+1, d, n_var_rotations))

    # Define generator to build circuit.
    # This allows the caller to define circuit parameters.
    def gen_circuit() -> Iterable[Any]:
        for l in range(n_layers):
            yield variational_layer_fn(qubits, var_thetas[l])
            yield entangling_layer_fn(qubits)

    return gen_circuit, (var_thetas.flatten().tolist(),)



def variational_encoding_pqc(
    qubits: QubitListType,
    n_layers: int,
    n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
    variational_layer_fn: VariationalCircuitFunctionType = variational_rotation_layer,
    entangling_layer_fn: EntanglingCircuitFunctionType = lambda qubits: circular_entangling_layer(qubits, gate=cirq.CZ),
    encoding_layer_fn: EncodingCircuitFunctionType = lambda qubits, symbols: single_rotation_encoding_layer(qubits, symbols, gate=cirq.rx),
    symbol_superscript_index: int = None
    ) -> ParameterizedCircuitFunctionReturnType:
    """More complex parameterized variational + encoding circuit.

    Contains variational layers with Rx, Ry, Rz rotations parameterized by $\theta$, followed by a next-neighbor entanglement layer, followed by an encoding layer to encode the state $s$. The final layer in the circuit is a variational layer.
    """
    d = len(qubits) # Dimension of qubits.
    
    # Variational parameters.
    var_thetas = sympy.symbols(f"theta{f'^{{({symbol_superscript_index})}}' if symbol_superscript_index is not None else ''}(0:{n_var_rotations*(n_layers+1)*d})") # Add +1 here because there will be a final variational layer at the end.
    var_thetas = np.asarray(var_thetas).reshape((n_layers+1, d, n_var_rotations))
    
    # Encoding parameters.
    enc_inputs = sympy.symbols(f"x{f'^{{({symbol_superscript_index})}}' if symbol_superscript_index is not None else ''}(0:{n_layers})_(0:{d})")
    enc_inputs = np.asarray(enc_inputs).reshape((n_layers, d))
    
    # Define generator to build circuit.
    # This allows the caller to define circuit parameters.
    def gen_circuit() -> Iterable[Any]:
        for l in range(n_layers):
            # Variational layer.
            yield variational_layer_fn(qubits, var_thetas[l])
            yield entangling_layer_fn(qubits)
            
            # Encoding layer.
            yield encoding_layer_fn(qubits, enc_inputs[l])
            
        # Last variational layer at the end.
        yield variational_layer_fn(qubits, var_thetas[n_layers])
    
    return gen_circuit, (var_thetas.flatten().tolist(), enc_inputs.flatten().tolist())





