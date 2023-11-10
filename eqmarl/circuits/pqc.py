from __future__ import annotations
from ..types import *
from typing import Any, Callable, Generator, Iterable, Type
import cirq
# import numpy as np
import sympy
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as np
from ..tools import flatten_to_operations


## Adapted from: https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning

def variational_rotation_3(
    wire: WireType,
    symbols: tuple[float, float, float],
    ) -> list[Operation]:
    """Applies 3 rotation gates (Rx, Ry, Rz) to a single qubit using provided symbols for parameterization."""
    return [
        qml.RX(phi=symbols[0], wires=wire),
        qml.RY(phi=symbols[1], wires=wire),
        qml.RZ(phi=symbols[2], wires=wire),
    ]


def variational_rotation_layer(
    wires: WireListType,
    symbols: SymbolMatrixType,
    variational_rotation_fn: Callable[[QubitType, SymbolListType], Any] = variational_rotation_3,
    ) -> list[Operation]:
    return [variational_rotation_fn(wire, symbols[i]) for i, wire in enumerate(wires)]


def circular_entangling_layer(
    wires: WireListType,
    gate: Operation = qml.CZ,
    ) -> list[Operation]:
    """Entangles a list of qubits with their next-neighbor in circular fashion (i.e., ensures first and last qubit are also entangled)."""
    ops = []
    for w0, w1 in zip(wires, wires[1:]):
        ops.append(gate(wires=[w0, w1]))
    if len(wires) != 2:
        ops.append(gate(wires=[wires[0], wires[-1]])) # Entangle the first and last qubit.
    return ops


def neighbor_entangling_layer(
    wires: WireListType,
    gate: Operation = qml.CNOT,
    ) -> list[Operation]:
    """Entangles a list of qubits with their next-neighbor (does not entangle first and last qubit)."""
    return [gate(w0, w1) for w0, w1 in zip(wires, wires[1:])]


def single_rotation_encoding_layer(
    wires: WireListType,
    symbols: SymbolListType,
    gate: Operation = qml.RX,
    ) -> list[Operation]:
    return [gate(symbols[i], wires=wire) for i, wire in enumerate(wires)]


class ParameterizedOperation(Operation):
    """Performs a list of parameterized operations on each qubit.
    
    Implements `shape()` to determine parameter shapes.
    """
    operations: list[type[Operation]] = [] # Default is no operations, which will throw an error; users must override this or provide an operations list at runtime.
    
    def __init__(self, 
        weights: np.tensor,
        wires: WireListType,
        id: str = None,
        operations: list[type[Operation]] = None,
        ):
        self._hyperparameters = {"operations": operations or self.operations}
        super().__init__(weights, wires=wires, id=id)

    @classmethod
    def compute_decomposition(cls, 
        weights: np.tensor,
        wires: WireListType,
        operations: list[Type[Operation]] = None,
        ):
        # Ensure weights have proper shape.
        req_shape = cls.shape(wires, operations)
        assert weights.numpy().shape == req_shape, f'parameters must have shape {req_shape}'
        
        # Use default rotations for class instance if none were provided.
        if operations is None: 
            operations = cls.operations

        # Decompose rotations into operations.
        op_list = []
        for i, wire in enumerate(wires):
            for j, op in enumerate(operations):
                op_list.append(op(weights[i, j], wires=wire))

        return op_list

    @classmethod
    def shape(cls, wires: WireListType, operations: list[Type[Operation]] = None):

        # Use default operations for class instance if none were provided.
        if operations is None: 
            operations = cls.operations
        
        assert len(operations) > 0, 'at least one operation is required'
        
        if isinstance(wires, (int, str)):
            wires = [wires]
        
        return (len(wires), len(operations),)


class VariationalRotationLayer(ParameterizedOperation):
    """Parameterized variational rotation layer.
    
    Implements `shape()` to determine parameter shapes.
    """
    operations = [qml.RX, qml.RY, qml.RZ] # Default is 3 rotation sequence RX, RY, RZ.


class EncodingLayer(ParameterizedOperation):
    """Parameterized variational rotation layer.
    
    Implements `shape()` to determine parameter shapes.
    """
    operations = [qml.RX]



def variational_pqc(
    wires: WireListType,
    n_layers: int,
    n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
    variational_layer_fn: VariationalCircuitFunctionType = variational_rotation_layer,
    entangling_layer_fn: EntanglingCircuitFunctionType = lambda wires: neighbor_entangling_layer(wires, gate=qml.CNOT),
    symbol_superscript_index: int = None,
    ):
    """Simple parameterized variational circuit.

    Contains variational layer with Rx, Ry, Rz rotations parameterized by $\theta$, followed by a next-neighbor entanglement layer.
    """
    d = len(wires) # Dimension of qubits.
    
    # Variational parameters.
    var_thetas = sympy.symbols(f"theta{f'^{{({symbol_superscript_index})}}' if symbol_superscript_index is not None else ''}(0:{n_var_rotations*(n_layers+1)*d})") # Add +1 here because there will be a final variational layer at the end.
    var_thetas = np.asarray(var_thetas).reshape((n_layers+1, d, n_var_rotations))
    
    for l in range(n_layers):
        variational_layer_fn(wires, var_thetas[l])
        entangling_layer_fn(wires)

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



class VariationalEncodingPQC(Operation):
    _hyperparameters = {
        "variational_layer": VariationalRotationLayer,
        "encoding_layer": EncodingLayer,
        "entangling_layer": circular_entangling_layer,
    }
    
    def __init__(self,
        weights_var: np.tensor,
        weights_enc: np.tensor,
        n_layers: int,
        wires: WireListType,
        variational_layer: Type[ParameterizedOperation] = None,
        encoding_layer: Type[ParameterizedOperation] = None,
        entangling_layer: Type[Operation] = None,
        id: str = None,
        ):
        self._hyperparameters = {
            "variational_layer": variational_layer or self._hyperparameters["variational_layer"],
            "encoding_layer": encoding_layer or self._hyperparameters["encoding_layer"],
            "entangling_layer": entangling_layer or self._hyperparameters["entangling_layer"],
        }
        super().__init__(weights_var, weights_enc, n_layers, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(
        weights_var: np.tensor,
        weights_enc: np.tensor,
        n_layers: int,
        wires: WireListType,
        variational_layer: Type[ParameterizedOperation],
        encoding_layer: Type[ParameterizedOperation],
        entangling_layer: Type[Operation],
        ):

        op_list = []
        for l in range(n_layers):
            # Variational layer.
            op_list.extend(flatten_to_operations(variational_layer(weights=weights_var[l], wires=wires)))
            op_list.extend(flatten_to_operations(entangling_layer(wires=wires)))
            
            # Encoding layer.
            op_list.extend(flatten_to_operations(encoding_layer(weights=weights_enc[l], wires=wires)))

        # Last variational layer at the end.
        op_list.extend(flatten_to_operations(variational_layer(weights=weights_var[l], wires=wires)))

        return op_list

    @classmethod
    def shape(cls,
        n_layers: int,
        wires: WireListType,
        variational_layer: Type[ParameterizedOperation] = None,
        encoding_layer: Type[ParameterizedOperation] = None,
        ):
        """Returns tuple of (shape_var, shape_enc)."""

        # Compute shape for single variational layer.
        shape_var = (variational_layer or cls._hyperparameters["variational_layer"]).shape(wires)

        # Compute shape for all variational layers.
        shape_var = (n_layers + 1, *shape_var) # +1 because there is one additional variational layer at the end.

        # Compute shape for single encoding layer.
        shape_enc = (encoding_layer or cls._hyperparameters["encoding_layer"]).shape(wires)
        
        # Compute shape for all encoding layers.
        shape_enc = (n_layers, *shape_enc)
        
        return shape_var, shape_enc