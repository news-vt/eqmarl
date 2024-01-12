from __future__ import annotations
from .types import *
from typing import Any, Type
import pennylane as qml
from pennylane.operation import Operation
from pennylane import numpy as np
from .tools import flatten_to_operations


## Functions to create various entangled input states.

def entangle_agents_phi_plus(wires: WireListType, d: int, n: int, op: Operation = qml.CNOT):
    """Entangles via $\\Phi^+$."""
    for i in range(d):
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_phi_minus(wires: WireListType, d: int, n: int, op: Operation = qml.CNOT):
    """Entangles via $\\Phi^-$."""
    for i in range(d):
        qml.PauliX(wires=wires[i])
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_psi_plus(wires: WireListType, d: int, n: int, op: Operation = qml.CNOT):
    """Entangles via $\\Psi^+$."""
    for i in range(d):
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.PauliX(wires=wires[(j+1)*d + i])
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_psi_minus(wires: WireListType, d: int, n: int, op: Operation = qml.CNOT):
    """Entangles via $\\Psi^-$."""
    for i in range(d):
        qml.PauliX(wires=wires[i])
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.PauliX(wires=wires[(j+1)*d + i])
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def prepare_state_from_state_vector(
    wires: WireListType,
    state: np.ndarray,
    ):
    """Prepares an arbitrary state from state-vector representation.
    
    This function wraps the built-in `qml.StatePrep` to accommodate a list of qubits. 
    """
    qml.StatePrep(state=state, wires=wires)
    # yield cirq.StatePreparationChannel(target_state=target_state, name=name)(*qubits)


## Variational / Encoding / Entangling operations


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
        
        # Ensure weights have proper shape.
        req_shape = self.shape(wires, operations)
        n_req_shape = len(req_shape)
        weights_shape = qml.math.shape(weights)
        n_weights_shape = len(weights_shape)
        assert n_weights_shape == n_req_shape or n_weights_shape == n_req_shape + 1, (
            f"Weights tensor must be {n_req_shape}-dimensional with shape {req_shape}"
            f"or {n_req_shape+1}-dimensional if batching; got shape {weights_shape}"
        )
        
        # Validate operations.
        operations = operations or self.operations
        assert len(operations) > 0, 'at least one operation is required'
        
        self._hyperparameters = {"operations": operations}
        super().__init__(weights, wires=wires, id=id)

    # @staticmethod
    @classmethod
    def compute_decomposition(cls,
        weights: np.tensor,
        wires: WireListType,
        operations: list[Type[Operation]],
        ):

        # Decompose rotations into operations.
        op_list = []
        for i, wire in enumerate(wires):
            for j, op in enumerate(operations):
                op_list.append(op(weights[..., i, j], wires=wire))

        return op_list

    @classmethod
    def shape(cls,
        wires: WireListType,
        operations: list[Type[Operation]] = None,
        ):
        """Returns tuple of (n_wires, n_operations).
        
        If no operations are provided then defaults to class operations.
        
        Note that the returned shape does not include a batch dimension.
        """

        # Use default operations for class instance if none were provided.
        if operations is None: 
            operations = cls.operations
        
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



# def variational_pqc(
#     wires: WireListType,
#     n_layers: int,
#     n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
#     variational_layer_fn: VariationalCircuitFunctionType = variational_rotation_layer,
#     entangling_layer_fn: EntanglingCircuitFunctionType = lambda wires: neighbor_entangling_layer(wires, gate=qml.CNOT),
#     symbol_superscript_index: int = None,
#     ):
#     """Simple parameterized variational circuit.

#     Contains variational layer with Rx, Ry, Rz rotations parameterized by $\theta$, followed by a next-neighbor entanglement layer.
#     """
#     d = len(wires) # Dimension of qubits.
    
#     # Variational parameters.
#     var_thetas = sympy.symbols(f"theta{f'^{{({symbol_superscript_index})}}' if symbol_superscript_index is not None else ''}(0:{n_var_rotations*(n_layers+1)*d})") # Add +1 here because there will be a final variational layer at the end.
#     var_thetas = np.asarray(var_thetas).reshape((n_layers+1, d, n_var_rotations))
    
#     for l in range(n_layers):
#         variational_layer_fn(wires, var_thetas[l])
#         entangling_layer_fn(wires)

#     return gen_circuit, (var_thetas.flatten().tolist(),)


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
            op_list.extend(flatten_to_operations(variational_layer(weights=weights_var[..., l, :, :], wires=wires))) # Uses `...` notation to account for possible batch dimension.
            op_list.extend(flatten_to_operations(entangling_layer(wires=wires)))
            
            # Encoding layer.
            op_list.extend(flatten_to_operations(encoding_layer(weights=weights_enc[..., l, :, :], wires=wires))) # Uses `...` notation to account for possible batch dimension.

        # Last variational layer at the end.
        op_list.extend(flatten_to_operations(variational_layer(weights=weights_var[..., l+1, :, :], wires=wires))) # Uses `...` notation to account for possible batch dimension.

        return op_list

    @classmethod
    def shape(cls,
        n_layers: int,
        wires: WireListType,
        variational_layer: Type[ParameterizedOperation] = None,
        encoding_layer: Type[ParameterizedOperation] = None,
        ):
        """Returns tuple of (shape_var, shape_enc).
        
        Note that the returned shapes do not include a batch dimension.
        """

        # Compute shape for single variational layer.
        shape_var = (variational_layer or cls._hyperparameters["variational_layer"]).shape(wires)

        # Compute shape for all variational layers.
        shape_var = (n_layers + 1, *shape_var) # +1 because there is one additional variational layer at the end.

        # Compute shape for single encoding layer.
        shape_enc = (encoding_layer or cls._hyperparameters["encoding_layer"]).shape(wires)
        
        # Compute shape for all encoding layers.
        shape_enc = (n_layers, *shape_enc)
        
        return shape_var, shape_enc