from __future__ import annotations
from typing import Any, Type
import numpy as np
import cirq
from ..tools import flatten_to_operations


## Functions to create various entangled input states.

def entangle_agents_phi_plus(qubits: list, d: int, n: int, op: cirq.Gate = cirq.CNOT) -> list[cirq.Operation]:
    """Entangles via $\\Phi^+$."""
    ops = []
    for i in range(d):
        ops.append(cirq.H(qubits[i]))
        for j in range(n-1):
            ops.append(op(qubits[j*d + i], qubits[(j+1)*d + i]))
    return ops


def entangle_agents_phi_minus(qubits: list, d: int, n: int, op: cirq.Gate = cirq.CNOT) -> list[cirq.Operation]:
    """Entangles via $\\Phi^-$."""
    ops = []
    for i in range(d):
        ops.append(cirq.X(qubits[i]))
        ops.append(cirq.H(qubits[i]))
        for j in range(n-1):
            ops.append(op(qubits[j*d + i], qubits[(j+1)*d + i]))
    return ops


def entangle_agents_psi_plus(qubits: list, d: int, n: int, op: cirq.Gate = cirq.CNOT) -> list[cirq.Operation]:
    """Entangles via $\\Psi^+$."""
    ops = []
    for i in range(d):
        ops.append(cirq.H(qubits[i]))
        for j in range(n-1):
            ops.append(cirq.X(qubits[(j+1)*d + i]))
            ops.append(op(qubits[j*d + i], qubits[(j+1)*d + i]))
    return ops


def entangle_agents_psi_minus(qubits: list, d: int, n: int, op: cirq.Gate = cirq.CNOT) -> list[cirq.Operation]:
    """Entangles via $\\Psi^-$."""
    ops = []
    for i in range(d):
        ops.append(cirq.X(qubits[i]))
        ops.append(cirq.H(qubits[i]))
        for j in range(n-1):
            ops.append(cirq.X(qubits[(j+1)*d + i]))
            ops.append(op(qubits[j*d + i], qubits[(j+1)*d + i]))
    return ops


def prepare_state_from_state_vector(
    qubits: list,
    state: np.ndarray,
    **kwargs,
    ):
    """Prepares an arbitrary state from state-vector representation.
    
    This function wraps the function `cirq.StatePreparationChannel` to accommodate a list of qubits.
    """
    # qml.StatePrep(state=state, wires=wires)
    return cirq.StatePreparationChannel(target_state=state, **kwargs)(*qubits)


## Variational / Encoding / Entangling operations


def circular_entangling_layer(
    qubits: list,
    gate: cirq.Gate = cirq.CZ,
    ) -> list[cirq.Operation]:
    """Entangles a list of qubits with their next-neighbor in circular fashion (i.e., ensures first and last qubit are also entangled)."""
    ops = []
    for q0, q1 in zip(qubits, qubits[1:]):
        ops.append(gate(q0, q1))
    if len(qubits) != 2:
        ops.append(gate(qubits[0], qubits[-1])) # Entangle the first and last qubit.
    return ops


def neighbor_entangling_layer(
    qubits: list,
    gate: cirq.Gate = cirq.CNOT,
    ) -> list[cirq.Operation]:
    """Entangles a list of qubits with their next-neighbor (does not entangle first and last qubit)."""
    return [gate(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]


class ParameterizedOperationGate(cirq.Gate):
    """
    Applies a sequence of gates corresponding to a parameter matrix. By default gates are assumed to operate on single qubits. Every row of the parameter matrix is a separate set of qubits on which to apply the operation sequence.
    """
    
    operations: list[cirq.Gate] = []
    
    def __init__(self, params: np.ndarray, name: str = None):
        super().__init__()
        self.params = params
        self.name = name or self.__class__.__name__
        
    def _num_qubits_(self):
        return self.params.shape[0]

    def _decompose_(self, qubits):
        # Decompose rotations into operations.
        for i, q in enumerate(qubits):
            for j, op in enumerate(self.operations):
                yield op(self.params[..., i, j])(q)
                
    def _circuit_diagram_info_(self, args):
        return [f'{self.name}({self.params[i]})' for i in range(self.params.shape[0])]
    
    @classmethod
    def get_shape(cls,
        n_qubits: int,
        operations: list[cirq.Gate] = None,
        ):
        """Returns tuple of (n_wires, n_operations).
        
        If no operations are provided then defaults to class operations.
        
        Note that the returned shape does not include a batch dimension.
        """

        # Use default operations for class instance if none were provided.
        if operations is None: 
            operations = cls.operations

        return (n_qubits, len(operations),)


class ParameterizedRotationLayer_Rx(ParameterizedOperationGate):
    """Parameterized rotation layer using Rx.
    """
    operations = [cirq.rx]


class ParameterizedRotationLayer_RxRyRz(ParameterizedOperationGate):
    """Parameterized rotation layer using Rx, Ry, Rz.
    """
    operations = [cirq.rx, cirq.ry, cirq.rz]


VariationalRotationLayer = ParameterizedRotationLayer_RxRyRz # Default is 3 rotation sequence RX, RY, RZ.
EncodingLayer = ParameterizedRotationLayer_Rx # Default is 1 rotation Rx.