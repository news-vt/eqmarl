from __future__ import annotations
import cirq
import numpy as np

## Functions to create various entangled input states.

def entangle_agents_phi_plus(qubits: list[cirq.LineQubit], d: int, n: int):
    """Entangles via $\\Phi^+$."""
    for i in range(d):
        yield cirq.H(qubits[i])
        for j in range(n-1):
            yield cirq.CNOT(qubits[j*d + i], qubits[(j+1)*d + i])


def entangle_agents_phi_minus(qubits: list[cirq.LineQubit], d: int, n: int):
    """Entangles via $\\Phi^-$."""
    for i in range(d):
        yield cirq.X(qubits[i])
        yield cirq.H(qubits[i])
        for j in range(n-1):
            yield cirq.CNOT(qubits[j*d + i], qubits[(j+1)*d + i])


def entangle_agents_psi_plus(qubits: list[cirq.LineQubit], d: int, n: int):
    """Entangles via $\\Psi^+$."""
    for i in range(d):
        yield cirq.H(qubits[i])
        for j in range(n-1):
            yield cirq.X(qubits[(j+1)*d + i])
            yield cirq.CNOT(qubits[j*d + i], qubits[(j+1)*d + i])


def entangle_agents_psi_minus(qubits: list[cirq.LineQubit], d: int, n: int):
    """Entangles via $\\Psi^-$."""
    for i in range(d):
        yield cirq.X(qubits[i])
        yield cirq.H(qubits[i])
        for j in range(n-1):
            yield cirq.X(qubits[(j+1)*d + i])
            yield cirq.CNOT(qubits[j*d + i], qubits[(j+1)*d + i])


def prepare_state_from_state_vector(
    qubits: list[cirq.LineQubit],
    target_state: np.ndarray, name: str = "StatePreparation",
    ):
    """Prepares an arbitrary state vector.
    
    This function wraps the built-in `cirq.StatePreparationChannel` to accommodate a list of qubits. 
    """
    yield cirq.StatePreparationChannel(target_state=target_state, name=name)(*qubits)