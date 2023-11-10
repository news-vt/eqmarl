from __future__ import annotations
import pennylane as qml
import numpy as np
from ..types import *

## Functions to create various entangled input states.

def entangle_agents_phi_plus(wires: WireListType, d: int, n: int):
    """Entangles via $\\Phi^+$."""
    for i in range(d):
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.CNOT(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_phi_minus(wires: WireListType, d: int, n: int):
    """Entangles via $\\Phi^-$."""
    for i in range(d):
        qml.PauliX(wires=wires[i])
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.CNOT(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_psi_plus(wires: WireListType, d: int, n: int):
    """Entangles via $\\Psi^+$."""
    for i in range(d):
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.PauliX(wires=wires[(j+1)*d + i])
            qml.CNOT(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_psi_minus(wires: WireListType, d: int, n: int):
    """Entangles via $\\Psi^-$."""
    for i in range(d):
        qml.PauliX(wires=wires[i])
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.PauliX(wires=wires[(j+1)*d + i])
            qml.CNOT(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def prepare_state_from_state_vector(
    wires: WireListType,
    state: np.ndarray,
    ):
    """Prepares an arbitrary state from state-vector representation.
    
    This function wraps the built-in `qml.StatePrep` to accommodate a list of qubits. 
    """
    qml.StatePrep(state=state, wires=wires)
    # yield cirq.StatePreparationChannel(target_state=target_state, name=name)(*qubits)