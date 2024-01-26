import cirq
import functools
import itertools


def permute_observables(observables: list[list]) -> list:
    """Permutes lists of Pauli observables.
    
    Given a list of lists `[a, b, ...]` where `a = [a0, a1, ...]` and `b = [b0, b1, ...]` this function creates a new permuted list `[a0*b0, a0*b1, ..., a1*b0, a1*b1, ...]`
    """
    return [
        functools.reduce(lambda x, y: x*y, obs)
        for obs in itertools.product(*observables)
        ]


def make_observables_CoinGame2(qubits: list) -> list:
    """Quantum observables used in CoinGame2."""
    return [
        cirq.Z(qubits[0]),
        cirq.Z(qubits[1]),
        cirq.Z(qubits[2]),
        cirq.Z(qubits[3]),
    ]

def make_observables_CartPole(qubits: list) -> list:
    """Quantum observables for CartPole environment.
    
    Observables are:
    - Z0 * Z1
    - Z2 * Z3
    """
    return [
        cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
        cirq.Z(qubits[2]) * cirq.Z(qubits[3]),
    ]

def make_observables_CartPole_alternating(qubits: list) -> list:
    """Alternating quantum observables for CartPole environment.
    
    Observables are:
    - Z0 * Z1 * Z2 * Z3
    - -(Z0 * Z1 * Z2 * Z3)
    """
    return [
        cirq.Z(qubits[0]) * cirq.Z(qubits[1]) * cirq.Z(qubits[2]) * cirq.Z(qubits[3]),
        -cirq.Z(qubits[0]) * cirq.Z(qubits[1]) * cirq.Z(qubits[2]) * cirq.Z(qubits[3]),
    ]