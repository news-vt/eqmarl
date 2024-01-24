from __future__ import annotations
from typing import Callable, Sequence
import cirq
import numpy as np
import qutip
import sympy
import pennylane as qml
from time import perf_counter
from contextlib import contextmanager
import functools
import itertools


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def permute_observables(observables: list[list]) -> list:
    return [
        functools.reduce(lambda x, y: x*y, obs)
        for obs in itertools.product(*observables)
        ]


def flatten_to_operations(op: cirq.Operation | Sequence[cirq.Operation]) -> list[cirq.Operation]:
    """Flattens a nested sequence of operations into a single list."""
    # Single operation, so return list of size 1.
    if isinstance(op, cirq.Operation):
        return [op]
    # Sequence of operations, so convert to list and return.
    elif hasattr(op, '__iter__'):
        return list(itertools.chain.from_iterable(flatten_to_operations(o) for o in op))
    # Return operation as-is.
    else:
        raise ValueError(f'operation must be one of {{{cirq.Operation}, hasattr(__iter__)}} but received {type(op)}')


def extract_unitary_from_parameterized_circuit(
    circuit: cirq.Circuit,
    symbol_list: list[list[list[sympy.Symbol]]] = None,
    symbol_resolver_fn: Callable[..., dict] = None,
    return_circuit: bool = False,
    ) -> np.ndarray|tuple[np.ndarray, cirq.Circuit]:
    """Constructs a unitary matrix that represents the given circuit.
    
    Provided circuit can either be parameterized or static (i.e., no symbols). If parameterized, then a symbol list and symbol resolver function is required.
    """

    # Default resolver returns empty dictionary.
    if symbol_resolver_fn is None:
        symbol_resolver_fn = lambda *args, **kwargs: {}

    # Resolve symbols to concrete values.
    symbol_dict = symbol_resolver_fn(symbol_list)

    # Create a new circuit with concrete parameter values.
    circuit_resolved = cirq.resolve_parameters(circuit, param_resolver=cirq.ParamResolver(symbol_dict))

    # Extract unitary from resolved circuit.
    U = cirq.unitary(circuit_resolved)
    
    # Return the unitary and the resolved circuit.
    if return_circuit:
        return U, circuit_resolved
    # Otherwise, just return the unitary.
    else:
        return U


def partial_trace_of_state_vector(
    state_vector: np.ndarray,
    keep_dims: int|list[int],
    vector_dims: tuple[list[int],...] = None,
    ) -> np.ndarray:
    """Computes the partial trace of an input state vector with respect to specific dimensions.
    
    The argument `keep_dims` specifies which dimension(s) to preserve during the partial trace.
    For example, with a system consisting of A x B x C, the partial trace with respect to A would have `keep_dims=0` (i.e., trace-out B and C).
    If we instead wanted to trace-out only C, then we would have `keep_dims=[0, 1]` to keep dims for A and B.
    """
    state_vector = np.asarray(state_vector)
    if vector_dims is None:
        n_states = len(state_vector.flatten())
        n_qubits_per_state = int(np.log2(n_states))
        vector_dims = [[2]*n_qubits_per_state, [1]*n_qubits_per_state]
    psi = qutip.Qobj(state_vector, dims=vector_dims)
    return np.asarray(psi.ptrace(sel=keep_dims))