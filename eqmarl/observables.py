import pennylane as qml
# from pennylane import numpy as np
import numpy as np
from numpy.typing import NDArray
from pennylane.operation import Operation
from functools import reduce


def TensorPauliZ(wires, n, d) -> list:
    all_obs = []
    for aidx in range(n):
        qidx = aidx * d # Starting qubit index for the specified agent.
        ops = [qml.PauliZ(w) for w in wires[qidx:qidx+d]]
        obs = reduce((lambda a, b: a @ b), ops)
        all_obs.append(obs)
        all_obs.append(-obs)
    return all_obs


def IndividualPauliZ(wires) -> list:
    all_obs = []
    for w in wires:
        obs = qml.PauliZ(w)
        all_obs.append(obs)
    return all_obs



def PauliObservables(wires: list, op: Operation = qml.PauliZ) -> NDArray:
    """Creates a list of observables by applying the specified Pauli operator (defaults to PauliZ) on each wire."""
    return np.asarray([op(wires=w) for w in wires])


def TensorObservables(obs: NDArray) -> NDArray:
    """Creates a single observable via tensor product of a list of observables.
    
    Input observables can either be 1-dimensional list, or nested lists.
    
    Returns a single 1-dimensional list containing tensor product of all observables. If input is a nested list, then each returned entry is the tensor product of the corresponding nested list.
    """
    obs = np.asarray(obs) # Ensure type is numpy array.
    if len(obs.shape) > 1:
        obs = [TensorObservables(o) for o in obs]
    else:
        obs = [reduce((lambda a, b: a @ b), obs)] # Reduce to tensor product.
    return np.asarray(obs).reshape((-1,)) # Ensure final output is 1-dimensional. 


def WeightedObservables(obs: NDArray, weights: NDArray) -> NDArray:
    """Multiplies a list/matrix of weights by a list/matrix of observables.
    
    Creates a weighted Hamiltonian for each observable.
    
    Returns a 1-dimensional list of weighted observables.
    """
    obs = np.asarray(obs) # Ensure type is numpy array.
    if len(obs.shape) != 2: # Ensures shape is (n_obs, 1)
        obs = np.expand_dims(obs.reshape((-1,)), axis=-1)
    res = weights * obs
    return res.reshape((-1,)) # Ensure final output is 1-dimensional.


def AlternatingWeightedObservables(obs: NDArray, n_reps: int, weight: float = -1.) -> list:
    """Duplicates observables in alternating fashion with $(-1)^i * O$ weight where $i$ is the repetition index.
    
    Returns a 1-dimensional list of weighted observables.
    """
    weights = np.asarray([weight**i for i in range(n_reps)])
    res = WeightedObservables(weights=weights, obs=obs)
    return res.reshape((-1,)) # Ensure final output is 1-dimensional.


def GroupedTensorObservables(
    wires: list,
    n_groups: int,
    d_qubits: int,
    op: Operation = qml.PauliZ,
    ) -> NDArray:
    """Creates Pauli tensor product observables for groups of qubits.
    
    This is useful for multi-agent reinforcement learning (MARL).
    """
    all_obs = []
    for gidx in range(n_groups):
        qidx = gidx * d_qubits # Starting qubit index for the specified group.
        obs = PauliObservables(wires=wires[qidx:qidx+d_qubits], op=op) # Z0, Z1, ..., Z(d-1)
        obs = TensorObservables(obs) # Z0 @ Z1 @ ... @ Z(d-1)
        all_obs.extend(obs)
    return np.asarray(all_obs)


def AlternatingGroupedTensorObservables(
    wires: list,
    n_groups: int,
    d_qubits: int,
    n_reps: int,
    weight: float = -1.,
    op: Operation = qml.PauliZ,
    ) -> NDArray:
    """Duplicates grouped Pauli tensor product observables for groups of qubits.
    
    This is useful for multi-agent reinforcement learning (MARL).
    """
    obs = GroupedTensorObservables(
        wires=wires,
        n_groups=n_groups,
        d_qubits=d_qubits,
        op=op,
    )
    obs = AlternatingWeightedObservables(
        obs=obs,
        n_reps=n_reps,
        weight=weight,
    )
    return obs