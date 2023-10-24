from __future__ import annotations
from typing import Any, Callable, Generator, Iterable
import cirq
import numpy as np
import sympy


## Adapted from: https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning

def variational_rotation_3(qubit: cirq.LineQubit, symbols: tuple[float, float, float]) -> tuple[Any, Any, Any]:
    """Applies 3 rotation gates (Rx, Ry, Rz) to a single qubit using provided symbols for parameterization."""
    return [
        cirq.rx(symbols[0])(qubit),
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit),
    ]

def variational_rotation_layer(
    qubits: list[cirq.LineQubit],
    symbols: list[list[float]],
    variational_rotation_fn: Callable[[cirq.LineQubit, list[float]], Any] = variational_rotation_3,
    ):
    return [variational_rotation_fn(qubit, symbols[i]) for i, qubit in enumerate(qubits)]


def circular_entangling_layer(
    qubits: list[cirq.LineQubit],
    gate: Callable[[cirq.LineQubit, cirq.LineQubit], Any] = cirq.CZ,
    ) -> Generator:
    """Entangles a list of qubits with their next-neighbor in circular fashion (i.e., ensures first and last qubit are also entangled)."""
    yield [gate(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    if len(qubits) != 2:
        yield gate(qubits[0], qubits[-1]) # Entangle the first and last qubit.


def neighbor_entangling_layer(
    qubits: list[cirq.LineQubit],
    gate: Callable[[cirq.LineQubit, cirq.LineQubit], Any] = cirq.CNOT,
    ) -> Generator:
    """Entangles a list of qubits with their next-neighbor (does not entangle first and last qubit)."""
    yield [gate(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]


def single_rotation_encoding_layer(
    qubits: list[cirq.LineQubit],
    symbols: list[float],
    gate: Callable[[cirq.LineQubit], Any] = cirq.rx,
    ) -> Any:
    yield [gate(symbols[i])(qubit) for i, qubit in enumerate(qubits)]


def parameterized_variational_policy_circuit(
    qubits: list,
    n_layers: int,
    n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
    variational_layer_fn: Callable[[list[cirq.LineQubit], list[list[float]]], Any] = variational_rotation_layer,
    entangling_layer_fn: Callable[[list[cirq.LineQubit]], Any] = lambda qubits: neighbor_entangling_layer(qubits, gate=cirq.CNOT),
    symbol_superscript_index: int = None
    ) -> tuple[Callable[[], Iterable[Any]], list[list]]:
    """Simple parameterized variational policy.

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


## Define parameter resolvers for this variational-only policy.

class ParamResolverRandomTheta:
    def __call__(self, symbols: list[list[sympy.Symbol]]):
        """Resolver for random variational thetas and random encoder inputs."""
        var_thetas_matrix = symbols
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        param_dict = {
            **{symbol: var_thetas_values.flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
        }
        return param_dict

class ParamResolverIdenticalTheta:
    """Resolver for identical variational thetas.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: list[list[sympy.Symbol]]):
        var_thetas_matrix = symbols
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        if self.var_thetas_values is None:
            self.var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        param_dict = {
            **{symbol: self.var_thetas_values.flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
        }
        return param_dict


class ParamResolverNearlyIdenticalTheta:
    """Resolver for nearly identical variational thetas.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None

    def __call__(self, symbols: list[list[sympy.Symbol]]):
        var_thetas_matrix = symbols
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        if self.var_thetas_values is None:
            self.var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        thetas_offset = np.random.uniform(low=0., high=0.5, size=var_thetas_matrix.shape) # Random theta offset (so that thetas between runs are nearly identical).
        param_dict = {
            **{symbol: (self.var_thetas_values + thetas_offset).flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
        }
        return param_dict


def parameterized_variational_encoding_policy_circuit(
    qubits: list,
    n_layers: int,
    n_var_rotations: int = 3, # Number of rotational gates to apply for each qubit in the variational layer (e.g., Rx, Ry, Rz).
    variational_layer_fn: Callable[[list[cirq.LineQubit], list[list[float]]], Any] = variational_rotation_layer,
    entangling_layer_fn: Callable[[list[cirq.LineQubit]], Any] = lambda qubits: circular_entangling_layer(qubits, gate=cirq.CZ),
    encoding_layer_fn: Callable[[list[cirq.LineQubit], list[float]], Any] = lambda qubits, symbols: single_rotation_encoding_layer(qubits, symbols, gate=cirq.rx),
    symbol_superscript_index: int = None
    ) -> tuple[Callable[[], Iterable[Any]], list[list]]:
    """More complex parameterized variational + encoding policy.

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


## Define parameter resolvers for this variational + encoding policy.


class ParamResolverRandomThetaRandomS:
    def __call__(self, symbols: list[list[sympy.Symbol]]):
        """Resolver for random variational thetas and random encoder inputs."""
        var_thetas_matrix, enc_inputs_matrix = zip(*symbols) # Unpack list of tuples to tuple of lists.
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        enc_inputs_matrix = np.asarray(enc_inputs_matrix)
        var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        enc_inputs_values = np.random.uniform(low=0., high=np.pi, size=enc_inputs_matrix.shape)
        param_dict = {
            **{symbol: var_thetas_values.flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
            **{symbol: enc_inputs_values.flatten()[i] for i, symbol in enumerate(enc_inputs_matrix.flatten())},
        }
        return param_dict

class ParamResolverIdenticalThetaIdenticalS:
    """Resolver for identical variational thetas and identical encoder inputs.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: list[list[sympy.Symbol]]):
        var_thetas_matrix, enc_inputs_matrix = zip(*symbols) # Unpack list of tuples to tuple of lists.
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        enc_inputs_matrix = np.asarray(enc_inputs_matrix)
        if self.var_thetas_values is None:
            self.var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        if self.enc_inputs_values is None:
            self.enc_inputs_values = np.random.uniform(low=0., high=np.pi, size=enc_inputs_matrix.shape)
        param_dict = {
            **{symbol: self.var_thetas_values.flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
            **{symbol: self.enc_inputs_values.flatten()[i] for i, symbol in enumerate(enc_inputs_matrix.flatten())},
        }
        return param_dict

class ParamResolverNearlyIdenticalThetaIdenticalS:
    """Resolver for nearly identical variational thetas and identical encoder inputs.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: list[list[sympy.Symbol]]):
        var_thetas_matrix, enc_inputs_matrix = zip(*symbols) # Unpack list of tuples to tuple of lists.
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        enc_inputs_matrix = np.asarray(enc_inputs_matrix)
        if self.var_thetas_values is None:
            self.var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        thetas_offset = np.random.uniform(low=0., high=0.5, size=var_thetas_matrix.shape) # Random theta offset (so that thetas between runs are nearly identical).
        if self.enc_inputs_values is None:
            self.enc_inputs_values = np.random.uniform(low=0., high=np.pi, size=enc_inputs_matrix.shape)
        param_dict = {
            **{symbol: (self.var_thetas_values + thetas_offset).flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
            **{symbol: self.enc_inputs_values.flatten()[i] for i, symbol in enumerate(enc_inputs_matrix.flatten())},
        }
        return param_dict

class ParamResolverNearlyIdenticalThetaRandomS:
    """Resolver for nearly identical variational thetas and random encoder inputs.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: list[list[sympy.Symbol]]):
        var_thetas_matrix, enc_inputs_matrix = zip(*symbols) # Unpack list of tuples to tuple of lists.
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        enc_inputs_matrix = np.asarray(enc_inputs_matrix)
        if self.var_thetas_values is None:
            self.var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        thetas_offset = np.random.uniform(low=0., high=0.5, size=var_thetas_matrix.shape) # Random theta offset (so that thetas between runs are nearly identical).
        self.enc_inputs_values = np.random.uniform(low=0., high=np.pi, size=enc_inputs_matrix.shape) # Random state encoding values.
        param_dict = {
            **{symbol: (self.var_thetas_values + thetas_offset).flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
            **{symbol: self.enc_inputs_values.flatten()[i] for i, symbol in enumerate(enc_inputs_matrix.flatten())},
        }
        return param_dict