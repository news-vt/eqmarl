from itertools import chain
from typing import (
    Sequence,
    Type,
    Union,
)
import pennylane as qml
from pennylane import numpy as np # Must import Numpy from PennyLane.
from pennylane.operation import Operation


def flatten_to_operations(op: Union[Operation,Sequence[Operation]]) -> list[Operation]:
    """Flattens a nested sequence of operations into a single list."""
    # Single operation, so return list of size 1.
    if isinstance(op, Operation):
        return [op]
    # Sequence of operations, so convert to list and return.
    elif hasattr(op, '__iter__'):
        return list(chain.from_iterable(flatten_to_operations(o) for o in op))
    # Return operation as-is.
    else:
        raise ValueError(f'operation must be one of {{{Operation}, hasattr(__iter__)}} but received {type(op)}')



def circular_entangling_layer(
    wires: list[int],
    gate: Type[Operation] = qml.CZ,
    ) -> list[Operation]:
    """Entangles a list of qubits with their next-neighbor in circular fashion (i.e., ensures first and last qubit are also entangled)."""
    ops = []
    for w0, w1 in zip(wires, wires[1:]):
        ops.append(gate(wires=[w0, w1]))
    if len(wires) != 2:
        ops.append(gate(wires=[wires[0], wires[-1]])) # Entangle the first and last qubit.
    return ops


## Functions to create various entangled input states.

def entangle_agents_phi_plus(wires: list, d: int, n: int, op: Type[Operation] = qml.CNOT):
    """Entangles via $\\Phi^+$."""
    for i in range(d):
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_phi_minus(wires: list, d: int, n: int, op: Type[Operation] = qml.CNOT):
    """Entangles via $\\Phi^-$."""
    for i in range(d):
        qml.PauliX(wires=wires[i])
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_psi_plus(wires: list, d: int, n: int, op: Type[Operation] = qml.CNOT):
    """Entangles via $\\Psi^+$."""
    for i in range(d):
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.PauliX(wires=wires[(j+1)*d + i])
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])


def entangle_agents_psi_minus(wires: list, d: int, n: int, op: Type[Operation] = qml.CNOT):
    """Entangles via $\\Psi^-$."""
    for i in range(d):
        qml.PauliX(wires=wires[i])
        qml.Hadamard(wires=wires[i])
        for j in range(n-1):
            qml.PauliX(wires=wires[(j+1)*d + i])
            op(wires=[wires[j*d + i], wires[(j+1)*d + i]])




class ParameterizedOperation(Operation):
    """Performs a list of parameterized operations on each qubit.
    
    Implements `shape()` to determine parameter shapes.
    """
    operations: list[type[Operation]] = [] # Default is no operations, which will throw an error; users must override this or provide an operations list at runtime.
    
    def __init__(self, 
        weights: np.tensor,
        wires: list[int],
        id: str = None,
        operations: list[type[Operation]] = None,
        ):
        
        # Ensure weights have proper shape.
        req_shape = self.get_shape(wires, operations)
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
        wires: list[int],
        operations: list[Type[Operation]],
        ):

        # Decompose rotations into operations.
        op_list = []
        for i, wire in enumerate(wires):
            for j, op in enumerate(operations):
                op_list.append(op(weights[..., i, j], wires=wire))

        return op_list

    @classmethod
    def get_shape(cls,
        n_wires: int,
        operations: list[Type[Operation]] = None,
        ):
        """Returns tuple of (n_wires, n_operations).
        
        If no operations are provided then defaults to class operations.
        
        Note that the returned shape does not include a batch dimension.
        """

        # Use default operations for class instance if none were provided.
        if operations is None: 
            operations = cls.operations
        
        return (n_wires, len(operations),)


class ParameterizedRotationLayer_Rx(ParameterizedOperation):
    """Parameterized rotation layer using Rx.
    """
    operations = [qml.RX]


class ParameterizedRotationLayer_RxRyRz(ParameterizedOperation):
    """Parameterized rotation layer using Rx, Ry, Rz.
    """
    operations = [qml.RX, qml.RY, qml.RZ]


VariationalRotationLayer = ParameterizedRotationLayer_RxRyRz # Default is 3 rotation sequence RX, RY, RZ.
EncodingLayer = ParameterizedRotationLayer_Rx # Default is 1 rotation Rx.


class VariationalEncodingParameterizedOperation(Operation):
    """Parameterized quantum operation with variational and encoding layers.
    
    This is implemented as an instance of the `Operation` class to allow for easy drop-in use in larger quantum circuits. If you want to create a variational and encoding circuit using this operation, see the `..circuits.pennylane_circuits.VariationalEncodingCircuit` class.
    """
    _hyperparameters = {
        "variational_layer": VariationalRotationLayer,
        "encoding_layer": EncodingLayer,
        "entangling_layer": circular_entangling_layer,
    }
    
    def __init__(self,
        weights_var: np.tensor,
        weights_enc: np.tensor,
        n_layers: int,
        wires: list[int],
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
        wires: list[int],
        variational_layer: Type[ParameterizedOperation],
        encoding_layer: Type[ParameterizedOperation],
        entangling_layer: Type[Operation],
        ):

        op_list = []
        for l in range(n_layers):
            # Variational layer.
            op_list.extend(flatten_to_operations(variational_layer(weights=weights_var[..., l, :, :], wires=wires))) # Uses `...` notation to account for possible batch dimension.
            
            # Entangling layer.
            op_list.extend(flatten_to_operations(entangling_layer(wires=wires)))
            
            # Encoding layer.
            op_list.extend(flatten_to_operations(encoding_layer(weights=weights_enc[..., l, :, :], wires=wires))) # Uses `...` notation to account for possible batch dimension.

        # Last variational layer at the end.
        op_list.extend(flatten_to_operations(variational_layer(weights=weights_var[..., l+1, :, :], wires=wires))) # Uses `...` notation to account for possible batch dimension.

        return op_list

    @classmethod
    def get_shape(cls,
        n_layers: int,
        wires: list[int],
        variational_layer: Type[ParameterizedOperation] = None,
        encoding_layer: Type[ParameterizedOperation] = None,
        ):
        """Returns tuple of (shape_var, shape_enc).
        
        Note that the returned shapes do not include a batch dimension.
        """
        
        n_wires = len(wires)

        # Compute shape for single variational layer.
        shape_var = (variational_layer or cls._hyperparameters["variational_layer"]).get_shape(n_wires)

        # Compute shape for all variational layers.
        shape_var = (n_layers + 1, *shape_var) # +1 because there is one additional variational layer at the end.

        # Compute shape for single encoding layer.
        shape_enc = (encoding_layer or cls._hyperparameters["encoding_layer"]).get_shape(n_wires)
        
        # Compute shape for all encoding layers.
        shape_enc = (n_layers, *shape_enc)
        
        return shape_var, shape_enc