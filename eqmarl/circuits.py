import pennylane as qml

from .types import WireListType



class QuantumCircuit:
    """Base class for quantum circuits intended to be used with PennyLane API.
    
    This class allows for custom circuits to be created with support for state variables (such as `self.wires` and `self.n_wires`).
    The addition of state allows for reduced parameters when calling the class instance like a function.
    
    Example usage is:
    >>> wires = list(range(4))
    >>> circuit = Circuit(wires=wires)
    >>> dev = qml.device('default.qubit', wires=len(wires))
    >>> qnode = qml.QNode(func=circuit, device=dev)
    >>> qnode()
    
    Or can create a qnode directly using the class methods:
    >>> wires = list(range(4))
    >>> circuit = Circuit(wires=wires)
    >>> dev = circuit.device('default.qubit')
    >>> qnode = circuit.qnode(device=dev)
    >>> qnode()
    """
    
    def __init__(self, 
        wires: WireListType,
        ):
        assert isinstance(wires, (list, tuple)), f'Wires must either be a list or tuple; got {wires}'
        self.wires = wires
        self.n_wires = len(wires)
    
    def __call__(self, inputs = None):
        """Construct the circuit and pass parameters.
        
        This function must abide by the constraints for all PennyLane quantum functions (see https://docs.pennylane.ai/en/stable/introduction/circuits.html#quantum-functions).
        
        The argument `inputs` is required for compatibility with with `KerasLayer` or `TorchLayer`. The default `None` value allows this class to be used without specifying inputs (if designed with proper error handling). The index of `inputs` within the argument sequence does not matter, just as long as the name is reserved.
        """
        raise NotImplementedError()

    def device(self, *args, **kwargs):
        """Helper for creating a `qml.device` object from the current circuit."""
        return qml.device(*args, wires=self.wires, **kwargs)
    
    def qnode(self, *args, device: qml.Device | str | dict = None, **kwargs):
        """Helper for creating a `qml.QNode` object from the current circuit.
        
        The `device` argument can either be a dictionary
        """
        # Create a basic default device if none was specified.
        if device is None:
            device = self.device(name='default.qubit')
        # Create device from name string.
        elif isinstance(device, str):
            device = self.device(name=device)
        # Create device from keyword arguments.
        elif isinstance(device, dict):
            device = self.device(**device)
        # Default case is to treat device as a `qml.Device` object.
        assert list(device.wires) == list(self.wires), 'Circuit wires must match device wires'

        # Create quantum node using the current circuit as a callable.
        kwargs = dict(**kwargs, device=device) # Update kwargs to include device.
        return qml.QNode(self, *args, **kwargs)

    @property
    def shape(self):
        raise NotImplementedError()
    
    @property
    def output_shape(self):
        """Returns number of observables at output.
        
        This is useful in combination with `qml.KerasLayer`.
        
        Note 1: The returned shape does not include the batch dimension.
        """
        raise NotImplementedError()

    @property
    def input_shape(self):
        """Returns required shape for `inputs` argument.

        Note 1: The returned shape does not include the batch dimension.
        Note 2: PennyLane will compress inputs to 2D when batching for usage with `KerasLayer` or `TorchLayer`.
        """
        raise NotImplementedError()

    @staticmethod
    def get_shape(*args, **kwargs):
        raise NotImplementedError()