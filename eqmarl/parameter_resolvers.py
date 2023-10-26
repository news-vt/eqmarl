from .types import *
import numpy as np


class ParamResolverBase:
    """Base class for parameter resolvers that require stateful operation.
    
    Ensures that all child classes abide by the typing: `ParameterResolverFunctionType`
    """
    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
        raise NotImplementedError()


## Define parameter resolvers for the variational-only policy.

class ParamResolverRandomTheta(ParamResolverBase):
    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
        """Resolver for random variational thetas and random encoder inputs."""
        var_thetas_matrix = symbols
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        param_dict = {
            **{symbol: var_thetas_values.flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
        }
        return param_dict


class ParamResolverIdenticalTheta(ParamResolverBase):
    """Resolver for identical variational thetas.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
        var_thetas_matrix = symbols
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        if self.var_thetas_values is None:
            self.var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        param_dict = {
            **{symbol: self.var_thetas_values.flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
        }
        return param_dict


class ParamResolverNearlyIdenticalTheta(ParamResolverBase):
    """Resolver for nearly identical variational thetas.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None

    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
        var_thetas_matrix = symbols
        var_thetas_matrix = np.asarray(var_thetas_matrix)
        if self.var_thetas_values is None:
            self.var_thetas_values = np.random.uniform(low=0., high=np.pi, size=var_thetas_matrix.shape)
        thetas_offset = np.random.uniform(low=0., high=0.5, size=var_thetas_matrix.shape) # Random theta offset (so that thetas between runs are nearly identical).
        param_dict = {
            **{symbol: (self.var_thetas_values + thetas_offset).flatten()[i] for i, symbol in enumerate(var_thetas_matrix.flatten())},
        }
        return param_dict


## Define parameter resolvers for this variational + encoding policy.


class ParamResolverRandomThetaRandomS(ParamResolverBase):
    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
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

class ParamResolverIdenticalThetaIdenticalS(ParamResolverBase):
    """Resolver for identical variational thetas and identical encoder inputs.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
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

class ParamResolverNearlyIdenticalThetaIdenticalS(ParamResolverBase):
    """Resolver for nearly identical variational thetas and identical encoder inputs.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
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

class ParamResolverNearlyIdenticalThetaRandomS(ParamResolverBase):
    """Resolver for nearly identical variational thetas and random encoder inputs.
    
    This is written as a class that overloads `__call__` to act like a function because the function needs to define the random values based on the shape of the inputs the first time it is called.
    """
    def __init__(self):
        self.var_thetas_values = None
        self.enc_inputs_values = None

    def __call__(self, symbols: SymbolMatrixType) -> SymbolValueDict:
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