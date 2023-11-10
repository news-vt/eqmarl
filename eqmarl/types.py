from __future__ import annotations
from typing import Any, Callable, Iterable, Union
import cirq
import sympy


WireType = int | str
WireListType = list[WireType]


###############

QubitType = Union[cirq.LineQubit, cirq.GridQubit]
QubitListType = list[QubitType]

TwoQubitGateFunctionType = Callable[[QubitType, QubitType], Any]
QubitGateFunctionType = Callable[[QubitType], Any]

VariationalCircuitFunctionType = Callable[[QubitListType, list[list[float]]], Any]
EncodingCircuitFunctionType = Callable[[QubitListType, list[float]], Any]
EntanglingCircuitFunctionType = Callable[[QubitListType], Any]
CircuitGeneratorFunctionType = Callable[[], Iterable[Any]] # Generator function that yields gate operations that define a given quantum circuit.

SymbolType = Union[str, sympy.Symbol]
SymbolListType = list[SymbolType]
SymbolMatrixType = list[list[SymbolType]]
SymbolValueDict = dict[SymbolType, Any]
ParameterResolverFunctionType = Callable[[SymbolMatrixType], SymbolValueDict]



ParameterizedCircuitFunctionReturnType = tuple[CircuitGeneratorFunctionType, tuple[SymbolListType,...]]

ParameterizedCircuitFunctionType = Callable[[QubitListType], ParameterizedCircuitFunctionReturnType] # Single-agent variant inputs qubit list.
MultiAgentParameterizedCircuitFunctionType = Callable[[QubitListType, int], ParameterizedCircuitFunctionReturnType] # Multi-agent variant inputs qubit list and agent index.

InitialStatePrepCircuitFunctionType = Callable[[QubitListType], Iterable[Any]] # Single-agent variant of a initial state preparation function.
MultiAgentInitialStatePrepCircuitFunctionType = Callable[[QubitListType, int, int], Iterable[Any]] # Multi-agent variant of initial state preparation function that requires qubit list, qubit dimension `d`, and number of agents `n`.