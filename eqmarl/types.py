from __future__ import annotations
from typing import Any, Callable, Iterable
import cirq
import sympy

QubitType = cirq.LineQubit | cirq.GridQubit
QubitListType = list[QubitType]

TwoQubitGateFunctionType = Callable[[QubitType, QubitType], Any]
QubitGateFunctionType = Callable[[QubitType], Any]

VariationalCircuitFunctionType = Callable[[QubitListType, list[list[float]]], Any]
EncodingCircuitFunctionType = Callable[[QubitListType, list[float]], Any]
EntanglingCircuitFunctionType = Callable[[QubitListType], Any]
CircuitGeneratorFunctionType = Callable[[], Iterable[Any]] # Generator function that yields gate operations that define a given quantum circuit.

SymbolType = str | sympy.Symbol
SymbolListType = list[SymbolType]
SymbolMatrixType = list[list[SymbolType]]
SymbolValueDict = dict[SymbolType, Any]
ParameterResolverFunctionType = Callable[[SymbolMatrixType], SymbolValueDict]

ParameterizedPolicyCircuitFunctionReturnType = tuple[CircuitGeneratorFunctionType, tuple[SymbolListType,...]]