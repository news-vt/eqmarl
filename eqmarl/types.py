from __future__ import annotations
from typing import Any, Callable
import cirq

QubitType = cirq.LineQubit | cirq.GridQubit
QubitListType = list[QubitType]
VariationalCircuitFunctionType = Callable[[QubitListType, list[list[float]]], Any]
EncodingCircuitFunctionType = Callable[[QubitListType, list[float]], Any]
EntanglingCircuitFunctionType = Callable[[QubitListType], Any]
TwoQubitGateFunctionType = Callable[[QubitType, QubitType], Any]
QubitGateFunctionType = Callable[[QubitType], Any]