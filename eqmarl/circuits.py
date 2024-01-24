import numpy as np
import sympy
import cirq

from .ops import *


def generate_partite_variational_encoding_circuit(
    qubits: list,
    n_parts: int, # Number of partitions.
    d_qubits: int, # Number of qubits per partition.
    n_layers: int,
    decompose: bool = False,
    variational_layer_cls: ParameterizedOperationGate = VariationalRotationLayer,
    encoding_layer_cls: ParameterizedOperationGate = EncodingLayer,
    ) -> tuple[cirq.Circuit, tuple[np.ndarray,...]]:
    
    shape_var = variational_layer_cls.get_shape(d_qubits)
    shape_enc = encoding_layer_cls.get_shape(d_qubits)
    
    ### Define weights for circuit.
    #
    ## Variational shape
    theta_var = sympy.symbols(f'var^{{(0:{n_parts})}}(0:{n_layers+1})_' + '_'.join(f'(0:{x})' for x in shape_var))
    theta_var = np.asarray(theta_var).reshape((n_parts, n_layers+1, *shape_var))
    ## Encoding shape
    theta_enc = sympy.symbols(f'enc^{{(0:{n_parts})}}(0:{n_layers})_' + '_'.join(f'(0:{x})' for x in shape_enc))
    theta_enc = np.asarray(theta_enc).reshape((n_parts, n_layers, *shape_enc))
    
    # Build the circuit.
    # circuit = cirq.Circuit()
    ops = []
    
    # Add GHZ entangling layer at the start.
    ops.append(
        entangle_agents_phi_plus(qubits, d_qubits, n_parts)
    )
    
    # Build circuit in partitions.
    for pidx in range(n_parts):
        qidx = pidx * d_qubits # Starting qubit index for the current partition.
    
        for l in range(n_layers):
            # Variational layer.
            ops.append(
                variational_layer_cls(theta_var[pidx, l], name=f'{pidx}-v{l}')(*qubits[qidx:qidx + d_qubits])
            )
            
            # Entangling layer.
            ops.append(
                circular_entangling_layer(qubits[qidx:qidx + d_qubits])
            )
            
            # Encoding layer.
            ops.append(
                encoding_layer_cls(theta_enc[pidx, l], name=f'{pidx}-e{l}')(*qubits[qidx:qidx + d_qubits])
            )
        # Last variational layer at the end.
        ops.append(
            variational_layer_cls(theta_var[pidx, l+1], name=f'v{l+1}')(*qubits[qidx:qidx + d_qubits])
        )
    
    # Decompose circuit into minimal gate representation.
    # This is required when custom gates are implemented for use with TensorFlowQuantum
    if decompose:
        ops = [cirq.decompose(o) for o in cirq.ops.flatten_to_ops(ops)]

    circuit = cirq.Circuit(ops)

    return circuit, (theta_var, theta_enc)