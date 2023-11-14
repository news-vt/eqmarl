from __future__ import annotations
from contextlib import contextmanager
from matplotlib import pyplot as plt
from typing import Any, Callable, Iterable
import cirq
import pandas as pd
import seaborn as sns
import sympy
from .types import *
from .circuits import MARLCircuit, QuantumCircuit
from .math import softmax
from .observables import *

import pennylane as qml
# from pennylane import numpy as np
import numpy as np # Must import numpy last to ensure no conflicts with PennyLane wrapper.
from numpy.typing import NDArray

# This flag globally disables all quantum circuit tests within the function `test_runner` (defined below).
# Use this flag to prevent lots of tests from running and taking up significant execution time.
GLOBAL_TEST_FLAG = False


@contextmanager
def testing_context(*args, **kwargs):
    """Context manager that explicitly allows using test functions within the context.
    
    Enables testing within the context, and disables it after, thereby prevent any future tests from being run outside of the context.
    """
    global GLOBAL_TEST_FLAG
    original_flag = GLOBAL_TEST_FLAG # Preserve original flag value.
    try:
        GLOBAL_TEST_FLAG = True # Enable testing.
        yield
    finally:
        GLOBAL_TEST_FLAG = original_flag # Reset the flag to original value.


def plot_state_histogram_all(samples, axis, d: int, n: int, key: str = 'all', color: str|tuple[float, float, float] = None):
    hist_all = samples.histogram(key=key)
    df_hist_all = pd.DataFrame.from_records([hist_all], columns=list(range(2**(d*n)))).melt(value_vars=list(range(2**(d*n)))).rename(columns={'variable': 'qubit state', 'value': 'result count'})
    if color is None:
        color = sns.color_palette()[0]
    sns.barplot(df_hist_all, x='qubit state', y='result count', ax=axis, color=color)
    ctr = samples.histogram(key=key)
    mean = np.mean([v for k, v in ctr.items()])
    std = np.std([v for k, v in ctr.items()])
    axis.axhline(y=mean, color='red', linestyle='--', label='mean')
    axis.axhline(y=mean+std, color='green', linestyle='--', label='mean+std')
    axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
    axis.xaxis.set_major_formatter(lambda x, pos: f"{int(x):0{d*n}b}")
    axis.legend()
    return axis


def plot_state_histogram_agents(samples, axis, d: int, n: int, key_prefix: str = 'agent', colors: list[str|tuple[float,float,float]] = None):
    hist_agents = [samples.histogram(key=f"{key_prefix}{k}") for k in range(n)]
    df_hist_agents = pd.DataFrame.from_records(hist_agents, columns=list(range(2**d)))
    df_hist_agents = df_hist_agents\
        .rename_axis(index='agent')\
        .reset_index()\
        .melt(id_vars='agent', value_vars=list(range(2**d)))\
        .rename(columns={'variable': 'qubit state', 'value': 'result count'})
    if colors is None:
        colors = sns.color_palette()[1:] # Remove first one because it is assumed to be reserved for the "all" plot.
    sns.barplot(df_hist_agents, x='qubit state', y='result count', hue='agent', ax=axis, palette=colors)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=90)
    axis.xaxis.set_major_formatter(lambda x, pos: f"{int(x):0{d}b}")
    return axis


def build_parameterized_system_circuit(
    d: int, 
    n: int, 
    policy_fn: MultiAgentParameterizedCircuitFunctionType,
    initial_state_prep_fn: MultiAgentInitialStatePrepCircuitFunctionType = None,
    meas_key_all: str = 'all',
    meas_key_prefix_agent: str = 'agent',
    meas_flag: bool = True, # Denotes whether to perform measurements (True) or not (False).
    ) -> tuple[cirq.Circuit, list[list[list[sympy.Symbol]]]]:
    """Constructs an parameterized circuit representing the entire system (all agents).
    """
    qubits = cirq.LineQubit.range(d * n)
    circuit = cirq.Circuit()
    # var_thetas, enc_inputs = [], []
    symbol_list = []
    if initial_state_prep_fn is not None:
        circuit.append(initial_state_prep_fn(qubits))
    for aidx in range(n):
        qidx = aidx * d # Starting qubit index for the specified agent.
        
        # Policy circuit.
        agent_circuit_gen_fn, agent_symbol_list = policy_fn(qubits[qidx:qidx+d], aidx)
        symbol_list.append(agent_symbol_list) # Preserve symbols for agent.
        circuit.append(agent_circuit_gen_fn())
        
        # Measure (only if requested).
        if meas_flag: circuit.append(cirq.measure(qubits[qidx:qidx+d], key=f"{meas_key_prefix_agent}{aidx}")) # Measure agent qubits.
    if meas_flag: circuit.append(cirq.measure(qubits, key=meas_key_all)) # Store measurement of entire system circuit.
    return circuit, symbol_list


def simulate_parameterized_circuit(
    circuit: cirq.Circuit,
    symbol_dict: dict,
    repetitions: int = 100,
    ) -> tuple[float, dict]:
    param_resolver = cirq.ParamResolver(symbol_dict)

    # Compute entanglement entropy of output state.
    extra_data_fields = {}
    extra_data_fields['rho'] = cirq.final_density_matrix(circuit, param_resolver=param_resolver)
    extra_data_fields['entropy'] = cirq.von_neumann_entropy(extra_data_fields['rho'])
    
    # Simulate circuit and generate histogram of measurement outcomes.
    sim = cirq.Simulator()
    samples = sim.run(circuit, repetitions=repetitions, param_resolver=param_resolver)
    return samples, extra_data_fields

def test_runner(
    d: int,
    n: int,
    policy_fn: MultiAgentParameterizedCircuitFunctionType,
    symbol_resolver_fn: ParameterResolverFunctionType = None,
    figure_kwargs: dict = dict(layout='constrained', figsize=(20,14)),
    subfigure_kwargs: dict = dict(nrows=2, ncols=2, wspace=0.09, hspace=0.09),
    figure_title: str = None,
    repetitions: int = 100,
    initial_state_prep_fn: MultiAgentInitialStatePrepCircuitFunctionType = None,
    plot_all_histogram: bool = True,
    plot_agent_histogram: bool = True,
    return_sim_results: bool = False,
    ) -> plt.Figure|tuple[plt.Figure,list[tuple[dict, dict]]]:
    assert plot_all_histogram or plot_agent_histogram, 'must designate at least one histogram plot'

    # Default resolver returns empty dictionary.
    if symbol_resolver_fn is None:
        symbol_resolver_fn = lambda *args, **kwargs: {}
    
    # Nullify all return values if testing is disabled.
    if not GLOBAL_TEST_FLAG:
        fig, all_sim_results = None, None

    else:
        fig = plt.figure(**figure_kwargs)
        subfigs = fig.subfigures(**subfigure_kwargs)
        subfigs = subfigs.flatten()
        fig.suptitle(figure_title)
        
        n_circuits = len(subfigs)
        all_sim_results: list[tuple[dict, dict]] = []
        for i in range(n_circuits):
            circuit, symbol_list = build_parameterized_system_circuit(
                d=d,
                n=n,
                policy_fn=policy_fn,
                initial_state_prep_fn=initial_state_prep_fn,
                meas_key_all='all',
                meas_key_prefix_agent='agent',
            )

            # Build parameter dictionary.
            # Resolves parameter name to value.
            symbol_dict = symbol_resolver_fn(symbol_list)

            # Simulate.
            samples, extra_data_fields = simulate_parameterized_circuit(circuit=circuit, symbol_dict=symbol_dict, repetitions=repetitions)
            all_sim_results.append((samples, extra_data_fields)) # Append the tuple of samples and extra fields.

            # Plot measurement histogram for all qubits and qubits for each agent.
            if plot_agent_histogram and plot_all_histogram:
                axs = subfigs[i].subplots(2, 1).flatten()
                plot_state_histogram_all(samples, axs[0], d, n, key='all')
                plot_state_histogram_agents(samples, axs[1], d, n, key_prefix='agent')
            else:
                axs = [subfigs[i].subplots(1, 1)]
                if plot_all_histogram:
                    plot_state_histogram_all(samples, axs[0], d, n, key='all')
                elif plot_agent_histogram:
                    plot_state_histogram_agents(samples, axs[0], d, n, key_prefix='agent')
            
            extra_key = 'entropy'
            if extra_key in extra_data_fields: # Display entropy.
                axs[0].set_title(f"{extra_key}={extra_data_fields[extra_key]:.4f}")

    if return_sim_results:
        return fig, all_sim_results
    else:
        # Return the completed figure.
        return fig


def batched_experiment(
    func: Callable[[int], Any],
    repetitions: int = 100,
    batch_size: int = 16,
    ) -> list:
    """Calls a given function `repetitions` times in chunks of `batch_size` and returns collective results of call history.
    
    Target function requires one argument which is the current chunk size.
    
    Note that chunks may be different sizes based upon total number of repetitions.
    """

    # Divide runs into batched chunks.
    history = []
    for i in range(0, repetitions, batch_size):
        # Ensure chunk size never causes total number of runs to exceed maximum.
        chunk_size = min(repetitions - i, batch_size)
        
        # Call function for current batch.
        x = func(chunk_size)
        
        # Preserve result.
        history.extend(x)

    return history


def postprocess_circuit_measurements(x: NDArray) -> tuple[NDArray, NDArray[np.int32]]:
    """Postprocessing operations for circuit measurement results.
    
    Performs the following:
    - Softmax activation along last axis.
    - Obtains index where softmax results are maximum on last axis.
    - Determines number of unique combinations of argmax states across all measurement runs.
    """
    x = softmax(x, axis=-1)
    x = np.argmax(x, axis=-1)
    states, counts = np.unique(np.asarray(x), axis=0, return_counts=True)
    return states, counts