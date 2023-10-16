# eqmarl
eQMARL: Quantum multi-agent reinforcement learning with entangled agents.

## Example Circuits

Entanglement Type | Circuit
----------------- | -------
None | ![](images/pqc_2_agents_2_qubits_1_layers_no_entanglement.svg?raw=true)
$\ket{\Phi^{+}}$ | ![](images/pqc_2_agents_2_qubits_1_layers_entangled_phi_plus.svg?raw=true)
$\ket{\Psi^{+}}$ | ![](images/pqc_2_agents_2_qubits_1_layers_entangled_psi_plus.svg?raw=true)


## Current Results

The following table shows images from RL agents using PQCs with encoding and variational layers (as shown above) with variations on the entanglement type and parameter choices.

Entanglement Type | Random $\theta$, Random $s$ | Identical $\theta$, Identical $s$ | Nearly Identical $\theta$, Identical $s$ |
------------------| -------| --------- | ---------------- |
None | ![](images/no_entanglement_random_theta_random_s.png?raw=true) | ![](images/no_entanglement_identical_theta_identical_s.png?raw=true) | ![](images/no_entanglement_nearly_identical_theta_identical_s.png?raw=true)
$\ket{\Phi^{+}}$ | ![](images/entangled_phi_plus_random_theta_random_s.png?raw=true) | ![](images/entangled_phi_plus_identical_theta_identical_s.png?raw=true) | ![](images/entangled_phi_plus_nearly_identical_theta_identical_s.png?raw=true)
$\ket{\Psi^{+}}$ | ![](images/entangled_psi_plus_random_theta_random_s.png?raw=true) | ![](images/entangled_psi_plus_identical_theta_identical_s.png?raw=true) | ![](images/entangled_psi_plus_nearly_identical_theta_identical_s.png?raw=true)