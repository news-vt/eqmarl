# eqmarl
eQMARL: Quantum multi-agent reinforcement learning with entangled agents.

## Example Circuits

Entanglement Type | Circuit
----------------- | -------
None | ![](https://github.com/news-vt/eqmarl/blob/main/images/pqc_2_agents_2_qubits_1_layers_no_entanglement.svg?raw=true&sanitize=true)
$\ket{\Phi^{+}}$ | ![](https://github.com/news-vt/eqmarl/blob/main/images/pqc_2_agents_2_qubits_1_layers_entangled_phi_plus.svg?raw=true&sanitize=true)
$\ket{\Psi^{+}}$ | ![](https://github.com/news-vt/eqmarl/blob/main/images/pqc_2_agents_2_qubits_1_layers_entangled_psi_plus.svg?raw=true&sanitize=true)


## Current Results

The following table shows images from RL agents using PQCs with encoding and variational layers (as shown above) with variations on the entanglement type and parameter choices.

Entanglement Type | Random $\theta$, Random $s$ | Identical $\theta$, Identical $s$ | Nearly Identical $\theta$, Identical $s$ | Nearly Identical $\theta$, Random $s$
------------------| ----------------------------| --------------------------------- | ---------------------------------------- | -------------------------------------
None | ![](https://github.com/news-vt/eqmarl/blob/main/images/no_entanglement_random_theta_random_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/no_entanglement_identical_theta_identical_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/no_entanglement_nearly_identical_theta_identical_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/no_entanglement_nearly_identical_theta_random_s.png?raw=true)
$\ket{\Phi^{+}}$ | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_phi_plus_random_theta_random_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_phi_plus_identical_theta_identical_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_phi_plus_nearly_identical_theta_identical_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_phi_plus_nearly_identical_theta_random_s.png?raw=true)
$\ket{\Psi^{+}}$ | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_psi_plus_random_theta_random_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_psi_plus_identical_theta_identical_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_psi_plus_nearly_identical_theta_identical_s.png?raw=true) | ![](https://github.com/news-vt/eqmarl/blob/main/images/entangled_psi_plus_nearly_identical_theta_random_s.png?raw=true)

## Installation

Installation of this repo is a little finicky because of the requirements for `tensorflow-quantum` on various systems. See instructions below for your particular use case.

## macOS

If using macOS bare-metal then you can use Anaconda (or another virtual environment manager) but it is required that you have Python `>=3.7,<=3.8` (from our configuration attempts we found there is no compatible `tensorflow-quantum` package for Python `3.9`).

```bash
$ conda create -n myenv python=3.8
$ conda activate myenv
$ pip install -r requirements.txt -r requirements-dev.txt
```

## Linux

If using Linux then you are able to use Python `>=3.7,<=3.9`.

## Docker

The repo comes with a Docker configuration via `Dockerfile` (see [Dockerfile](./Dockerfile)) using the `python:3.9` base image.

## VSCode Devcontainer

The repo also comes with a VSCode devcontainer setup (see [devcontainer.json](./.devcontainer/devcontainer.json)) if you want to work within the Docker environment and use VSCode's editing capabilities natively.