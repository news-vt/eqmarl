# eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/Python->=3.9,<3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.7.0-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org)

This repository is the official implementation of _eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels_.

## Installation

The codebase is provided as an installable Python package called `eqmarl`. To install the package via `pip`, you can run:

```bash
# Navigate to `eqmarl` source folder.
$ cd path/to/eqmarl/

# Install `eqmarl` package.
$ python -m pip install .
```

You can verify the package was successfully install by running:

```bash
$ python -c "import importlib.metadata; version=importlib.metadata.version('eqmarl'); print(version)"
1.0.0
```

### Requirements

If instead you just want to install the requirements without the package, you can run:

```bash
$ python -m pip install -r requirements.txt -r requirements-dev.txt
```

### Notes on Tensorflow Quantum installation with Anaconda

Installation of this repo can be little finicky because of the requirements for `tensorflow-quantum` on various systems.

If you are using Anaconda to manage Python on macOS, be aware that the version of Python may have been built using an outdated version of macOS. To check this, you can run:

```bash
$ python -c "from distutils import util; print(util.get_platform())"
macosx-10.9-x86_64
```

Notice that in the above example we see the installation of Python was built against `macosx-10.9-x86_64`, whereas the wheel for `tensorflow-quantum` requires `macosx-12.1-x86_64` or later.

To circumvent this, you can download the wheel for `tensorflow-quantum==0.7.2` from here <https://pypi.org/project/tensorflow-quantum/0.7.2/#files> and change the name of the filename from `tensorflow_quantum-0.7.2-cp39-cp39-macosx_12_1_x86_64.whl` to `tensorflow_quantum-0.7.2-cp39-cp39-macosx_10_9_x86_64.whl`. Once you've done that you can install the wheel via:

```bash
# Activate your environment.
$ conda activate myenv

# Install wheel file manually.
$ python -m pip install tensorflow_quantum-0.7.2-cp39-cp39-macosx_10_9_x86_64.whl
```

## Training

To train using the frameworks in the paper, run this command:

```bash
$ python ./scripts/experiment_runner.py ./experiments/<experiment_name>.yml
```

This invokes the [`experiment_runner.py`](./scripts/experiment_runner.py) script, which runs experiments based on YAML configurations. 
Note that the option `-r`/`--n-train-rounds` can be used to train over multiple seed rounds (defaults to 1 round).
The experiment configuration for each of the frameworks discussed in the paper is described as a YAML file in the [experiments](./experiments) folder.

The full list of experiments is as follows:

Experiment YAML File | Description
--- | --- 
[`coingame_maa2c_mdp_eqmarl_noentanglement.yml`](./experiments/coingame_maa2c_mdp_eqmarl_noentanglement.yml) | MDP experiment using $\texttt{eQMARL}$ with $\texttt{None}$ entanglement.
[`coingame_maa2c_mdp_eqmarl_phi+.yml`](./experiments/coingame_maa2c_mdp_eqmarl_phi+.yml) | MDP experiment using $\texttt{eQMARL}$ with $\Phi^{+}$ entanglement.
[`coingame_maa2c_mdp_eqmarl_phi-.yml`](./experiments/coingame_maa2c_mdp_eqmarl_phi-.yml) | MDP experiment using $\texttt{eQMARL}$ with $\Phi^{-}$ entanglement.
[`coingame_maa2c_mdp_eqmarl_psi+.yml`](./experiments/coingame_maa2c_mdp_eqmarl_psi+.yml) | MDP experiment using $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement.
[`coingame_maa2c_mdp_eqmarl_psi-.yml`](./experiments/coingame_maa2c_mdp_eqmarl_psi-.yml) | MDP experiment using $\texttt{eQMARL}$ with $\Psi^{-}$ entanglement.
[`coingame_maa2c_mdp_fctde.yml`](./experiments/coingame_maa2c_mdp_fctde.yml) | MDP experiment using $\texttt{fCTDE}$.
[`coingame_maa2c_mdp_qfctde.yml`](./experiments/coingame_maa2c_mdp_qfctde.yml) | MDP experiment using $\texttt{qfCTDE}$.
[`coingame_maa2c_mdp_sctde.yml`](./experiments/coingame_maa2c_mdp_sctde.yml) | MDP experiment using $\texttt{sCTDE}$.
[`coingame_maa2c_pomdp_eqmarl_noentanglement.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_noentanglement.yml) | POMDP experiment using $\texttt{eQMARL}$ with $\texttt{None}$ entanglement.
[`coingame_maa2c_pomdp_eqmarl_phi+.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_phi+.yml) | POMDP experiment using $\texttt{eQMARL}$ with $\Phi^{+}$ entanglement.
[`coingame_maa2c_pomdp_eqmarl_phi-.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_phi-.yml) | POMDP experiment using $\texttt{eQMARL}$ with $\Phi^{-}$ entanglement.
[`coingame_maa2c_pomdp_eqmarl_psi+.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_psi+.yml) | POMDP experiment using $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement.
[`coingame_maa2c_pomdp_eqmarl_psi-.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_psi-.yml) | POMDP experiment using $\texttt{eQMARL}$ with $\Psi^{-}$ entanglement.
[`coingame_maa2c_pomdp_fctde.yml`](./experiments/coingame_maa2c_pomdp_fctde.yml) | POMDP experiment using $\texttt{fCTDE}$.
[`coingame_maa2c_pomdp_qfctde.yml`](./experiments/coingame_maa2c_pomdp_qfctde.yml) | POMDP experiment using $\texttt{qfCTDE}$.
[`coingame_maa2c_pomdp_sctde.yml`](./experiments/coingame_maa2c_pomdp_sctde.yml) | POMDP experiment using $\texttt{sCTDE}$.

## Results

The actor-critic models trained using the frameworks described in the paper achieved the performance outlined in the sections below.
Pre-trained models can be found in the supplementary materials, within a folder called `pre_trained_models/`, that accompanies this repository. 
<!-- Note that under the same folder structure as [`experiment_output`](./experiment_output/). -->

The training result metrics for all models reported in the paper are listed under the [`experiment_output`](./experiment_output/) folder.
Each experiment was conducted over 10 seeds (using the `-r 10` option as discussed in the [Training](#training) section).
All figures reported in the paper can be generated using the Jupyter notebook [`figure_generator.ipynb`](./scripts/figure_generator.ipynb), which references the figure configurations outlined in the [`figures`](./figures/) folder.

### Entanglement Style Comparison

The training results for the comparison of entanglement styles outlined in the paper are given in the table below:

Dynamics | Entanglement | Score: 20 | Score: 25 | Score: Max (_value_)
--- | --- | --- | --- | ---
MDP | $\Psi^{+}$ | **568** | 2332 | 2942 (_**25.67**_)
MDP | $\Psi^{-}$ | 595 | 1987 | 2849 (_25.45_)
MDP | $\Phi^{+}$ | 612 | **1883** | 2851 (_25.51_)
MDP | $\Phi^{-}$ | 691 | 2378 | 2984 (_25.23_)
MDP | $\mathtt{None}$ | 839 | 2337 | **2495** (_25.12_)
POMDP | $\Psi^{+}$ | **1049** | **1745** | 2950 (_26.28_)
POMDP | $\Psi^{-}$ | 1206 | 2114 | 2999 (_25.95_)
POMDP | $\Phi^{+}$ | 1269 | - | 2992 (_24.1_)
POMDP | $\Phi^{-}$ | 1838 | - | 2727 (_22.8_)
POMDP | $\mathtt{None}$ | 1069 | 1955 | **2841** (_**26.39**_)

The figures that aggregate the metric performance for each of the experiments are given in the table below:

Figure | Dynamics | Metric
--- | --- | ---
[fig_maa2c_mdp_entanglement_compare-undiscounted_reward.pdf](./figures/fig_maa2c_mdp_entanglement_compare/fig_maa2c_mdp_entanglement_compare-undiscounted_reward.pdf) | MDP | Score
[fig_maa2c_mdp_entanglement_compare-coins_collected.pdf](./figures/fig_maa2c_mdp_entanglement_compare/fig_maa2c_mdp_entanglement_compare-coins_collected.pdf) | MDP | Total coins collected
[fig_maa2c_mdp_entanglement_compare-own_coin_rate.pdf](./figures/fig_maa2c_mdp_entanglement_compare/fig_maa2c_mdp_entanglement_compare-own_coin_rate.pdf) | MDP | Own coin rate
[fig_maa2c_mdp_entanglement_compare-own_coins_collected.pdf](./figures/fig_maa2c_mdp_entanglement_compare/fig_maa2c_mdp_entanglement_compare-own_coins_collected.pdf) | MDP | Own coins collected
[fig_maa2c_pomdp_entanglement_compare-undiscounted_reward.pdf](./figures/fig_maa2c_pomdp_entanglement_compare/fig_maa2c_pomdp_entanglement_compare-undiscounted_reward.pdf) | POMDP | Score
[fig_maa2c_pomdp_entanglement_compare-coins_collected.pdf](./figures/fig_maa2c_pomdp_entanglement_compare/fig_maa2c_pomdp_entanglement_compare-coins_collected.pdf) | POMDP | Total coins collected
[fig_maa2c_pomdp_entanglement_compare-own_coin_rate.pdf](./figures/fig_maa2c_pomdp_entanglement_compare/fig_maa2c_pomdp_entanglement_compare-own_coin_rate.pdf) | POMDP | Own coin rate
[fig_maa2c_pomdp_entanglement_compare-own_coins_collected.pdf](./figures/fig_maa2c_pomdp_entanglement_compare/fig_maa2c_pomdp_entanglement_compare-own_coins_collected.pdf) | POMDP | Own coins collected




### Framework Comparison

The training results for the comparison of the frameworks outlined in the paper are given in the table below:

Dynamics | Framework | Score: 20 | Score: 25 | Score: Max (_value_) | Own coin rate: 0.95 | Own coin rate: 1.0 | Own coin rate: Max (_value_)
--- | --- | --- | --- | --- | --- | --- | ---
MDP | $\texttt{eQMARL-}\Psi^{+}$    | **568**  | **2332** | 2942 (_**25.67**_) | **376**  | **2136** | **2136** (_**1.0**_)
MDP | $\texttt{qfCTDE}$               | 678           | -             | **2378** (_23.38_) | 397           | -             | 2832 (_0.9972_)
MDP | $\texttt{sCTDE}$                | 1640          | 2615          | 2631 (_25.3_)           | 1511          | -             | 2637 (_0.9864_)
MDP | $\texttt{fCTDE}$                | 1917          | -             | 2925 (_23.67_)          | 1700          | -             | 2909 (_0.9857_)
POMDP | $\texttt{eQMARL-}\Psi^{+}$    | **1049** | **1745** | 2950 (_**26.28**_) | **773**  | -             | **2533** (_0.9997_)
POMDP | $\texttt{qfCTDE}$               | 1382          | 2124          | 2871 (_26.09_)          | 1038          | **2887** | 2887 (_**1.0**_)
POMDP | $\texttt{sCTDE}$                | 1738          | 2750          | 2999 (_25.33_)          | 1588          | -             | 2956 (_0.9894_)
POMDP | $\texttt{fCTDE}$                | 1798          | 2658          | **2824** (_25.49_) | 1574          | -             | 2963 (_0.9894_)

The figures that aggregate the metric performance for each of the experiments are given in the table below:

Figure | Dynamics | Metric
--- | --- | ---
[fig_maa2c_mdp-undiscounted_reward.pdf](./figures/fig_maa2c_mdp/fig_maa2c_mdp-undiscounted_reward.pdf) | MDP | Score
[fig_maa2c_mdp-coins_collected.pdf](./figures/fig_maa2c_mdp/fig_maa2c_mdp-coins_collected.pdf) | MDP | Total coins collected
[fig_maa2c_mdp-own_coin_rate.pdf](./figures/fig_maa2c_mdp/fig_maa2c_mdp-own_coin_rate.pdf) | MDP | Own coin rate
[fig_maa2c_mdp-own_coins_collected.pdf](./figures/fig_maa2c_mdp/fig_maa2c_mdp-own_coins_collected.pdf) | MDP | Own coins collected
[fig_maa2c_pomdp-undiscounted_reward.pdf](./figures/fig_maa2c_pomdp/fig_maa2c_pomdp-undiscounted_reward.pdf) | POMDP | Score
[fig_maa2c_pomdp-coins_collected.pdf](./figures/fig_maa2c_pomdp/fig_maa2c_pomdp-coins_collected.pdf) | POMDP | Total coins collected
[fig_maa2c_pomdp-own_coin_rate.pdf](./figures/fig_maa2c_pomdp/fig_maa2c_pomdp-own_coin_rate.pdf) | POMDP | Own coin rate
[fig_maa2c_pomdp-own_coins_collected.pdf](./figures/fig_maa2c_pomdp/fig_maa2c_pomdp-own_coins_collected.pdf) | POMDP | Own coins collected


## Authors

- [zanderman](https://github.com/zanderman) [![zanderman github](https://img.shields.io/badge/GitHub-zanderman-181717.svg?style=flat&logo=github)](https://github.com/zanderman)
- [saadwalid](https://github.com/saadwalid) [![saadwalid github](https://img.shields.io/badge/GitHub-saadwalid-181717.svg?style=flat&logo=github)](https://github.com/saadwalid)