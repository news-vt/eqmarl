# eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels

[![arXiv](https://img.shields.io/badge/quant--ph-arXiv:2405.17486-b31b1b.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2405.17486)
[![OpenReview](https://img.shields.io/badge/OpenReview.net-cR5GTis5II-8D1018.svg)](https://openreview.net/forum?id=cR5GTis5II)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/Python->=3.9,<3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.7.0-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org)

This repository is the official implementation of "eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels", published in the Thirteenth International Conference on Learning Representations (ICLR) 2025.

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

Experiment YAML File | Environment | Description
--- | --- | --- 
[`coingame_maa2c_mdp_eqmarl_noentanglement.yml`](./experiments/coingame_maa2c_mdp_eqmarl_noentanglement.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{eQMARL}$ with $\texttt{None}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_mdp_eqmarl_phi+.yml`](./experiments/coingame_maa2c_mdp_eqmarl_phi+.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{eQMARL}$ with $\Phi^{+}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_mdp_eqmarl_phi-.yml`](./experiments/coingame_maa2c_mdp_eqmarl_phi-.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{eQMARL}$ with $\Phi^{-}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_mdp_eqmarl_psi+.yml`](./experiments/coingame_maa2c_mdp_eqmarl_psi+.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_mdp_eqmarl_psi-.yml`](./experiments/coingame_maa2c_mdp_eqmarl_psi-.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{eQMARL}$ with $\Psi^{-}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_mdp_fctde.yml`](./experiments/coingame_maa2c_mdp_fctde.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{fCTDE}$ with $h=12$ hidden units.
[`coingame_maa2c_mdp_qfctde.yml`](./experiments/coingame_maa2c_mdp_qfctde.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{qfCTDE}$ with $L=5$ VQC layers.
[`coingame_maa2c_mdp_sctde.yml`](./experiments/coingame_maa2c_mdp_sctde.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{sCTDE}$ with $h=12$ hidden units.
[`coingame_maa2c_pomdp_eqmarl_noentanglement.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_noentanglement.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{eQMARL}$ with $\texttt{None}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_pomdp_eqmarl_phi+.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_phi+.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{eQMARL}$ with $\Phi^{+}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_pomdp_eqmarl_phi-.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_phi-.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{eQMARL}$ with $\Phi^{-}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_pomdp_eqmarl_psi+.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_psi+.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_pomdp_eqmarl_psi-.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_psi-.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{eQMARL}$ with $\Psi^{-}$ entanglement and $L=5$ VQC layers.
[`coingame_maa2c_pomdp_fctde.yml`](./experiments/coingame_maa2c_pomdp_fctde.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{fCTDE}$ with $h=12$ hidden units.
[`coingame_maa2c_pomdp_qfctde.yml`](./experiments/coingame_maa2c_pomdp_qfctde.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{qfCTDE}$ with $L=5$ VQC layers.
[`coingame_maa2c_pomdp_sctde.yml`](./experiments/coingame_maa2c_pomdp_sctde.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{sCTDE}$ with $h=12$ hidden units.
[`coingame_maa2c_mdp_eqmarl_psi+_L2.yml`](./experiments/coingame_maa2c_mdp_eqmarl_psi+_L2.yml) | $\texttt{CoinGame-2}$ | MDP experiment $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=2$ VQC layers.
[`coingame_maa2c_mdp_eqmarl_psi+_L10.yml`](./experiments/coingame_maa2c_mdp_eqmarl_psi+_L10.yml) | $\texttt{CoinGame-2}$ | MDP experiment $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=10$ VQC layers.
[`coingame_maa2c_mdp_qfctde_L2.yml`](./experiments/coingame_maa2c_mdp_qfctde_L2.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{qfCTDE}$ with $L=2$ VQC layers.
[`coingame_maa2c_mdp_qfctde_L10.yml`](./experiments/coingame_maa2c_mdp_qfctde_L10.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{qfCTDE}$ with $L=10$ VQC layers.
[`coingame_maa2c_mdp_fctde_size3.yml`](./experiments/coingame_maa2c_mdp_fctde_size3.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{fCTDE}$ with $h=3$ hidden units.
[`coingame_maa2c_mdp_fctde_size6.yml`](./experiments/coingame_maa2c_mdp_fctde_size6.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{fCTDE}$ with $h=6$ hidden units.
[`coingame_maa2c_mdp_fctde_size24.yml`](./experiments/coingame_maa2c_mdp_fctde_size24.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{fCTDE}$ with $h=24$ hidden units.
[`coingame_maa2c_mdp_sctde_size3.yml`](./experiments/coingame_maa2c_mdp_sctde_size3.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{sCTDE}$ with $h=3$ hidden units.
[`coingame_maa2c_mdp_sctde_size6.yml`](./experiments/coingame_maa2c_mdp_sctde_size6.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{sCTDE}$ with $h=6$ hidden units.
[`coingame_maa2c_mdp_sctde_size24.yml`](./experiments/coingame_maa2c_mdp_sctde_size24.yml) | $\texttt{CoinGame-2}$ | MDP experiment using $\texttt{sCTDE}$ with $h=24$ hidden units.
[`coingame_maa2c_pomdp_eqmarl_psi+_L2.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_psi+_L2.yml) | $\texttt{CoinGame-2}$ | POMDP experiment $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=2$ VQC layers.
[`coingame_maa2c_pomdp_eqmarl_psi+_L10.yml`](./experiments/coingame_maa2c_pomdp_eqmarl_psi+_L10.yml) | $\texttt{CoinGame-2}$ | POMDP experiment $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=10$ VQC layers.
[`coingame_maa2c_pomdp_qfctde_L2.yml`](./experiments/coingame_maa2c_pomdp_qfctde_L2.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{qfCTDE}$ with $L=2$ VQC layers.
[`coingame_maa2c_pomdp_qfctde_L10.yml`](./experiments/coingame_maa2c_pomdp_qfctde_L10.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{qfCTDE}$ with $L=10$ VQC layers.
[`coingame_maa2c_pomdp_fctde_size3.yml`](./experiments/coingame_maa2c_pomdp_fctde_size3.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{fCTDE}$ with $h=3$ hidden units.
[`coingame_maa2c_pomdp_fctde_size6.yml`](./experiments/coingame_maa2c_pomdp_fctde_size6.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{fCTDE}$ with $h=6$ hidden units.
[`coingame_maa2c_pomdp_fctde_size24.yml`](./experiments/coingame_maa2c_pomdp_fctde_size24.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{fCTDE}$ with $h=24$ hidden units.
[`coingame_maa2c_pomdp_sctde_size3.yml`](./experiments/coingame_maa2c_pomdp_sctde_size3.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{sCTDE}$ with $h=3$ hidden units.
[`coingame_maa2c_pomdp_sctde_size6.yml`](./experiments/coingame_maa2c_pomdp_sctde_size6.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{sCTDE}$ with $h=6$ hidden units.
[`coingame_maa2c_pomdp_sctde_size24.yml`](./experiments/coingame_maa2c_pomdp_sctde_size24.yml) | $\texttt{CoinGame-2}$ | POMDP experiment using $\texttt{sCTDE}$ with $h=24$ hidden units.
[`cartpole_maa2c_mdp_eqmarl_psi+.yml`](./experiments/cartpole_maa2c_mdp_eqmarl_psi+.yml) | $\texttt{CartPole}$ | MDP experiment using $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=5$ VQC layers.
[`cartpole_maa2c_mdp_fctde.yml`](./experiments/cartpole_maa2c_mdp_fctde.yml) | $\texttt{CartPole}$ | MDP experiment using $\texttt{fCTDE}$ with $h=12$ hidden units.
[`cartpole_maa2c_mdp_qfctde.yml`](./experiments/cartpole_maa2c_mdp_qfctde.yml) | $\texttt{CartPole}$ | MDP experiment using $\texttt{qfCTDE}$ with $L=5$ VQC layers.
[`cartpole_maa2c_mdp_sctde.yml`](./experiments/cartpole_maa2c_mdp_sctde.yml) | $\texttt{CartPole}$ | MDP experiment using $\texttt{sCTDE}$ with $h=12$ hidden units.
[`cartpole_maa2c_pomdp_eqmarl_psi+.yml`](./experiments/cartpole_maa2c_pomdp_eqmarl_psi+.yml) | $\texttt{CartPole}$ | POMDP experiment using $\texttt{eQMARL}$ with $\Psi^{+}$ entanglement and $L=5$ VQC layers.
[`cartpole_maa2c_pomdp_fctde.yml`](./experiments/cartpole_maa2c_pomdp_fctde.yml) | $\texttt{CartPole}$ | POMDP experiment using $\texttt{fCTDE}$ with $h=12$ hidden units.
[`cartpole_maa2c_pomdp_qfctde.yml`](./experiments/cartpole_maa2c_pomdp_qfctde.yml) | $\texttt{CartPole}$ | POMDP experiment using $\texttt{qfCTDE}$ with $L=5$ VQC layers.
[`cartpole_maa2c_pomdp_sctde.yml`](./experiments/cartpole_maa2c_pomdp_sctde.yml) | $\texttt{CartPole}$ | POMDP experiment using $\texttt{sCTDE}$ with $h=12$ hidden units.

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

### CoinGame experiments

The training results for the comparison of the frameworks in the $\texttt{CoinGame-2}$ environment outlined in the paper are given in the table below:

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

### CartPole experiments

The training results for the comparison of the frameworks in the $\texttt{CartPole}$ environment outlined in the paper are given in the tables below:

Dynamics | Framework | Reward: Mean | Reward: Std. Dev. | Reward: $95\%$ CI
--- | --- | --- | --- | ---
MDP | $\texttt{eQMARL-}\Psi^{+}$ | 79.11 | 50.62 | (77.40, 81.16)
MDP | $\texttt{qfCTDE}$ | 121.35 | 110.13 | (118.29, 125.12)
MDP | $\texttt{sCTDE}$ | 16.38 | 35.97 | (16.29, 16.48)
MDP | $\texttt{fCTDE}$ | 15.15 | 24.17 | (15.09, 15.22)
POMDP | $\texttt{eQMARL-}\Psi^{+}$ | 82.28 | 44.24 | (80.60, 83.89)
POMDP | $\texttt{qfCTDE}$ | 79.03 | 44.06 | (76.80, 80.98)
POMDP | $\texttt{sCTDE}$ | 40.56 | 37.36 | (38.17, 43.70)
POMDP | $\texttt{fCTDE}$ | 13.93 | 29.84 | (13.62, 14.19)

Dynamics | Framework | Reward: Mean (value) | Reward: Max (value)
--- | --- | --- | ---
MDP | $\texttt{eQMARL-}\Psi^{+}$ | 166 (_79.11_) | 555 (_134.16_)
MDP | $\texttt{qfCTDE}$ | 189 (_121.35_) | 810 (_262.43_)
MDP | $\texttt{sCTDE}$ | 9 (_16.38_) | 931 (_23.59_)
MDP | $\texttt{fCTDE}$ | 9 (_15.15_) | 38 (_18.55_)
POMDP | $\texttt{eQMARL-}\Psi^{+}$ | 251 (_82.28_) | 770 (_127.6_)
POMDP | $\texttt{qfCTDE}$ | 276 (_79.03_) | 648 (_137.66_)
POMDP | $\texttt{sCTDE}$ | 680 (_40.56_) | 999 (_167.32_)
POMDP | $\texttt{fCTDE}$ | 9 (_13.93_) | 999 (_28.66_)

The figures that aggregate the metric performance for each of the experiments are given in the table below:

Figure | Dynamics | Metric
--- | --- | ---
[fig_cartpole_maa2c_mdp-reward_mean.pdf](./figures/fig_cartpole_maa2c_mdp/fig_cartpole_maa2c_mdp-reward_mean.pdf) | MDP | Average reward
[fig_cartpole_maa2c_pomdp-reward_mean.pdf](./figures/fig_cartpole_maa2c_pomdp/fig_cartpole_maa2c_pomdp-reward_mean.pdf) | POMDP | Average reward

### MiniGrid experiments

The training results for the comparison of the frameworks in the $\texttt{MiniGrid}$ environment outlined in the paper are given in the tables below:

Dynamics | Framework | Reward: Mean (value) | Reward: $95\%$ CI | Number of Trainable Critic Parameters
--- | --- | --- | --- | ---
POMDP | $\texttt{fCTDE}$ | -63.04 | (-65.16, -61.06) | 29,601
POMDP | $\texttt{qfCTDE}$ | -85.86 | (-87.03, -84.72) | 3,697
POMDP | $\texttt{sCTDE}$ | -88.02 |  (-88.69, -87.10) | 29,801
POMDP | $\texttt{eQMARL}-\Psi^+$ | -13.32 | (-14.68, -11.91) | 3,697

The figures that aggregate the metric performance for each of the experiments are given in the table below:

Figure | Dynamics | Metric
--- | --- | ---
[fig_minigrid-reward_mean.pdf](./figures/fig_minigrid/fig_minigrid-reward_mean.pdf) | POMDP | Average reward

### Ablation experiments

The training results for the ablation experiment using in the $\texttt{CoinGame-2}$ environment outlined in the paper are given in the tables below:

Dynamics | Framework | Parameters | Score: Mean | Score: Std. Dev. | Score: $95\%$ CI | Own coin rate: Mean | Own coin rate: Std. Dev. | Own coin rate: $95\%$ CI
--- | --- | --- | --- | --- | --- | --- | --- | ---
MDP | $\texttt{fCTDE-3}$ | 223 | 2.42 | 2.35 | (2.35, 2.49) | 0.6720 | 0.2024 | (0.6685, 0.6769)
MDP | $\texttt{fCTDE-6}$ | 445 | 7.41 | 3.46 | (7.19, 7.65) | 0.7658 | 0.1414 | (0.7610, 0.7712)
MDP | $\texttt{fCTDE-12}$ | 889 | 12.36 | 4.41 | (12.09, 12.67) | 0.8202 | 0.1379 | (0.8139, 0.8262)
MDP | $\texttt{fCTDE-24}$ | 1777 | 17.63 | 2.58 | (17.25, 17.91) | 0.8823 | 0.0751 | (0.8770, 0.8875)
MDP | $\texttt{sCTDE-3}$ | 229 | 3.24 | 3.09 | (3.16, 3.33) | 0.6852 | 0.1991 | (0.6821, 0.6897)
MDP | $\texttt{sCTDE-6}$ | 457 | 8.54 | 3.67 | (8.29, 8.78) | 0.7857 | 0.1327 | (0.7804, 0.7924)
MDP | $\texttt{sCTDE-12}$ | 913 | 14.18 | 2.69 | (13.90, 14.60) | 0.8504 | 0.0928 | (0.8454, 0.8553)
MDP | $\texttt{sCTDE-24}$ | 1825 | 18.18 | 2.41 | (17.84, 18.53) | 0.8936 | 0.0673 | (0.8896, 0.8979)
MDP | $\texttt{qfCTDE-L2}$ | 121 | 6.58 | 3.92 | (6.47, 6.66) | 0.8482 | 0.1921 | (0.8435, 0.8518)
MDP | $\texttt{qfCTDE-L5}$ | 265 | 19.41 | 6.23 | (19.23, 19.59) | 0.9398 | 0.1020 | (0.9366, 0.9426)
MDP | $\texttt{qfCTDE-L10}$ | 505 | 22.08 | 2.22 | (21.91, 22.26) | 0.9691 | 0.0247 | (0.9665, 0.9723)
MDP | $\texttt{eQMARL-}\Psi^{+}\texttt{-L2}$ | 121 | 5.38 | 3.74 | (5.30, 5.46) | 0.8271 | 0.2213 | (0.8234, 0.8300)
MDP | $\texttt{eQMARL-}\Psi^{+}\texttt{-L5}$ | 265 | 21.11 | 2.65 | (20.92, 21.35) | 0.9640 | 0.0347 | (0.9601, 0.9667)
MDP | $\texttt{eQMARL-}\Psi^{+}\texttt{-L10}$ | 505 | 22.45 | 2.23 | (22.28, 22.62) | 0.9719 | 0.0219 | (0.9685, 0.9745)
POMDP | $\texttt{fCTDE-3}$ | 169 | 2.98 | 2.47 | (2.91, 3.05) | 0.7082 | 0.1890 | (0.7039, 0.7123)
POMDP | $\texttt{fCTDE-6}$ | 337 | 7.15 | 3.06 | (6.95, 7.37) | 0.7711 | 0.1388 | (0.7658, 0.7781)
POMDP | $\texttt{fCTDE-12}$ | 673 | 13.46 | 3.24 | (13.09, 13.76) | 0.8443 | 0.1026 | (0.8396, 0.8506)
POMDP | $\texttt{fCTDE-24}$ | 1345 | 17.38 | 2.65 | (17.06, 17.73) | 0.8889 | 0.0752 | (0.8840, 0.8945)
POMDP | $\texttt{sCTDE-3}$ | 175 | 2.68 | 2.60 | (2.61, 2.74) | 0.6834 | 0.1942 | (0.6792, 0.6866)
POMDP | $\texttt{sCTDE-6}$ | 349 | 6.35 | 3.53 | (6.18, 6.54) | 0.7677 | 0.1488 | (0.7633, 0.7725)
POMDP | $\texttt{sCTDE-12}$ | 697 | 13.70 | 2.79 | (13.44, 13.99) | 0.8466 | 0.0985 | (0.8411, 0.8515)
POMDP | $\texttt{sCTDE-24}$ | 1393 | 17.97 | 2.60 | (17.67, 18.25) | 0.8948 | 0.0723 | (0.8898, 0.9004)
POMDP | $\texttt{qfCTDE-L2}$ | 745 | 12.34 | 7.56 | (12.09, 12.60) | 0.8335 | 0.2058 | (0.8277, 0.8386)
POMDP | $\texttt{qfCTDE-L5}$ | 817 | 16.79 | 4.66 | (16.45, 17.04) | 0.9040 | 0.1135 | (0.8994, 0.9091)
POMDP | $\texttt{qfCTDE-L10}$ | 937 | 18.14 | 4.28 | (17.83, 18.31) | 0.9476 | 0.0660 | (0.9443, 0.9508)
POMDP | $\texttt{eQMARL-}\Psi^{+}\texttt{-L2}$ | 745 | 17.14 | 3.98 | (16.77, 17.47) | 0.8834 | 0.1106 | (0.8769, 0.8896)
POMDP | $\texttt{eQMARL-}\Psi^{+}\texttt{-L5}$ | 817 | 18.49 | 3.91 | (18.23, 18.80) | 0.9226 | 0.0831 | (0.9172, 0.9272)
POMDP | $\texttt{eQMARL-}\Psi^{+}\texttt{-L10}$ | 937 | 19.09 | 3.44 | (18.86, 19.46) | 0.9485 | 0.0603 | (0.9458, 0.9523)

Framework | Ablation Selection | Model | MDP dynamics | POMDP dynamics
--- | --- | --- | --- | ---
$\texttt{eQMARL}$ | $L=5$ | Actor | 136 | 412
$\texttt{eQMARL}$ | $L=5$ | Critic | 265 (132 per agent, 1 central) | 817 (408 per agent, 1 central)
$\texttt{qfCTDE}$ | $L=5$ | Actor | 136 | 412
$\texttt{qfCTDE}$ | $L=5$ | Critic | 265 | 817
$\texttt{fCTDE}$ | $h=12$ | Actor | 496 | 388
$\texttt{fCTDE}$ | $h=12$ | Critic | 889 | 673
$\texttt{sCTDE}$ | $h=12$ | Actor | 496 | 388
$\texttt{sCTDE}$ | $h=12$ | Critic | 913 (444 per agent, 25 central) | 697 (336 per agent, 25 central)

The figures that aggregate the metric performance for each of the experiments are given in the table below:

Figure | Dynamics | Metric
--- | --- | ---
[fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-undiscounted_reward.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-undiscounted_reward.pdf) | MDP | Score
[fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-coins_collected.pdf) | MDP | Total coins collected
[fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-own_coin_rate.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-own_coin_rate.pdf) | MDP | Own coin rate
[fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-own_coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+/fig_coingame2_maa2c_mdp_ablation_eqmarl_psi+-own_coins_collected.pdf) | MDP | Own coins collected
[fig_coingame2_maa2c_mdp_ablation_qfctde-undiscounted_reward.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_qfctde/fig_coingame2_maa2c_mdp_ablation_qfctde-undiscounted_reward.pdf) | MDP | Score
[fig_coingame2_maa2c_mdp_ablation_qfctde-coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_qfctde/fig_coingame2_maa2c_mdp_ablation_qfctde-coins_collected.pdf) | MDP | Total coins collected
[fig_coingame2_maa2c_mdp_ablation_qfctde-own_coin_rate.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_qfctde/fig_coingame2_maa2c_mdp_ablation_qfctde-own_coin_rate.pdf) | MDP | Own coin rate
[fig_coingame2_maa2c_mdp_ablation_qfctde-own_coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_qfctde/fig_coingame2_maa2c_mdp_ablation_qfctde-own_coins_collected.pdf) | MDP | Own coins collected
[fig_coingame2_maa2c_mdp_ablation_fctde-undiscounted_reward.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_fctde/fig_coingame2_maa2c_mdp_ablation_fctde-undiscounted_reward.pdf) | MDP | Score
[fig_coingame2_maa2c_mdp_ablation_fctde-coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_fctde/fig_coingame2_maa2c_mdp_ablation_fctde-coins_collected.pdf) | MDP | Total coins collected
[fig_coingame2_maa2c_mdp_ablation_fctde-own_coin_rate.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_fctde/fig_coingame2_maa2c_mdp_ablation_fctde-own_coin_rate.pdf) | MDP | Own coin rate
[fig_coingame2_maa2c_mdp_ablation_fctde-own_coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_fctde/fig_coingame2_maa2c_mdp_ablation_fctde-own_coins_collected.pdf) | MDP | Own coins collected
[fig_coingame2_maa2c_mdp_ablation_sctde-undiscounted_reward.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_sctde/fig_coingame2_maa2c_mdp_ablation_sctde-undiscounted_reward.pdf) | MDP | Score
[fig_coingame2_maa2c_mdp_ablation_sctde-coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_sctde/fig_coingame2_maa2c_mdp_ablation_sctde-coins_collected.pdf) | MDP | Total coins collected
[fig_coingame2_maa2c_mdp_ablation_sctde-own_coin_rate.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_sctde/fig_coingame2_maa2c_mdp_ablation_sctde-own_coin_rate.pdf) | MDP | Own coin rate
[fig_coingame2_maa2c_mdp_ablation_sctde-own_coins_collected.pdf](./figures/fig_coingame2_maa2c_mdp_ablation_sctde/fig_coingame2_maa2c_mdp_ablation_sctde-own_coins_collected.pdf) | MDP | Own coins collected

## Authors

- [zanderman](https://github.com/zanderman) [![zanderman github](https://img.shields.io/badge/GitHub-zanderman-181717.svg?style=flat&logo=github)](https://github.com/zanderman)
- [saadwalid](https://github.com/saadwalid) [![saadwalid github](https://img.shields.io/badge/GitHub-saadwalid-181717.svg?style=flat&logo=github)](https://github.com/saadwalid)

## Citation

If you use the code in this repository for your research or publication, please cite our paper published in ICLR 2025:
```bibtex
@inproceedings{derieux2025eqmarl,
    title={e{QMARL}: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels},
    author={Alexander DeRieux and Walid Saad},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=cR5GTis5II},
    doi={10.48550/arXiv.2405.17486}
}
```