#!/bin/bash

set -x

##### Ablation study.

sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_eqmarl_psi+_L2.yml -r 10
sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_eqmarl_psi+_L10.yml -r 10
sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_eqmarl_psi+_L2.yml -r 10
sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_eqmarl_psi+_L10.yml -r 10

sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_qfctde_L2.yml -r 10
sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_qfctde_L10.yml -r 10
sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_qfctde_L2.yml -r 10
sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_qfctde_L10.yml -r 10

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_fctde_size3.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_fctde_size6.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_fctde_size24.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_sctde_size3.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_sctde_size6.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_mdp_sctde_size24.yml -r 5

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_fctde_size3.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_fctde_size6.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_fctde_size24.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_sctde_size3.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_sctde_size6.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame_maa2c_pomdp_sctde_size24.yml -r 5


##### CartPole

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_fctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_sctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_qfctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_eqmarl_psi+.yml -r 5

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_eqmarl_psi-.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_eqmarl_phi+.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_eqmarl_phi-.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_mdp_eqmarl_noentanglement.yml -r 5

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_fctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_sctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_qfctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_eqmarl_psi+.yml -r 5

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_eqmarl_psi-.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_eqmarl_phi+.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_eqmarl_phi-.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/cartpole_maa2c_pomdp_eqmarl_noentanglement.yml -r 5

# ##### CoinGame4

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_mdp_fctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_mdp_sctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_mdp_qfctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_mdp_eqmarl_psi+.yml -r 5

# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_pomdp_fctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_pomdp_sctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_pomdp_qfctde.yml -r 5
# sbatch ./scripts/slurm_run_python.sh ./scripts/experiment_runner.py ./experiments/coingame4_maa2c_pomdp_eqmarl_psi+.yml -r 5

set +x
