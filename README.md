# TDGBA
code for "Trajectory-level Data Generation with Better Alignment for Offline Imitation Learning"

## run experiments
train_diffusion.py  --Train Trajectory-Diffuser

gen_traj.py    --Generate better trajectories

bc_data.py   --Constuct dataset for behavior cloning

train_actor.py     --Train and evaluate a BC policy

## References
Our code is modified based on:

1. https://github.com/Zzl35/flow-to-better
2. https://github.com/ott-jax/ott
3. https://github.com/ethanluoyc/optimal_transport_reward