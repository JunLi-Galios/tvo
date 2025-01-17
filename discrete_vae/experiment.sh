#!/bin/sh

# Note: Each run takes 15 - 24 hrs w/ a gpu.
# These jobs shoud be submitted to a cluster.

# The -F ./runs flag uses Sacred's FileStorageObserver, described here
# https://sacred.readthedocs.io/en/stable/observers.html#file-storage-observer


# ======================
#  Figure 6
# ======================
# Thermo w/ grid searched log_beta_min
# python run.py with loss=thermo S=5  K=5 log_beta_min=-2.0 epochs=$EPOCHS --name bl_comp -p -F ./runs
# python run.py with loss=thermo S=10 K=5 log_beta_min=-1.6989700043360187 epochs=$EPOCHS --name bl_comp -p -F ./runs
# python run.py with loss=thermo S=50 K=5 log_beta_min=-1.5228787452803376 epochs=$EPOCHS --name bl_comp -p -F ./runs


# Run baselines
# for S in 5 10 50; do
#    for loss in vae iwae; do
#        python run.py with loss=$loss S=$S epochs=$EPOCHS --name bl_comp -p -F ./runs
#    done;
# done;


# ======================
#  Figure 7
# ======================

for S in 5 10 50; do
    for K in 1 2 5 10; do
        for LR in 1e-2 1e-3 1e-4 1e-5 1e-6; do
            for ALPHA in 0.99 0.9 0.5 0.1 1.01 1.1 2; do
                for INTEGRATION in left right trapz; do
                    # lower lr for reinforce for stability
                    # python run.py with loss=reinforce S=$S lr=0.00001 epochs=$EPOCHS --name cov_est_var -p -F ./runs
                    # Set K = 1 for thermo to make equivalent to elbo
                    python run.py --train-mode thermo_alpha --architecture non_linear --cuda --integration $INTEGRATION --learning-rate $LR --num-particles $S --num-partitions $K --alpha=$ALPHA 
                    # Run vae as is
                    # python run.py with loss=vae S=$S epochs=$EPOCHS --name cov_est_var -p -F ./runs
                done;
            done;
        done;
    done;
done
