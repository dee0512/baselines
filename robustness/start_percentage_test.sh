#!/usr/bin/env bash

for seed in {0..99..1}
do
    for n_episodes in 1
    do
        for occlusion in {0..100..5}
        do
            sbatch submit_percentage.sh $seed $n_episodes $occlusion
        done
    done
done
