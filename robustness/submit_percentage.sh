#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.out
#SBATCH --cpus-per-task=8

seed=${1:-0}
n_episodes=${2:-100}
occlusion=${3:-0}

echo $seed $n_episodes $occlusion
python3 occlusion_percentage.py --seed $seed --n_episodes $n_episodes --occlusion $occlusion
exit