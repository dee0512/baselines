#!/usr/bin/env bash
#
#SBATCH --job-name=ann
#SBATCH --partition=1080ti-long
#SBATCH --time=07-00:00:00
#SBATCH --mem=64000
#SBATCH --account=rkozma
#SBATCH --output=res_%j.txt
#SBATCH -e res_%j.err
#SBATCH --gres=gpu:1


python3 train_breakout.py
exit