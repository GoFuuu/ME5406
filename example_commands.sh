#!/bin/bash
# Example commands for running the FetchReach project

# Training a new model
python train.py

# Testing a trained model
python test.py --log_dir logs/Tianshou_SAC_12_Apr_2025_22_55_02/ --model_file logs/Tianshou_SAC_12_Apr_2025_22_55_02/Tianshou_SAC_epoch1.pth

# Generating visualizations
python visualize.py --log_dir logs/Tianshou_SAC_12_Apr_2025_22_55_02/ --model_file logs/Tianshou_SAC_12_Apr_2025_22_55_02/Tianshou_SAC_epoch1.pth

# Running the complete pipeline
python run_all.py --train --test --visualize
