#!/bin/bash
# Run multiple model collapse experiments

# Experiment 1: Baseline (full config)
echo "Running baseline experiment..."
python train_with_switch.py --config config.yaml \
  --experiment_name baseline_switch100

# Experiment 2: Early switch
echo "Running early switch experiment..."
python train_with_switch.py --config config.yaml \
  --switch_epoch 50 \
  --total_epochs 200 \
  --experiment_name early_switch50

# Experiment 3: Late switch
echo "Running late switch experiment..."
python train_with_switch.py --config config.yaml \
  --switch_epoch 150 \
  --total_epochs 250 \
  --experiment_name late_switch150

# Experiment 4: Mean prediction (faster collapse)
echo "Running mean prediction experiment..."
python train_with_switch.py --config config.yaml \
  --synthetic_mode mean \
  --experiment_name mean_prediction

# Experiment 5: Mixed data (50/50)
echo "Running mixed data experiment..."
python train_with_switch.py --config config.yaml \
  --synthetic_ratio 0.5 \
  --experiment_name mixed_50_50

# Experiment 6: Large model
echo "Running large model experiment..."
python train_with_switch.py --config config.yaml \
  --emsize 1024 \
  --nlayers 12 \
  --nhid 2048 \
  --nhead 8 \
  --experiment_name large_model

echo "All experiments complete!"
