#!/bin/bash

# Run all underfit learning experiments
# This script runs both synthetic and control experiments for MLP and Mix GP priors

set -e  # Exit on error

echo "=========================================="
echo "Underfit Learning Experiments"
echo "=========================================="
echo ""
echo "This will run 8 experiments total:"
echo "  4 synthetic (switch to synthetic data at epoch 100)"
echo "  4 control (only true data for all 300 epochs)"
echo ""
echo "Each experiment takes significant time. Estimated total: 10-20+ hours"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"

# Check if train script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

# Function to run an experiment
run_experiment() {
    local config_file=$1
    local experiment_name=$(basename "$config_file" .yaml)

    echo ""
    echo "=========================================="
    echo "Running: $experiment_name"
    echo "Config: $config_file"
    echo "=========================================="
    echo ""

    python3 "$TRAIN_SCRIPT" --config "$config_file"

    if [ $? -eq 0 ]; then
        echo "✓ $experiment_name completed successfully"
    else
        echo "✗ $experiment_name failed!"
        return 1
    fi
}

# Track timing
START_TIME=$(date +%s)

# Run synthetic experiments
echo "================================================"
echo "PHASE 1: Synthetic Switch Experiments (4/8)"
echo "================================================"

run_experiment "$SCRIPT_DIR/configs/hard_mlp.yaml"
run_experiment "$SCRIPT_DIR/configs/hard_ultra_diverse_mlp.yaml"
run_experiment "$SCRIPT_DIR/configs/hard_mixgp.yaml"
run_experiment "$SCRIPT_DIR/configs/hard_ultra_diverse_mixgp.yaml"

# Run control experiments
echo ""
echo "================================================"
echo "PHASE 2: Control Experiments (4/8)"
echo "================================================"

run_experiment "$SCRIPT_DIR/configs/control/control_mlp.yaml"
run_experiment "$SCRIPT_DIR/configs/control/control_ultra_diverse_mlp.yaml"
run_experiment "$SCRIPT_DIR/configs/control/control_mixgp.yaml"
run_experiment "$SCRIPT_DIR/configs/control/control_ultra_diverse_mixgp.yaml"

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to:"
echo "  Metrics: $SCRIPT_DIR/metrics/"
echo "  Checkpoints: $SCRIPT_DIR/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Analyze metrics to compare synthetic vs control"
echo "  2. Compare standard vs ultra diverse priors"
echo "  3. Look for evidence that synthetic data helps/hurts learning"
echo ""
