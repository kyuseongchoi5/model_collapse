#!/bin/bash
# Simple wrapper to run training with diagnostics enabled
#
# Usage: ./run_with_diagnostics.sh configs/mixed_composite.yaml

CONFIG=$1

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

# Set environment variable to enable diagnostics
export ENABLE_DIAGNOSTICS=1

# Run training
cd /root/model_collapse/experiments/gp_complexity
python train.py --config "$CONFIG"
