#!/usr/bin/env python3
"""
Training script wrapper for underfit learning experiments.

This is a convenience wrapper that calls the main training script
while staying in the underfit_learning directory.

Usage:
    python train.py --config configs/hard_mlp.yaml
    python train.py --config configs/control/control_mlp.yaml
"""

import sys
import subprocess
from pathlib import Path

if __name__ == '__main__':
    # Get the actual training script path
    script_dir = Path(__file__).parent
    train_script = script_dir.parent / 'iterative_collapse' / 'train_with_switch.py'

    # Pass all arguments to the actual training script
    cmd = [sys.executable, str(train_script)] + sys.argv[1:]

    # Execute the training script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
