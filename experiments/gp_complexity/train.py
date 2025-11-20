#!/usr/bin/env python3
"""
Training script wrapper for GP complexity experiments.

This is a convenience wrapper that calls the main training script
while staying in the gp_complexity directory.

Usage:
    python train.py --config configs/baseline_single.yaml
    python train.py --config configs/mixed_composite.yaml
    python train.py --config configs/full_composite.yaml
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
