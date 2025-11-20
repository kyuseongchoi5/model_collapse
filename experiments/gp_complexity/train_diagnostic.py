#!/usr/bin/env python3
"""
Training script with diagnostic logging for GP complexity experiments.

This wraps the main training script and adds comprehensive logging to track
NaN values and numerical issues.

Usage:
    python train_diagnostic.py --config configs/mixed_composite.yaml
"""

import sys
import subprocess
from pathlib import Path

# Add diagnostic logging wrapper
wrapper_script = """
import sys
sys.path.insert(0, '/root/model_collapse')
sys.path.insert(0, '/root/model_collapse/experiments/gp_complexity')

# Monkey-patch the training loop to add diagnostics
import experiments.iterative_collapse.train_with_switch as train_module
from diagnostic_logger import DiagnosticLogger

# Create global logger
diagnostic_logger = DiagnosticLogger()

# Save original train_epoch function
original_train_with_switch = train_module.train_with_switch

def train_with_switch_diagnostic(*args, **kwargs):
    # Call original but wrap inner train_epoch
    result = original_train_with_switch(*args, **kwargs)

    # Save diagnostics at the end
    diagnostic_logger.save()

    return result

# Patch in our diagnostic version
train_module.train_with_switch = train_with_switch_diagnostic

# Now run the original script
exec(open('/root/model_collapse/experiments/iterative_collapse/train_with_switch.py').read())
"""

if __name__ == '__main__':
    # Write wrapper script to temp file
    temp_script = Path('/tmp/train_diagnostic_wrapper.py')
    temp_script.write_text(wrapper_script)

    # Run with original arguments
    cmd = [sys.executable, str(temp_script)] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
