"""Apply diagnostic logging to train_with_switch.py"""
import re

# Read original file
with open('/root/model_collapse/experiments/iterative_collapse/train_with_switch.py', 'r') as f:
    content = f.read()

# Add imports at top
imports_addition = """import json
import os

# Diagnostic logging setup
ENABLE_DIAGNOSTICS = os.environ.get('ENABLE_DIAGNOSTICS', '0') == '1'
diagnostic_data = {
    'training_data': [],
    'model_predictions': [],
    'losses': [],
} if ENABLE_DIAGNOSTICS else None

"""

# Find where to insert (after existing imports)
content = content.replace(
    "from evaluate import evaluate_on_test_set",
    "from evaluate import evaluate_on_test_set\n" + imports_addition
)

# Save modified version
with open('/root/model_collapse/experiments/gp_complexity/train_with_switch_diagnostic.py', 'w') as f:
    f.write(content)

print("Created train_with_switch_diagnostic.py")
print("Now manually adding diagnostic logging points...")
