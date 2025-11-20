"""
Script to add diagnostic logging to training pipeline.

This script will instrument:
1. Training data generation (check for NaN in GP samples)
2. Model predictions (check mean/variance for NaN/negative)
3. Loss computation (track what causes NaN)
4. Synthetic generation (check at epoch 100 switch)

Usage:
    python add_diagnostics.py
    Then run your training normally
"""

DIAGNOSTIC_CODE = """
# ============================================================================
# DIAGNOSTIC LOGGING - Add to train_with_switch.py
# ============================================================================

# At top of file, add:
import sys
sys.path.insert(0, '/root/model_collapse/experiments/gp_complexity')
from diagnostic_logger import DiagnosticLogger
diagnostic_logger = DiagnosticLogger()

# In train_epoch function, after getting data (around line 174):
# Add after: for batch, (data, targets) in enumerate(dataloader):

            # DIAGNOSTIC: Check training data
            if isinstance(data, tuple):
                x_data, y_data = data
            else:
                x_data = data
                y_data = targets

            diag_data = diagnostic_logger.check_training_data(
                x_data.to(device) if not x_data.is_cuda else x_data,
                y_data.to(device) if not y_data.is_cuda else y_data,
                epoch_num, batch
            )
            if diag_data['y_has_nan'] or diag_data['y_has_inf']:
                print(f"  [DIAG] Epoch {epoch_num}, Batch {batch}: Training data has NaN/Inf!")
                print(f"         y_has_nan={diag_data['y_has_nan']}, y_has_inf={diag_data['y_has_inf']}")
                print(f"         y_min={diag_data['y_min']}, y_max={diag_data['y_max']}")

# In train_epoch function, after model forward pass (around line 180):
# Add after: output = model(...)

            # DIAGNOSTIC: Check model predictions
            diag_pred = diagnostic_logger.check_model_predictions(output, epoch_num, batch)
            if diag_pred.get('mean_has_nan') or diag_pred.get('var_raw_has_negative'):
                print(f"  [DIAG] Epoch {epoch_num}, Batch {batch}: Model predictions have issues!")
                print(f"         mean_has_nan={diag_pred.get('mean_has_nan')}")
                print(f"         var_has_negative={diag_pred.get('var_raw_has_negative')}")
                print(f"         var_num_negative={diag_pred.get('var_raw_num_negative')}/{diag_pred.get('var_raw_num_total')}")

# In train_epoch function, after loss computation (around line 194):
# Add after: loss = losses.mean()

            # DIAGNOSTIC: Check loss computation
            if isinstance(criterion, nn.GaussianNLLLoss):
                diag_loss = diagnostic_logger.check_loss_computation(
                    loss, mean_pred, var_pred, targets.to(device), epoch_num, batch
                )
                if diag_loss['loss_is_nan']:
                    print(f"  [DIAG] Epoch {epoch_num}, Batch {batch}: Loss is NaN!")
                    print(f"         targets_has_nan={diag_loss['targets_has_nan']}")
                    print(f"         mean_pred_has_nan={diag_loss['mean_pred_has_nan']}")
                    print(f"         var_pred_min={diag_loss['var_pred_min']}")
                    print(f"         var_pred_max={diag_loss['var_pred_max']}")

# At end of train_with_switch function, before return:

    # DIAGNOSTIC: Save all logged diagnostics
    diagnostic_logger.save(f'{experiment_name}_diagnostics.json')

# ============================================================================
"""

print(DIAGNOSTIC_CODE)
print("\n" + "="*80)
print("INSTRUCTIONS:")
print("="*80)
print("1. Copy the diagnostic code sections above")
print("2. Add them to experiments/iterative_collapse/train_with_switch.py at the indicated locations")
print("3. Or use the patch below to auto-instrument")
print("="*80)
