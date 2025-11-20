"""
Diagnostic Logger for Tracking NaN and Numerical Issues

This module provides logging utilities to track where NaN and numerical
issues arise during training and synthetic data generation.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


class DiagnosticLogger:
    """Logger to track numerical issues in training."""

    def __init__(self, log_dir='./experiments/gp_complexity/diagnostics'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'training_data_issues': [],
            'model_prediction_issues': [],
            'loss_computation_issues': [],
            'synthetic_generation_issues': [],
            'summary': {}
        }

        self.epoch_stats = []

    def check_training_data(self, x, y, epoch, batch):
        """Check if training data contains NaN or Inf."""
        issues = {
            'epoch': epoch,
            'batch': batch,
            'x_has_nan': torch.isnan(x).any().item(),
            'x_has_inf': torch.isinf(x).any().item(),
            'y_has_nan': torch.isnan(y).any().item(),
            'y_has_inf': torch.isinf(y).any().item(),
            'x_min': x.min().item() if not torch.isnan(x).any() else None,
            'x_max': x.max().item() if not torch.isnan(x).any() else None,
            'y_min': y.min().item() if not torch.isnan(y).any() else None,
            'y_max': y.max().item() if not torch.isnan(y).any() else None,
            'y_mean': y.mean().item() if not torch.isnan(y).any() else None,
            'y_std': y.std().item() if not torch.isnan(y).any() and y.numel() > 1 else None,
        }

        if issues['x_has_nan'] or issues['x_has_inf'] or issues['y_has_nan'] or issues['y_has_inf']:
            self.diagnostics['training_data_issues'].append(issues)

        return issues

    def check_model_predictions(self, output, epoch, batch):
        """Check model predictions for numerical issues."""
        issues = {
            'epoch': epoch,
            'batch': batch,
        }

        # Check if GaussianNLL output (has mean and variance)
        if output.shape[-1] == 2:
            mean_pred = output[..., 0]
            var_pred_raw = output[..., 1]  # Before any processing
            var_pred_abs = var_pred_raw.abs()

            issues.update({
                'mean_has_nan': torch.isnan(mean_pred).any().item(),
                'mean_has_inf': torch.isinf(mean_pred).any().item(),
                'mean_min': mean_pred.min().item() if not torch.isnan(mean_pred).any() else None,
                'mean_max': mean_pred.max().item() if not torch.isnan(mean_pred).any() else None,
                'mean_mean': mean_pred.mean().item() if not torch.isnan(mean_pred).any() else None,

                'var_raw_has_nan': torch.isnan(var_pred_raw).any().item(),
                'var_raw_has_inf': torch.isinf(var_pred_raw).any().item(),
                'var_raw_has_negative': (var_pred_raw < 0).any().item(),
                'var_raw_min': var_pred_raw.min().item() if not torch.isnan(var_pred_raw).any() else None,
                'var_raw_max': var_pred_raw.max().item() if not torch.isnan(var_pred_raw).any() else None,
                'var_raw_num_negative': (var_pred_raw < 0).sum().item(),
                'var_raw_num_total': var_pred_raw.numel(),

                'var_abs_min': var_pred_abs.min().item() if not torch.isnan(var_pred_abs).any() else None,
                'var_abs_max': var_pred_abs.max().item() if not torch.isnan(var_pred_abs).any() else None,
            })
        else:
            issues.update({
                'output_has_nan': torch.isnan(output).any().item(),
                'output_has_inf': torch.isinf(output).any().item(),
            })

        if any(v for k, v in issues.items() if k.endswith('_has_nan') or k.endswith('_has_inf') or k.endswith('_has_negative')):
            self.diagnostics['model_prediction_issues'].append(issues)

        return issues

    def check_loss_computation(self, loss, mean_pred, var_pred, targets, epoch, batch):
        """Check loss computation for numerical issues."""
        issues = {
            'epoch': epoch,
            'batch': batch,
            'loss_is_nan': torch.isnan(loss).any().item() if torch.is_tensor(loss) else (loss != loss),
            'loss_is_inf': torch.isinf(loss).any().item() if torch.is_tensor(loss) else np.isinf(loss),
            'loss_value': loss.item() if torch.is_tensor(loss) else loss,

            'targets_has_nan': torch.isnan(targets).any().item(),
            'targets_min': targets.min().item() if not torch.isnan(targets).any() else None,
            'targets_max': targets.max().item() if not torch.isnan(targets).any() else None,

            'mean_pred_has_nan': torch.isnan(mean_pred).any().item(),
            'var_pred_min': var_pred.min().item() if not torch.isnan(var_pred).any() else None,
            'var_pred_max': var_pred.max().item() if not torch.isnan(var_pred).any() else None,

            # Compute individual loss components
            'log_var_component': None,
            'squared_error_component': None,
        }

        try:
            log_var = 0.5 * torch.log(var_pred)
            squared_error = 0.5 * (targets - mean_pred) ** 2 / var_pred

            issues['log_var_component'] = {
                'has_nan': torch.isnan(log_var).any().item(),
                'mean': log_var.mean().item() if not torch.isnan(log_var).any() else None,
            }

            issues['squared_error_component'] = {
                'has_nan': torch.isnan(squared_error).any().item(),
                'mean': squared_error.mean().item() if not torch.isnan(squared_error).any() else None,
            }
        except:
            pass

        if issues['loss_is_nan'] or issues['loss_is_inf']:
            self.diagnostics['loss_computation_issues'].append(issues)

        return issues

    def check_synthetic_generation(self, mean, var, epoch):
        """Check synthetic data generation for issues."""
        issues = {
            'epoch': epoch,
            'mean_has_nan': torch.isnan(mean).any().item(),
            'mean_has_inf': torch.isinf(mean).any().item(),
            'var_has_nan': torch.isnan(var).any().item(),
            'var_has_inf': torch.isinf(var).any().item(),
            'var_has_negative': (var < 0).any().item(),
            'var_min': var.min().item() if not torch.isnan(var).any() else None,
            'var_max': var.max().item() if not torch.isnan(var).any() else None,
            'var_num_negative': (var < 0).sum().item(),
            'var_num_total': var.numel(),
        }

        self.diagnostics['synthetic_generation_issues'].append(issues)
        return issues

    def log_epoch_summary(self, epoch, train_loss, test_loss, data_source):
        """Log summary statistics for each epoch."""
        self.epoch_stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'data_source': data_source,
            'train_loss_is_nan': train_loss != train_loss if isinstance(train_loss, float) else torch.isnan(torch.tensor(train_loss)).item(),
        })

    def save(self, filename=None):
        """Save diagnostics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'diagnostics_{timestamp}.json'

        # Compute summary
        self.diagnostics['summary'] = {
            'total_training_data_issues': len(self.diagnostics['training_data_issues']),
            'total_model_prediction_issues': len(self.diagnostics['model_prediction_issues']),
            'total_loss_computation_issues': len(self.diagnostics['loss_computation_issues']),
            'total_synthetic_generation_issues': len(self.diagnostics['synthetic_generation_issues']),
            'epochs_with_nan_loss': sum(1 for e in self.epoch_stats if e['train_loss_is_nan']),
            'total_epochs_logged': len(self.epoch_stats),
        }

        self.diagnostics['epoch_stats'] = self.epoch_stats

        output_path = self.log_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.diagnostics, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Diagnostics saved to: {output_path}")
        print(f"{'='*80}")
        print(f"Summary:")
        for key, value in self.diagnostics['summary'].items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")

        return output_path
