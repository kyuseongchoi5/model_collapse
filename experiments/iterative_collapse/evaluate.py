"""
Evaluation utilities for model collapse experiment.

Functions for evaluating model performance on fixed test sets and computing
various metrics related to model collapse.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import nn
import numpy as np
from typing import List, Tuple, Dict


@torch.no_grad()
def evaluate_on_test_set(model, test_batches, criterion, n_out, device):
    """
    Evaluate model on a fixed test set.

    Args:
        model: The PFN model
        test_batches: List of (data, targets) tuples
        criterion: Loss function
        n_out: Number of outputs
        device: Device to evaluate on

    Returns:
        avg_loss: Average loss across test set
        avg_variance: Average prediction variance (for uncertainty tracking)
    """
    model.eval()
    total_loss = 0.0
    total_variance = 0.0
    n_samples = 0

    for data, targets in test_batches:
        # Move data to device
        if isinstance(data, tuple):
            data = tuple(e.to(device) for e in data)
        else:
            data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        output = model(data, single_eval_pos=None)

        # Compute loss
        if isinstance(criterion, nn.GaussianNLLLoss):
            mean_pred = output[..., 0]
            var_pred = output[..., 1].abs()
            losses = criterion(mean_pred.flatten(), targets.flatten(), var=var_pred.flatten())
            # Track variance
            total_variance += var_pred.mean().item()
        elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
            losses = criterion(output.flatten(), targets.flatten())
            # For MSE, compute variance of predictions
            total_variance += output.var().item()
        else:
            losses = criterion(output.reshape(-1, n_out), targets.flatten())
            # For classification, use entropy as a proxy for uncertainty
            probs = torch.softmax(output, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            total_variance += entropy.item()

        total_loss += losses.mean().item()
        n_samples += 1

    model.train()
    return total_loss / n_samples, total_variance / n_samples


@torch.no_grad()
def compute_prediction_diversity(model, test_batches, device, n_samples=10):
    """
    Compute diversity of predictions across multiple forward passes.

    This measures whether the model is producing diverse outputs or
    collapsing to similar predictions.

    Args:
        model: The PFN model
        test_batches: List of (data, targets) tuples
        device: Device to evaluate on
        n_samples: Number of times to sample predictions

    Returns:
        diversity_score: Variance of predictions across samples
    """
    model.eval()
    all_predictions = []

    # Get predictions multiple times (sampling if stochastic)
    for _ in range(n_samples):
        batch_predictions = []
        for data, _ in test_batches:
            if isinstance(data, tuple):
                data = tuple(e.to(device) for e in data)
            else:
                data = data.to(device)

            output = model(data)

            # Extract mean prediction
            if output.shape[-1] == 2:  # GaussianNLL
                mean_pred = output[..., 0]
            else:
                mean_pred = output.squeeze(-1) if len(output.shape) > 2 else output

            batch_predictions.append(mean_pred.cpu())

        all_predictions.append(torch.cat(batch_predictions, dim=1))

    # Stack predictions: (n_samples, seq_len, batch_size)
    all_predictions = torch.stack(all_predictions, dim=0)

    # Compute variance across samples
    diversity = all_predictions.var(dim=0).mean().item()

    model.train()
    return diversity


@torch.no_grad()
def compute_calibration_error(model, test_batches, device, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) for probabilistic predictions.

    Only works for models that output uncertainties (GaussianNLL or BarDistribution).

    Args:
        model: The PFN model
        test_batches: List of (data, targets) tuples
        device: Device to evaluate on
        n_bins: Number of bins for calibration

    Returns:
        ece: Expected Calibration Error
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_variances = []

    for data, targets in test_batches:
        if isinstance(data, tuple):
            data = tuple(e.to(device) for e in data)
        else:
            data = data.to(device)

        output = model(data)

        if output.shape[-1] == 2:  # GaussianNLL
            mean_pred = output[..., 0]
            var_pred = output[..., 1].abs()

            all_predictions.append(mean_pred.flatten().cpu())
            all_variances.append(var_pred.flatten().cpu())
            all_targets.append(targets.flatten().cpu())

    if not all_variances:
        # Model doesn't output uncertainties
        model.train()
        return None

    predictions = torch.cat(all_predictions)
    variances = torch.cat(all_variances)
    targets = torch.cat(all_targets)

    # Compute squared errors
    squared_errors = (predictions - targets) ** 2

    # Bin by predicted variance
    var_min, var_max = variances.min(), variances.max()
    bin_edges = torch.linspace(var_min, var_max, n_bins + 1)

    ece = 0.0
    total_samples = len(variances)

    for i in range(n_bins):
        # Find samples in this bin
        mask = (variances >= bin_edges[i]) & (variances < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper edge in last bin
            mask = mask | (variances == bin_edges[i + 1])

        if mask.sum() == 0:
            continue

        # Average predicted variance in this bin
        avg_pred_var = variances[mask].mean()

        # Average squared error in this bin
        avg_squared_error = squared_errors[mask].mean()

        # Calibration error for this bin
        bin_error = abs(avg_pred_var - avg_squared_error)

        # Weight by number of samples in bin
        bin_weight = mask.sum().float() / total_samples

        ece += bin_weight * bin_error

    model.train()
    return ece.item()


@torch.no_grad()
def sample_functions_from_model(model, test_batch, device, n_functions=10):
    """
    Sample multiple function predictions from the model.

    Args:
        model: The PFN model
        test_batch: A single (data, targets) tuple
        device: Device to use
        n_functions: Number of functions to sample

    Returns:
        x_values: Input x values (seq_len, num_features)
        y_samples: Sampled y values (n_functions, seq_len)
        y_true: True y values (seq_len,)
    """
    model.eval()

    data, targets = test_batch

    # Take first sequence from batch
    if isinstance(data, tuple):
        x_seq = data[0][:, 0, :]  # (seq_len, num_features)
        data_single = tuple(e[:, 0:1, :] for e in data)
    else:
        x_seq = data[:, 0, :]
        data_single = data[:, 0:1, :]

    y_true = targets[:, 0]  # (seq_len,)

    y_samples = []

    for _ in range(n_functions):
        if isinstance(data_single, tuple):
            data_input = tuple(e.to(device) for e in data_single)
        else:
            data_input = data_single.to(device)

        output = model(data_input)

        # Extract prediction
        if output.shape[-1] == 2:  # GaussianNLL
            mean_pred = output[..., 0, 0]
            var_pred = output[..., 1, 0].abs()
            # Sample from Gaussian
            y_pred = torch.normal(mean_pred, torch.sqrt(var_pred))
        else:
            y_pred = output[:, 0].squeeze()

        y_samples.append(y_pred.cpu())

    model.train()

    return x_seq.cpu(), torch.stack(y_samples), y_true.cpu()


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def load_metrics(metrics_path):
    """
    Load metrics from JSON file.

    Args:
        metrics_path: Path to metrics JSON file

    Returns:
        metrics: Dictionary of metrics
    """
    import json
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


if __name__ == '__main__':
    # Test evaluation functions
    print("Evaluation utilities loaded successfully")
    print("\nAvailable functions:")
    print("  - evaluate_on_test_set: Compute loss on test set")
    print("  - compute_prediction_diversity: Measure prediction diversity")
    print("  - compute_calibration_error: Compute ECE for uncertainty calibration")
    print("  - sample_functions_from_model: Sample predicted functions")
    print("  - load_checkpoint: Load model checkpoint")
    print("  - load_metrics: Load training metrics")
