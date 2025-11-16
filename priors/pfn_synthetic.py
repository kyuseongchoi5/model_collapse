"""
PFN Synthetic Data Prior

This prior generates synthetic data using a trained PFN model's predictions.
Used for studying model collapse when training on self-generated data.
"""

import torch
from torch import nn
import torch.distributions as dist

from .utils import get_batch_to_dataloader
from utils import default_device


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device,
              model=None, loss_function='gaussnll', sampling_mode='sample',
              x_sampling='uniform', **kwargs):
    """
    Generate a batch of synthetic data using a trained PFN model.

    Args:
        batch_size: Number of sequences in batch
        seq_len: Length of each sequence
        num_features: Number of input features
        device: Device to use
        model: Trained PFN model to generate predictions from
        loss_function: Type of loss ('gaussnll', 'mse', 'ce', 'barnll')
        sampling_mode: How to generate Y values:
            - 'mean': Use mean prediction only
            - 'sample': Sample from predictive distribution
        x_sampling: How to sample X values:
            - 'uniform': Uniform random in [0, 1]
            - 'normal': Standard normal
        **kwargs: Additional arguments (unused, for compatibility)

    Returns:
        x: Input features, shape (seq_len, batch_size, num_features)
        y: Predicted targets, shape (seq_len, batch_size)
        y: Targets again (for compatibility with prior interface)
    """

    if model is None:
        raise ValueError("model parameter is required for PFN synthetic prior")

    # Sample X values (same distribution as true GP prior for consistency)
    if x_sampling == 'uniform':
        x = torch.rand(batch_size, seq_len, num_features, device=device)
    elif x_sampling == 'normal':
        x = torch.randn(batch_size, seq_len, num_features, device=device)
    else:
        raise ValueError(f"Unknown x_sampling mode: {x_sampling}")

    # Transpose to (seq_len, batch_size, num_features) for model input
    x_transposed = x.transpose(0, 1)

    # Set model to eval mode
    model.eval()
    model.to(device)

    # Generate predictions using the PFN model
    # The model expects input in the format that depends on fuse_x_y setting
    # For simplicity, we'll use the standard forward pass

    # We need to generate predictions autoregressively for each position
    # At position t, we condition on observations 0..t-1 and predict at t
    y_samples = []

    for t in range(seq_len):
        if t == 0:
            # First position: no context, model predicts from prior
            # Create dummy context (zeros)
            context_x = torch.zeros(1, batch_size, num_features, device=device)
            context_y = torch.zeros(1, batch_size, device=device)
        else:
            # Use previous predictions as context
            context_x = x_transposed[:t]
            context_y = torch.stack(y_samples[:t], dim=0)

        # Prepare input based on model's expected format
        # The model expects (x, y) tuple when fuse_x_y=False
        # or fused tensor when fuse_x_y=True
        # For compatibility, we'll check model's architecture

        # Create a simple input: current x position
        current_x = x_transposed[t:t+1]  # (1, batch_size, num_features)

        # For first position or if model expects simple input
        if t == 0:
            # Just predict from the x value
            # Use a dummy single eval pos
            output = model(current_x, single_eval_pos=0)
        else:
            # Concatenate context
            full_x = torch.cat([context_x, current_x], dim=0)
            # Prepare y context with zeros for current position
            full_y = torch.cat([context_y, torch.zeros(1, batch_size, device=device)], dim=0)

            # Create input tuple if model expects it
            try:
                # Try with tuple input (unfused mode)
                output = model((full_x, full_y), single_eval_pos=t)
            except:
                # Try with fused input
                # Fuse x and y: concatenate previous y values with x
                fused = torch.cat([
                    full_x,
                    torch.cat([torch.zeros_like(full_y[:1]), full_y[:-1]], 0).unsqueeze(-1)
                ], dim=-1)
                output = model(fused, single_eval_pos=t)

        # Extract prediction at current position
        # Output shape depends on loss function
        pred_at_t = output[-1]  # (batch_size, n_out) or (batch_size,)

        # Generate Y value based on loss function and sampling mode
        if loss_function in ['gaussnll']:
            # Output is (mean, var) or just mean
            if pred_at_t.shape[-1] == 2:
                # Mean and variance predicted
                mean = pred_at_t[:, 0]
                var = pred_at_t[:, 1].clamp(min=1e-4)  # Ensure positive variance
            else:
                # Only mean predicted
                mean = pred_at_t.squeeze(-1) if len(pred_at_t.shape) > 1 else pred_at_t
                var = torch.ones_like(mean) * 0.1  # Default variance

            if sampling_mode == 'sample':
                # Sample from Gaussian distribution
                y_t = torch.normal(mean, torch.sqrt(var))
            else:
                # Use mean only
                y_t = mean

        elif loss_function == 'mse':
            # Output is just the prediction
            y_t = pred_at_t.squeeze(-1) if len(pred_at_t.shape) > 1 else pred_at_t

        elif loss_function == 'ce':
            # Classification: output is logits
            if sampling_mode == 'sample':
                # Sample from categorical distribution
                probs = torch.softmax(pred_at_t, dim=-1)
                y_t = torch.multinomial(probs, num_samples=1).squeeze(-1).float()
            else:
                # Use argmax
                y_t = torch.argmax(pred_at_t, dim=-1).float()

        elif loss_function in ['barnll', 'adaptivebarnll', 'adaptivefullsupportbarnll']:
            # Bar distribution: output is logits over buckets
            if sampling_mode == 'sample':
                # Sample from the discrete distribution
                probs = torch.softmax(pred_at_t, dim=-1)
                bucket_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
                # For now, use bucket index as y value
                # In practice, you might want to map this back to continuous values
                y_t = bucket_idx.float()
            else:
                # Use mode (argmax)
                y_t = torch.argmax(pred_at_t, dim=-1).float()
        else:
            raise ValueError(f"Unknown loss_function: {loss_function}")

        y_samples.append(y_t)

    # Stack y values: (seq_len, batch_size)
    y = torch.stack(y_samples, dim=0)

    # Set model back to train mode
    model.train()

    return x_transposed, y, y


# Simplified version that doesn't do autoregressive generation
# Instead, generates all predictions at once (faster but less accurate)
@torch.no_grad()
def get_batch_simple(batch_size, seq_len, num_features, device=default_device,
                    model=None, loss_function='gaussnll', sampling_mode='sample',
                    x_sampling='uniform', **kwargs):
    """
    Simplified batch generation: predict all positions at once.
    Faster but less accurate than autoregressive generation.
    """

    if model is None:
        raise ValueError("model parameter is required for PFN synthetic prior")

    # Sample X values
    if x_sampling == 'uniform':
        x = torch.rand(batch_size, seq_len, num_features, device=device)
    elif x_sampling == 'normal':
        x = torch.randn(batch_size, seq_len, num_features, device=device)
    else:
        raise ValueError(f"Unknown x_sampling mode: {x_sampling}")

    # Transpose to (seq_len, batch_size, num_features)
    x_transposed = x.transpose(0, 1)

    # Set model to eval mode
    model.eval()
    model.to(device)

    # Generate all predictions at once
    # Use single_eval_pos=1 to generate predictions for positions 1..seq_len-1
    # (using position 0 as minimal context)
    single_eval_pos = 1  # Minimal context - predict from position 1 onwards

    # Prepare input as (x, y) tuple for unfused mode
    y_context = torch.zeros(seq_len, batch_size, device=device)

    try:
        # Try with tuple input (unfused mode)
        output = model((x_transposed, y_context), single_eval_pos=single_eval_pos)
    except:
        # If that fails, try with fused input
        try:
            fused = torch.cat([
                x_transposed,
                torch.zeros(seq_len, batch_size, 1, device=device)
            ], dim=-1)
            output = model(fused, single_eval_pos=single_eval_pos)
        except:
            # Last resort: simple forward with single_eval_pos
            output = model(x_transposed, single_eval_pos=single_eval_pos)

    # Output shape: (seq_len - single_eval_pos, batch_size, n_out)
    # For single_eval_pos=1, this gives us predictions for positions 1..seq_len-1

    # Process output based on loss function to get y values for positions 1..seq_len-1
    if loss_function in ['gaussnll']:
        if output.shape[-1] == 2:
            mean = output[..., 0]
            var = output[..., 1].clamp(min=1e-4)
        else:
            mean = output.squeeze(-1) if len(output.shape) > 2 else output
            var = torch.ones_like(mean) * 0.1

        if sampling_mode == 'sample':
            y_pred = torch.normal(mean, torch.sqrt(var))
        else:
            y_pred = mean

    elif loss_function == 'mse':
        y_pred = output.squeeze(-1) if len(output.shape) > 2 else output

    elif loss_function == 'ce':
        if sampling_mode == 'sample':
            probs = torch.softmax(output, dim=-1)
            y_pred = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(-1, batch_size).float()
        else:
            y_pred = torch.argmax(output, dim=-1).float()

    elif loss_function in ['barnll', 'adaptivebarnll', 'adaptivefullsupportbarnll']:
        if sampling_mode == 'sample':
            probs = torch.softmax(output, dim=-1)
            y_pred = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(-1, batch_size).float()
        else:
            y_pred = torch.argmax(output, dim=-1).float()
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}")

    # Pad with a dummy value for position 0 (since we only predicted positions 1..seq_len-1)
    # Use a zero or sample from a simple prior
    y_0 = torch.zeros(1, batch_size, device=device)  # Dummy value for position 0
    y = torch.cat([y_0, y_pred], dim=0)  # Now shape is (seq_len, batch_size)

    # Set model back to train mode
    model.train()

    return x_transposed, y, y


# Create DataLoader classes for both versions
DataLoader = get_batch_to_dataloader(get_batch_simple)  # Use simple version by default (faster)
DataLoader.num_outputs = 1
DataLoader.num_features = None  # Will be set during initialization

DataLoaderAutoregressive = get_batch_to_dataloader(get_batch)
DataLoaderAutoregressive.num_outputs = 1
DataLoaderAutoregressive.num_features = None
