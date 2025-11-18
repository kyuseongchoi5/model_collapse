#!/usr/bin/env python3
"""
Standalone script to generate plots from saved metrics JSON files.
Replicates the auto-plotting functionality from train_with_switch.py
"""

import argparse
import json
import re
from pathlib import Path


def parse_experiment_name(filename):
    """
    Parse experiment configuration from filename.
    Expected format: {prefix}_{prior}_d{dims}_seq{len}_sw{switch}of{total}_{auto}_{mode}_r{ratio}_{timestamp}_metrics.json
    Or old format: {prior}_switch{switch}_{mode}_{timestamp}_metrics.json
    """
    config = {}

    # Remove _metrics.json suffix
    name = filename.replace('_metrics.json', '')

    # Try new format first
    match = re.search(r'(?:(.+?)_)?(\w+)_d(\d+)_seq(\d+)_sw(\d+)of(\d+)_(auto|noauto)_(\w+)_r([\d.]+)', name)
    if match:
        prefix, prior, dims, seq, switch, total, auto, mode, ratio = match.groups()
        config = {
            'prior': prior,
            'num_features': int(dims),
            'bptt': int(seq),
            'switch_epoch': int(switch),
            'total_epochs': int(total),
            'use_autoregressive_synthetic': (auto == 'auto'),
            'synthetic_mode': mode,
            'synthetic_ratio': float(ratio),
            'experiment_name': name
        }
        return config

    # Try old format
    match = re.search(r'(\w+)_switch(\d+)_(\w+)', name)
    if match:
        prior, switch, mode = match.groups()
        config = {
            'prior': prior,
            'num_features': None,  # Unknown from filename
            'bptt': None,
            'switch_epoch': int(switch),
            'total_epochs': None,  # Will infer from data
            'use_autoregressive_synthetic': False,  # Unknown
            'synthetic_mode': mode,
            'synthetic_ratio': 1.0,  # Assume full synthetic
            'experiment_name': name
        }
        return config

    # If parsing fails, return minimal config
    return {
        'prior': 'unknown',
        'num_features': None,
        'bptt': None,
        'switch_epoch': None,
        'total_epochs': None,
        'use_autoregressive_synthetic': None,
        'synthetic_mode': 'unknown',
        'synthetic_ratio': None,
        'experiment_name': name
    }


def generate_plot(metrics_file, output_dir=None, config_override=None):
    """
    Generate plots from metrics JSON file.

    Args:
        metrics_file: Path to metrics JSON file
        output_dir: Directory to save plot (default: ./experiments/iterative_collapse/plots)
        config_override: Dict to override parsed config values
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    metrics_file = Path(metrics_file)

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)

    # Parse config from filename
    config = parse_experiment_name(metrics_file.name)

    # Override with provided values
    if config_override:
        config.update(config_override)

    # Infer missing values from data
    epochs = metrics_data['epochs']
    if config['total_epochs'] is None:
        config['total_epochs'] = epochs[-1]

    # If switch_epoch is unknown, try to detect from data_source
    if config['switch_epoch'] is None and 'data_source' in metrics_data:
        data_sources = metrics_data['data_source']
        for i, source in enumerate(data_sources):
            if source == 'synthetic':
                config['switch_epoch'] = epochs[i]
                break
        if config['switch_epoch'] is None:
            config['switch_epoch'] = config['total_epochs']  # No switch

    # Setup output directory
    if output_dir is None:
        output_dir = Path('./experiments/iterative_collapse/plots')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    train_loss = metrics_data['train_loss']
    test_loss = metrics_data['test_loss_true']
    variance = metrics_data['prediction_variance']
    lr_vals = metrics_data['lr']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Collapse Analysis: {config["experiment_name"]}', fontsize=14, fontweight='bold')

    # Plot 1: Train vs Test Loss
    ax1 = axes[0, 0]
    switch_epoch = config['switch_epoch']
    pre_switch = [i for i, e in enumerate(epochs) if e < switch_epoch]
    post_switch = [i for i, e in enumerate(epochs) if e >= switch_epoch]

    if pre_switch:
        ax1.plot([epochs[i] for i in pre_switch], [train_loss[i] for i in pre_switch],
                'b-', alpha=0.7, linewidth=1.5, label='Train (True Data)')
        ax1.plot([epochs[i] for i in pre_switch], [test_loss[i] for i in pre_switch],
                'g-', alpha=0.7, linewidth=1.5, label='Test (True Data)')
    if post_switch:
        ax1.plot([epochs[i] for i in post_switch], [train_loss[i] for i in post_switch],
                'b--', alpha=0.7, linewidth=1.5, label='Train (Synthetic)')
        ax1.plot([epochs[i] for i in post_switch], [test_loss[i] for i in post_switch],
                'r-', alpha=0.7, linewidth=1.5, label='Test (Synthetic)')

    ax1.axvline(switch_epoch, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Switch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Test Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Variance Trajectory
    ax2 = axes[0, 1]
    if pre_switch:
        ax2.plot([epochs[i] for i in pre_switch], [variance[i] for i in pre_switch],
                'g-', linewidth=2, label='True Data')
    if post_switch:
        ax2.plot([epochs[i] for i in post_switch], [variance[i] for i in post_switch],
                'r-', linewidth=2, label='Synthetic Data')
    ax2.axvline(switch_epoch, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Prediction Variance')
    ax2.set_title('Variance Trajectory (Mode Collapse Indicator)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Learning Rate Schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, lr_vals, 'purple', linewidth=2)
    ax3.axvline(switch_epoch, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Find the index for the last epoch before switch (switch_epoch - 1)
    # Epochs array is [1, 2, 3, ..., N], so epoch M is at index M-1
    # We want the last epoch trained on true data, which is switch_epoch - 1
    try:
        pre_switch_idx = epochs.index(switch_epoch - 1)
    except ValueError:
        # If switch_epoch - 1 not in epochs, use the closest before switch
        pre_switch_idx = max(0, switch_epoch - 2)

    pre_baseline = test_loss[pre_switch_idx]
    post_final = test_loss[-1]
    var_baseline = variance[pre_switch_idx]
    var_final = variance[-1]

    # Format config values for display
    dims_str = str(config['num_features']) if config['num_features'] is not None else 'unknown'
    bptt_str = str(config['bptt']) if config['bptt'] is not None else 'unknown'
    auto_str = str(config['use_autoregressive_synthetic']) if config['use_autoregressive_synthetic'] is not None else 'unknown'
    ratio_str = str(config['synthetic_ratio']) if config['synthetic_ratio'] is not None else 'unknown'

    summary_text = f"""
EXPERIMENT SUMMARY

Configuration:
  Prior: {config['prior']}
  Dimensions: {dims_str}
  Sequence Length: {bptt_str}
  Switch Epoch: {switch_epoch}/{config['total_epochs']}
  Autoregressive: {auto_str}
  Synthetic Ratio: {ratio_str}

Pre-Switch (Epoch {switch_epoch}):
  Test Loss: {pre_baseline:.4f}
  Variance: {var_baseline:.4f}

Post-Switch (Epoch {config['total_epochs']}):
  Test Loss: {post_final:.4f}
  Variance: {var_final:.4f}

Degradation:
  Test Loss: {post_final/pre_baseline:.2f}x worse
  Variance: {var_final/var_baseline:.2f}x change

Max Test Loss: {max(test_loss):.2f}
Min Train Loss: {min(train_loss):.4f}
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save plot
    plot_filename = config['experiment_name'] + '.png'
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Plot saved to: {plot_path}")

    # Print summary to console
    print(f"\nSummary for {config['experiment_name']}:")
    print(f"  Pre-switch test loss:  {pre_baseline:.4f}")
    print(f"  Post-switch test loss: {post_final:.4f}")
    print(f"  Degradation:           {post_final/pre_baseline:.2f}x")
    print(f"  Variance change:       {var_final/var_baseline:.2f}x")

    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from model collapse metrics JSON files'
    )
    parser.add_argument('metrics_file', type=str,
                       help='Path to metrics JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: ./experiments/iterative_collapse/plots)')
    parser.add_argument('--switch_epoch', type=int, default=None,
                       help='Override switch epoch from filename')
    parser.add_argument('--num_features', type=int, default=None,
                       help='Override number of features/dimensions')
    parser.add_argument('--bptt', type=int, default=None,
                       help='Override sequence length')

    args = parser.parse_args()

    # Build config override dict
    config_override = {}
    if args.switch_epoch is not None:
        config_override['switch_epoch'] = args.switch_epoch
    if args.num_features is not None:
        config_override['num_features'] = args.num_features
    if args.bptt is not None:
        config_override['bptt'] = args.bptt

    # Generate plot
    generate_plot(args.metrics_file, args.output_dir, config_override)


if __name__ == '__main__':
    main()
