"""
Visualization script for model collapse experiment.

Creates plots showing:
- Training/test loss over time with switch point
- Prediction variance evolution
- Sample functions at different epochs
- Calibration plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch import nn

from evaluate import (load_metrics, load_checkpoint, sample_functions_from_model,
                     compute_calibration_error)
from transformer import TransformerModel
import encoders
import positional_encodings
import priors


def plot_loss_over_time(metrics, save_path=None):
    """
    Plot training and test loss over epochs, with vertical line at switch point.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    test_loss = metrics['test_loss_true']

    # Find switch point
    switch_epoch = None
    for i, source in enumerate(metrics['data_source']):
        if source != 'true_gp':
            switch_epoch = epochs[i]
            break

    # Plot losses
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, test_loss, 'r-', label='Test Loss (True GP)', linewidth=2)

    # Mark switch point
    if switch_epoch:
        ax.axvline(x=switch_epoch, color='green', linestyle='--',
                   linewidth=2, label=f'Switch to Synthetic (epoch {switch_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Over Time: Training vs. Test on True GP Data', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_variance_over_time(metrics, save_path=None):
    """
    Plot prediction variance over epochs.
    """
    if 'prediction_variance' not in metrics or not metrics['prediction_variance']:
        print("No variance data available")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = metrics['epochs']
    variance = metrics['prediction_variance']

    # Find switch point
    switch_epoch = None
    for i, source in enumerate(metrics['data_source']):
        if source != 'true_gp':
            switch_epoch = epochs[i]
            break

    # Plot variance
    ax.plot(epochs, variance, 'purple', linewidth=2)

    # Mark switch point
    if switch_epoch:
        ax.axvline(x=switch_epoch, color='green', linestyle='--',
                   linewidth=2, label=f'Switch to Synthetic (epoch {switch_epoch})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Prediction Variance', fontsize=12)
    ax.set_title('Prediction Uncertainty Over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved variance plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_rate_schedule(metrics, save_path=None):
    """
    Plot learning rate schedule over epochs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = metrics['epochs']
    lr = metrics['lr']

    # Find switch point
    switch_epoch = None
    for i, source in enumerate(metrics['data_source']):
        if source != 'true_gp':
            switch_epoch = epochs[i]
            break

    # Plot learning rate
    ax.plot(epochs, lr, 'orange', linewidth=2)

    # Mark switch point
    if switch_epoch:
        ax.axvline(x=switch_epoch, color='green', linestyle='--',
                   linewidth=2, label=f'Switch to Synthetic')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning rate plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_combined_metrics(metrics, save_path=None):
    """
    Plot all metrics in a single figure with subplots.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    epochs = metrics['epochs']

    # Find switch point
    switch_epoch = None
    for i, source in enumerate(metrics['data_source']):
        if source != 'true_gp':
            switch_epoch = epochs[i]
            break

    # Subplot 1: Losses
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, metrics['test_loss_true'], 'r-', label='Test Loss (True GP)', linewidth=2)
    if switch_epoch:
        ax1.axvline(x=switch_epoch, color='green', linestyle='--',
                    linewidth=2, label=f'Switch to Synthetic (epoch {switch_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Variance
    ax2 = fig.add_subplot(gs[1, 0])
    if metrics['prediction_variance']:
        ax2.plot(epochs, metrics['prediction_variance'], 'purple', linewidth=2)
        if switch_epoch:
            ax2.axvline(x=switch_epoch, color='green', linestyle='--', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Prediction Variance', fontsize=11)
        ax2.set_title('Prediction Uncertainty', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    # Subplot 3: Learning Rate
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, metrics['lr'], 'orange', linewidth=2)
    if switch_epoch:
        ax3.axvline(x=switch_epoch, color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Model Collapse Experiment: Training Metrics', fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined metrics plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_collapse_summary(metrics, save_path=None):
    """
    Create a summary plot focusing on the collapse phenomenon.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    epochs = np.array(metrics['epochs'])
    test_loss = np.array(metrics['test_loss_true'])

    # Find switch point
    switch_epoch = None
    for i, source in enumerate(metrics['data_source']):
        if source != 'true_gp':
            switch_epoch = epochs[i]
            switch_idx = i
            break

    if switch_epoch is None:
        print("No switch point found in data")
        return

    # Left plot: Loss comparison before/after switch
    ax1 = axes[0]
    before_switch = epochs < switch_epoch
    after_switch = epochs >= switch_epoch

    ax1.plot(epochs[before_switch], test_loss[before_switch],
             'b-', linewidth=3, label='Before Switch (True GP Data)')
    ax1.plot(epochs[after_switch], test_loss[after_switch],
             'r-', linewidth=3, label='After Switch (Synthetic Data)')
    ax1.axvline(x=switch_epoch, color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='Switch Point')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Loss (True GP)', fontsize=12)
    ax1.set_title('Model Performance on True GP Test Set', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: Relative degradation
    ax2 = axes[1]

    # Calculate loss at switch point
    baseline_loss = test_loss[switch_idx - 1]  # Loss just before switch
    relative_loss = (test_loss - baseline_loss) / baseline_loss * 100

    ax2.plot(epochs, relative_loss, 'darkred', linewidth=3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(x=switch_epoch, color='green', linestyle='--',
                linewidth=2, alpha=0.7, label='Switch Point')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Relative Change in Test Loss (%)', fontsize=12)
    ax2.set_title('Performance Degradation After Switch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved collapse summary plot to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print summary statistics
    print("\n" + "="*80)
    print("COLLAPSE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Switch epoch: {switch_epoch}")
    print(f"Test loss before switch (epoch {switch_epoch-1}): {baseline_loss:.4f}")
    print(f"Test loss after switch (epoch {switch_epoch}): {test_loss[switch_idx]:.4f}")
    print(f"Final test loss (epoch {epochs[-1]}): {test_loss[-1]:.4f}")
    print(f"Relative change at switch: {relative_loss[switch_idx]:.2f}%")
    print(f"Relative change at end: {relative_loss[-1]:.2f}%")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize model collapse experiment results')
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='Path to metrics JSON file')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--show', action='store_true',
                       help='Show plots instead of saving')

    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from {args.metrics_file}...")
    metrics = load_metrics(args.metrics_file)

    # Create output directory
    if not args.show:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plot filenames
        base_name = Path(args.metrics_file).stem
        loss_plot = output_dir / f'{base_name}_loss.png'
        variance_plot = output_dir / f'{base_name}_variance.png'
        lr_plot = output_dir / f'{base_name}_lr.png'
        combined_plot = output_dir / f'{base_name}_combined.png'
        collapse_plot = output_dir / f'{base_name}_collapse_summary.png'
    else:
        loss_plot = variance_plot = lr_plot = combined_plot = collapse_plot = None

    # Create plots
    print("\nGenerating plots...")
    plot_loss_over_time(metrics, loss_plot)
    plot_variance_over_time(metrics, variance_plot)
    plot_learning_rate_schedule(metrics, lr_plot)
    plot_combined_metrics(metrics, combined_plot)
    plot_collapse_summary(metrics, collapse_plot)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
