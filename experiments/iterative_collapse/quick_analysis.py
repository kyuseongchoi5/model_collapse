#!/usr/bin/env python3
"""Quick analysis script to visualize model collapse results."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open('experiments/iterative_collapse/metrics/gp_switch100_sample_20251116_070131_metrics.json', 'r') as f:
    metrics = json.load(f)

epochs = np.array(metrics['epochs'])
test_loss = np.array(metrics['test_loss_true'])
train_loss = np.array(metrics['train_loss'])
variance = np.array(metrics['prediction_variance'])
lr = np.array(metrics['lr'])

switch_epoch = 100

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Collapse Analysis: PFN Trained on GP Data', fontsize=16, fontweight='bold')

# Plot 1: Test Loss (with log scale for post-switch)
ax1 = axes[0, 0]
pre_switch = epochs <= switch_epoch
post_switch = epochs > switch_epoch

ax1.plot(epochs[pre_switch], test_loss[pre_switch], 'g-', linewidth=2, label='True GP Data', marker='o', markersize=3)
ax1.plot(epochs[post_switch], test_loss[post_switch], 'r-', linewidth=2, label='Synthetic Data', marker='x', markersize=3)
ax1.axvline(switch_epoch, color='black', linestyle='--', linewidth=2, label=f'Switch (epoch {switch_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Test Loss (True GP)', fontsize=12)
ax1.set_title('Test Loss Trajectory', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # Log scale to show full range

# Plot 2: Train vs Test Loss
ax2 = axes[0, 1]
ax2.plot(epochs, train_loss, 'b-', linewidth=1.5, alpha=0.7, label='Train Loss')
ax2.plot(epochs, test_loss, 'r-', linewidth=1.5, alpha=0.7, label='Test Loss')
ax2.axvline(switch_epoch, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Train vs Test Loss (Note: Train goes negative!)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-5, 10)  # Focus on reasonable range

# Plot 3: Prediction Variance (Mode Collapse)
ax3 = axes[1, 0]
ax3.plot(epochs[pre_switch], variance[pre_switch], 'g-', linewidth=2, label='True GP Data', marker='o', markersize=3)
ax3.plot(epochs[post_switch], variance[post_switch], 'r-', linewidth=2, label='Synthetic Data', marker='x', markersize=3)
ax3.axvline(switch_epoch, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Prediction Variance', fontsize=12)
ax3.set_title('Prediction Variance (Mode Collapse)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Key Statistics
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate statistics
pre_test_mean = np.mean(test_loss[pre_switch][-20:])  # Last 20 epochs before switch
post_test_mean = np.mean(test_loss[post_switch][:20])  # First 20 epochs after switch
post_test_final = test_loss[-1]
pre_var_mean = np.mean(variance[pre_switch][-20:])
post_var_final = variance[-1]
degradation = post_test_final / pre_test_mean
var_collapse = post_var_final / pre_var_mean

stats_text = f"""
KEY FINDINGS:

Pre-Switch Performance (epochs 81-100):
  • Mean test loss: {pre_test_mean:.4f}
  • Mean variance: {pre_var_mean:.4f}
  • Status: ✅ Converged successfully

Post-Switch Performance (epochs 101-120):
  • Mean test loss: {post_test_mean:.4f}
  • Initial degradation: {post_test_mean/pre_test_mean:.1f}x worse

Final Performance (epoch 200):
  • Test loss: {post_test_final:.2f}
  • Variance: {post_var_final:.6f}
  • Total degradation: {degradation:.0f}x worse
  • Variance collapse: {var_collapse:.1e} ({(1-var_collapse)*100:.3f}% reduction)

WORST MOMENTS:
  • Highest test loss: {np.max(test_loss[post_switch]):.1f} (epoch {epochs[post_switch][np.argmax(test_loss[post_switch])]})
  • Lowest variance: {np.min(variance[post_switch]):.2e} (epoch {epochs[post_switch][np.argmin(variance[post_switch])]})

DIAGNOSIS: 🔴 CATASTROPHIC MODEL COLLAPSE
  • Severe mode collapse (variance → 0)
  • GaussianNLL exploitation (negative train loss)
  • Complete loss of generalization
  • Model is unusable after epoch ~150
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('experiments/iterative_collapse/plots/model_collapse_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved plot to experiments/iterative_collapse/plots/model_collapse_analysis.png")

# Print summary to console
print("\n" + "="*80)
print("MODEL COLLAPSE ANALYSIS - SUMMARY")
print("="*80)
print(f"\n📊 EXPERIMENT: GP Prior → Synthetic Data Switch at Epoch {switch_epoch}")
print(f"\n✅ PRE-SWITCH (Epochs 1-{switch_epoch}):")
print(f"   Test loss: {test_loss[0]:.3f} → {test_loss[switch_epoch-1]:.3f}")
print(f"   Final 20 epochs mean: {pre_test_mean:.4f}")
print(f"   Variance: {pre_var_mean:.4f} (healthy)")

print(f"\n❌ POST-SWITCH (Epochs {switch_epoch+1}-200):")
print(f"   Test loss trajectory:")
print(f"     Epoch {switch_epoch+1}: {test_loss[switch_epoch]:.3f} ({test_loss[switch_epoch]/pre_test_mean:.1f}x worse)")
print(f"     Epoch 120: {test_loss[119]:.3f}")
print(f"     Epoch 150: {test_loss[149]:.3f}")
print(f"     Epoch 200: {test_loss[199]:.3f} ({test_loss[199]/pre_test_mean:.0f}x worse)")
print(f"     MAX: {np.max(test_loss[post_switch]):.1f} at epoch {epochs[post_switch][np.argmax(test_loss[post_switch])]}")

print(f"\n🔍 MODE COLLAPSE:")
print(f"   Pre-switch variance: {pre_var_mean:.4f}")
print(f"   Post-switch variance: {post_var_final:.6f}")
print(f"   Reduction: {(1-var_collapse)*100:.3f}%")

print(f"\n🚨 PATHOLOGICAL BEHAVIOR:")
print(f"   Train loss went NEGATIVE: {np.min(train_loss[post_switch]):.3f}")
print(f"   This indicates GaussianNLL exploitation!")

print("\n" + "="*80)
print("CONCLUSION: Severe model collapse with GaussianNLL pathology")
print("="*80 + "\n")
