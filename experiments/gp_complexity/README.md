# GP Complexity Experiments

This folder contains experiments to test how **prior complexity** affects model collapse in the synthetic data training paradigm.

## Motivation

Previous experiments in `underfit_learning/` confounded two factors:
1. **Task difficulty** (dimension/context ratio)
2. **Prior complexity** (diversity of GP hyperparameters)

This experiment isolates the effect of **prior complexity** by:
- Keeping task difficulty **constant** (d=30, seq=50)
- Varying **prior complexity** through composite kernels

## Composite Kernels

### What are Composite Kernels?

Instead of using a single kernel `k`, composite kernels combine two kernels:
- **Additive**: `k = k₁ + k₂` (superposition of patterns)
- **Multiplicative**: `k = k₁ × k₂` (modulated patterns)

Each component gets **independent hyperparameters** sampled from the same priors, massively increasing the diversity of the function space.

### Examples

1. **RBF + Periodic**: Smooth baseline with oscillations
2. **Matern × Linear**: Non-stationary smoothness with trend
3. **Matern(λ=0.1) + Matern(λ=5.0)**: Multi-scale structure

## Experiments

### 1. Baseline Single Kernels (`baseline_single.yaml`)
- `composite_kernel_prob = 0.0`
- All GPs use **single kernels** (matern, rbf, periodic, or linear)
- **Backward compatible** with `underfit_learning` configs
- **Purpose**: Baseline for comparison

### 2. Mixed Composite (`mixed_composite.yaml`)
- `composite_kernel_prob = 0.5`
- 50% single kernels, 50% composite kernels (sum/product split 50-50)
- **Purpose**: Test intermediate prior complexity

### 3. Full Composite (`full_composite.yaml`)
- `composite_kernel_prob = 1.0`
- All GPs use **composite kernels**
- Maximum prior diversity
- **Purpose**: Test if pure prior complexity drives collapse independently

## Research Questions

### Q1: Does prior complexity independently drive collapse?

**Hypothesis A**: Prior size doesn't matter when model has spare capacity
- Prediction: `baseline_single` ≈ `full_composite` (similar collapse)
- Interpretation: Task difficulty (D/N ratio) is the primary driver

**Hypothesis B**: Prior size always matters
- Prediction: `baseline_single` < `mixed_composite` < `full_composite` (worse collapse)
- Interpretation: Larger prior → more diversity lost in synthetic → worse collapse

### Q2: How does this compare to d=100, seq=100?

From `underfit_learning/`:
- Ultra diverse d=100, seq=100: Catastrophic collapse (test loss 6.86x)
- Very diverse d=30, seq=50: Moderate degradation (test loss 2.97x)

Is the d=100 collapse due to:
1. **Underfit regime** (D/N = 1.0) → Model finds synthetic easier
2. **Prior complexity** (4 kernels + random nu) → Hard to maintain diversity

This experiment answers: "Would d=30 with full composite kernels show similar collapse patterns?"

## Usage

### Run Experiments

```bash
cd experiments/gp_complexity

# Baseline (single kernels only)
python train.py --config configs/baseline_single.yaml

# Mixed composite (50% composite)
python train.py --config configs/mixed_composite.yaml

# Full composite (100% composite)
python train.py --config configs/full_composite.yaml
```

### Compare Results

All three experiments use:
- Same dimensions (d=30)
- Same context length (seq=50)
- Same model capacity
- Same training procedure

**Only difference**: `composite_kernel_prob` (0.0 vs 0.5 vs 1.0)

## Implementation Details

### Modified Files

1. **`priors/fast_gp_mix.py`**:
   - Added `_create_base_kernel()` helper function
   - Added `composite_kernel_prob` parameter support
   - Added `period_concentration` and `period_rate` for periodic kernels
   - Composite logic: Randomly select sum/product, create two independent components

2. **`experiments/gp_complexity/config_utils.py`**:
   - Added new GP hyperparameters to `GP_NUMERIC_PARAMS`

### Backward Compatibility

Setting `composite_kernel_prob = 0.0` (or omitting it) recovers the **exact original behavior**:
- 100% single kernels
- Identical to `underfit_learning` experiments

## Expected Outputs

Each experiment generates:
- `metrics/*.json`: Training/test loss, variance trajectories
- `checkpoints/*.pt`: Model checkpoints
- `plots/*.png`: Visualization plots

Compare:
- Pre-switch baseline (all should be similar if model capacity is sufficient)
- Post-switch degradation (test how prior complexity affects collapse)
- Variance trajectories (does complexity increase epistemic uncertainty?)

## Next Steps

After running these experiments, we can:
1. Quantify the **independent effect of prior complexity**
2. Test if composite kernels on d=100, seq=100 still show catastrophic collapse
3. Design interventions (e.g., curriculum learning, diversity regularization)
