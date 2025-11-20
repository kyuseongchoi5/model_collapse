# GP Complexity Implementation Summary

## Overview

This implementation adds support for **composite kernels** to test how prior complexity affects model collapse, independently from task difficulty.

## Files Created/Modified

### 1. New Folder Structure

```
experiments/gp_complexity/
├── README.md                    # Documentation
├── IMPLEMENTATION.md            # This file
├── train.py                     # Wrapper script
├── config_utils.py              # Config processing (copied from underfit_learning)
├── configs/
│   ├── baseline_single.yaml     # composite_kernel_prob = 0.0
│   ├── mixed_composite.yaml     # composite_kernel_prob = 0.5
│   └── full_composite.yaml      # composite_kernel_prob = 1.0
├── checkpoints/                 # Model checkpoints (generated)
├── metrics/                     # Training metrics (generated)
└── plots/                       # Plots (generated)
```

### 2. Modified: `priors/fast_gp_mix.py`

#### New Helper Function: `_create_base_kernel()`

```python
def _create_base_kernel(kernel_type, hyperparameters, x, aug_batch_shape):
    """Helper function to create a single base kernel component.

    Supports: matern, rbf, periodic, linear
    Handles 'random' kernel_type and 'random' nu parameter
    Uses period_concentration/period_rate for periodic kernels
    """
```

**Purpose**: Extract kernel creation logic for reuse in composite kernels

#### Modified: `get_model()`

**New hyperparameters**:
- `composite_kernel_prob`: Probability of creating composite vs single kernel [0.0, 1.0]
- `period_concentration`: Gamma prior parameter for periodic kernel periods
- `period_rate`: Gamma prior parameter for periodic kernel periods

**Logic**:
```python
if random.random() < composite_kernel_prob:
    # Create COMPOSITE kernel
    composition = random.choice(['sum', 'product'])  # 50-50 split

    # Create two components with independent hyperparameters
    k1 = ScaleKernel(_create_base_kernel(kernel_type1, ...))
    k2 = ScaleKernel(_create_base_kernel(kernel_type2, ...))

    # Compose
    if composition == 'sum':
        covar_module = k1 + k2
    else:
        covar_module = k1 * k2
else:
    # Create SINGLE kernel (original behavior)
    covar_module = ScaleKernel(_create_base_kernel(kernel_type, ...))
```

**Key features**:
- Backward compatible: `composite_kernel_prob=0.0` (or omitted) → original behavior
- Independent hyperparameters: Each component samples lengthscale, outputscale, nu separately
- Allows same kernel types: Matern + Matern with different hyperparameters is valid
- Binary composition only: k1 ⊕ k2 (not higher-order)

### 3. Modified: `experiments/gp_complexity/config_utils.py`

Added new GP hyperparameters to `GP_NUMERIC_PARAMS`:
```python
GP_NUMERIC_PARAMS = {
    'lengthscale_concentration', 'lengthscale_rate',
    'outputscale_concentration', 'outputscale_rate',
    'noise_concentration', 'noise_rate',
    'period_concentration', 'period_rate',      # NEW
    'composite_kernel_prob'                     # NEW
}
```

**Purpose**: Ensure these stay as numbers (not converted to callables) when processing configs

## Experimental Design

### Controlled Comparison

All experiments use **identical** settings except `composite_kernel_prob`:

| Config | composite_kernel_prob | Prior Complexity |
|--------|----------------------|------------------|
| baseline_single | 0.0 | Single kernels only (4 types) |
| mixed_composite | 0.5 | 50% single, 50% composite |
| full_composite | 1.0 | 100% composite kernels |

**Task parameters** (all identical):
- Dimensions: d=30
- Context length: seq=50
- D/N ratio: 0.6
- Model capacity: Fixed (512 emsize, 6 layers)

### Prior Complexity Calculation

**Baseline (single kernels)**:
- 4 kernel types: {matern, rbf, periodic, linear}
- 3 nu values for Matern: {0.5, 1.5, 2.5}
- Continuous hyperparameters: lengthscale, outputscale, period
- Effective diversity: ~4 kernel families × continuous parameters

**Full composite**:
- Component 1: 4 kernel types
- Component 2: 4 kernel types (independently sampled)
- Composition: 2 types {sum, product}
- Effective diversity: ~32 kernel families (4×4×2) × continuous parameters

**Complexity increase**: ~8x more diverse function space

## How Composite Kernels Increase Diversity

### Additive (k1 + k2)

Examples:
- **RBF + Periodic**: Smooth trend + oscillations
- **Matern + Linear**: Local variations + global trend
- **Matern(short) + Matern(long)**: Multi-scale structure

Function space: Superposition of two independent patterns

### Multiplicative (k1 × k2)

Examples:
- **RBF × Periodic**: Periodic with varying amplitude
- **Linear × Matern**: Non-stationary smoothness
- **Linear × Periodic**: Periodic with trend modulation

Function space: One pattern modulates another

### Independent Hyperparameters

Each component gets its own:
- Lengthscale(s): λ₁ and λ₂ sampled independently
- Outputscale: σ₁² and σ₂² sampled independently
- Smoothness (for Matern): ν₁ and ν₂ sampled independently
- Period (for Periodic): p₁ and p₂ sampled independently

This creates **much richer function spaces** than single kernels.

## Expected Outcomes

### Scenario A: Prior Complexity Doesn't Matter (Task Difficulty Dominates)

If all three configs show similar collapse:
```
baseline_single: test_loss ↑ ~3x
mixed_composite: test_loss ↑ ~3x
full_composite:  test_loss ↑ ~3x
```

**Interpretation**:
- d=30, seq=50 is easy enough (D/N=0.6)
- Model has spare capacity
- Prior complexity doesn't affect collapse when model isn't underfit

### Scenario B: Prior Complexity Independently Drives Collapse

If collapse severity increases with complexity:
```
baseline_single: test_loss ↑ ~3x
mixed_composite: test_loss ↑ ~5x
full_composite:  test_loss ↑ ~7x+
```

**Interpretation**:
- Larger prior space → synthetic loses more diversity
- Prior complexity affects collapse even when model has capacity
- Confirms: Both task difficulty AND prior size matter

### Scenario C: Non-Monotonic (Sweet Spot)

If mixed composite shows unexpected behavior:
```
baseline_single: test_loss ↑ ~3x,  var ↓
mixed_composite: test_loss ↑ ~5x,  var ↑ (like d=30 very_diverse)
full_composite:  test_loss ↑ ~4x,  var ↓
```

**Interpretation**:
- Mixed composite hits "confusion zone"
- Synthetic inconsistency → epistemic uncertainty increases
- Similar to d=30 very_diverse result

## Testing the Implementation

### Basic Sanity Checks

1. **Backward compatibility**:
   ```bash
   # Should behave identically to underfit_learning very_diverse d=30
   python train.py --config configs/baseline_single.yaml
   ```

2. **Composite kernel creation**:
   ```python
   # In Python with torch available
   from priors.fast_gp_mix import get_model
   model, _ = get_model(x, y, {'composite_kernel_prob': 1.0}, sample=False)
   # Check: model.covar_module should be AdditiveKernel or ProductKernel
   ```

3. **Probability distribution**:
   ```python
   # With composite_kernel_prob=0.5, run 100 times
   # Should get ~50 composite, ~50 single
   ```

### Full Experiment Run

```bash
cd experiments/gp_complexity

# Run all three configs
for config in configs/*.yaml; do
    echo "Running $config..."
    python train.py --config $config
done
```

Expected runtime: ~2-3 hours per config (200 epochs × 20 steps × batch generation)

## Integration with Existing Code

### No Changes Required To:
- `experiments/iterative_collapse/train_with_switch.py` (used as-is)
- `experiments/iterative_collapse/evaluate.py` (used as-is)
- Training loop, optimization, evaluation metrics

### Transparent to:
- Model architecture (transformer is unchanged)
- Loss function (GaussianNLL still applies)
- Data generation (get_batch still works)

### Only Change:
- `get_model()` in `priors/fast_gp_mix.py` now supports composite kernels
- Configs specify `composite_kernel_prob` to control this

## Validation

To confirm implementation correctness:

1. ✓ Syntax valid (checked)
2. ✓ Backward compatible (composite_kernel_prob=0.0 uses original path)
3. ✓ Config files created with proper hyperparameters
4. ✓ README and documentation complete

Remaining validation (requires torch environment):
- [ ] Test composite kernel creation
- [ ] Test sampling from composite kernels
- [ ] Run short training (10 epochs) to verify end-to-end
- [ ] Verify metrics are recorded correctly

## Next Steps

1. Run `baseline_single.yaml` first
2. Compare pre-switch metrics with very_diverse d=30 (should match)
3. If baseline looks good, run `mixed_composite.yaml` and `full_composite.yaml`
4. Analyze results to answer: Does prior complexity independently drive collapse?
