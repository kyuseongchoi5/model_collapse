# Underfit Learning Experiments

## Research Question

**Can synthetic data help a model continue learning when it's still underfit?**

In the original iterative collapse experiments, models were trained until convergence before switching to synthetic data. The results showed different patterns:
- MLP: Stable test loss (crystallization)
- GP: Catastrophic degradation
- Very Diverse MixGP: Moderate degradation

This raises a key question: **Is prior diversity protective against collapse?**

To answer this, we test whether synthetic data can actually be *beneficial* when:
1. The task is hard enough that the model hasn't converged
2. The prior is diverse enough to generate useful synthetic samples

## Hypothesis

If prior diversity is sufficiently high, synthetic data should contain useful signal that allows the model to continue improving, rather than causing collapse or plateauing.

## Experimental Design

### Task Difficulty
- **Input dimension**: 100D (vs 30D in original experiments)
- **Sequence length**: 100 (vs 50 in original experiments)
- **Warmup epochs**: 30 (vs 50 in original experiments)

These changes ensure models are **clearly underfit** at epoch 100.

### Switch Point
- **Epoch 100**: Switch from true data to synthetic data
- **Total epochs**: 300 (to observe long-term trends)

### Diversity Levels

#### Standard Diversity
- MLP: Default hyperparameters (3 layers, 100 hidden, ReLU)
- Mix GP: Moderate hyperparameter variation

#### Ultra Diversity
- MLP:
  - Layers: 3-7 (random)
  - Hidden: 50-499 (random)
  - Activation: Random (ReLU/Tanh/ELU/GELU/LeakyReLU)
  - Init/Noise: Wide gamma distributions
- Mix GP:
  - Very wide hyperparameter priors
  - Random smoothness (nu ∈ {0.5, 1.5, 2.5})
  - **Random kernel type** (Matern/RBF/Periodic/Linear)

### Control Experiments
For each synthetic experiment, run a control where the model trains on true data only for all 300 epochs. This establishes the baseline: "How well can the model learn from true data alone?"

## Experiments Matrix

| Experiment | Prior | Diversity | Data Source | Config |
|------------|-------|-----------|-------------|--------|
| 1 | MLP | Standard | Synthetic (epoch 100+) | `hard_mlp.yaml` |
| 2 | MLP | Ultra | Synthetic (epoch 100+) | `hard_ultra_diverse_mlp.yaml` |
| 3 | Mix GP | Standard | Synthetic (epoch 100+) | `hard_mixgp.yaml` |
| 4 | Mix GP | Ultra | Synthetic (epoch 100+) | `hard_ultra_diverse_mixgp.yaml` |
| 5 | MLP | Standard | True only | `control/control_mlp.yaml` |
| 6 | MLP | Ultra | True only | `control/control_ultra_diverse_mlp.yaml` |
| 7 | Mix GP | Standard | True only | `control/control_mixgp.yaml` |
| 8 | Mix GP | Ultra | True only | `control/control_ultra_diverse_mixgp.yaml` |

## Success Criteria

### Scenario A: Synthetic Data is Helpful ✅
```
Control: epoch 100 → 300: loss 5.0 → 4.5
Synthetic: epoch 100 → 300: loss 5.0 → 4.0
```
→ **Synthetic data accelerates learning** (better than true data!)

### Scenario B: Synthetic Data is Neutral ➡️
```
Control: epoch 100 → 300: loss 5.0 → 4.5
Synthetic: epoch 100 → 300: loss 5.0 → 4.5
```
→ **Synthetic data maintains progress** (as good as true data)

### Scenario C: Synthetic Data is Harmful ❌
```
Control: epoch 100 → 300: loss 5.0 → 4.5
Synthetic: epoch 100 → 300: loss 5.0 → 5.5
```
→ **Traditional collapse** (worse than true data)

## Key Comparisons

1. **Synthetic vs Control** (Does synthetic help?)
   - `hard_mlp` vs `control_mlp`
   - `hard_ultra_diverse_mlp` vs `control_ultra_diverse_mlp`
   - etc.

2. **Standard vs Ultra Diverse** (Does diversity matter?)
   - `hard_mlp` vs `hard_ultra_diverse_mlp`
   - `hard_mixgp` vs `hard_ultra_diverse_mixgp`
   - etc.

3. **MLP vs Mix GP** (Does prior type matter?)
   - `hard_mlp` vs `hard_mixgp`
   - `hard_ultra_diverse_mlp` vs `hard_ultra_diverse_mixgp`
   - etc.

## Running the Experiments

### Run All Experiments
```bash
cd experiments/underfit_learning
./run_all.sh
```

This runs all 8 experiments sequentially. **Warning**: This will take 10-20+ hours depending on hardware.

### Run Individual Experiments
```bash
# Synthetic experiments
python3 ../iterative_collapse/train_with_switch.py --config configs/hard_mlp.yaml
python3 ../iterative_collapse/train_with_switch.py --config configs/hard_ultra_diverse_mlp.yaml
python3 ../iterative_collapse/train_with_switch.py --config configs/hard_mixgp.yaml
python3 ../iterative_collapse/train_with_switch.py --config configs/hard_ultra_diverse_mixgp.yaml

# Control experiments
python3 ../iterative_collapse/train_with_switch.py --config configs/control/control_mlp.yaml
python3 ../iterative_collapse/train_with_switch.py --config configs/control/control_ultra_diverse_mlp.yaml
python3 ../iterative_collapse/train_with_switch.py --config configs/control/control_mixgp.yaml
python3 ../iterative_collapse/train_with_switch.py --config configs/control/control_ultra_diverse_mixgp.yaml
```

## Results Location

- **Metrics**: `experiments/underfit_learning/metrics/*.json`
- **Checkpoints**: `experiments/underfit_learning/checkpoints/*.pt`
- **Plots**: `experiments/underfit_learning/metrics/*.png` (if `generate_plots: true`)

## Analyzing Results

After experiments complete, look for:

1. **Learning curves** (epoch 100-300):
   - Is test loss decreasing, stable, or increasing?
   - How does synthetic compare to control?

2. **Variance trends**:
   - Does variance increase (uncertainty) or decrease (confidence)?
   - Does ultra diversity maintain higher variance?

3. **Diversity effects**:
   - Does ultra diversity show different behavior than standard?
   - Which prior (MLP vs GP) benefits more from diversity?

Example analysis:
```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('metrics/hard_mlp_d100_seq100_*.json') as f:
    synth = json.load(f)
with open('metrics/control_mlp_d100_seq100_*.json') as f:
    control = json.load(f)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(synth['epochs'], synth['test_loss_true'], label='Synthetic')
plt.plot(control['epochs'], control['test_loss_true'], label='Control (True only)')
plt.axvline(100, color='k', linestyle='--', label='Switch point')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.legend()
plt.title('Test Loss: Synthetic vs Control')

plt.subplot(1, 2, 2)
plt.plot(synth['epochs'][100:], synth['test_loss_true'][100:], label='Synthetic')
plt.plot(control['epochs'][100:], control['test_loss_true'][100:], label='Control')
plt.xlabel('Epoch (post-switch)')
plt.ylabel('Test Loss')
plt.legend()
plt.title('Post-Switch Behavior (Epochs 100-300)')

plt.tight_layout()
plt.savefig('synthetic_vs_control_comparison.png')
```

## Implementation Details

### Code Modifications

This experiment required several enhancements to the codebase:

1. **Kernel Diversity** (`priors/fast_gp_mix.py`):
   - Added support for RBF, Periodic, and Linear kernels (not just Matern)
   - Enabled random kernel selection via `kernel_type: "random"`

2. **Activation Diversity** (`priors/utils.py`):
   - Added `random_activation_sampler()` for random activation functions
   - Supports ReLU, Tanh, ELU, GELU, LeakyReLU

3. **Config Processing** (`config_utils.py`):
   - Parse string specifications like `"uniform_int(3, 8)"` into callable samplers
   - Convert YAML configs into prior hyperparameters
   - Makes configs readable and maintainable

4. **Training Script** (`train_with_switch.py`):
   - Integrated `config_utils` for automatic hyperparameter processing
   - Supports control experiments (never switch) via `switch_epoch: 999`

### File Structure

```
experiments/underfit_learning/
├── README.md                          # This file
├── config_utils.py                    # Hyperparameter processing
├── run_all.sh                         # Run all experiments
├── configs/
│   ├── hard_mlp.yaml                  # Standard MLP + synthetic
│   ├── hard_ultra_diverse_mlp.yaml    # Ultra diverse MLP + synthetic
│   ├── hard_mixgp.yaml                # Standard GP + synthetic
│   ├── hard_ultra_diverse_mixgp.yaml  # Ultra diverse GP + synthetic
│   └── control/
│       ├── control_mlp.yaml           # Standard MLP + true only
│       ├── control_ultra_diverse_mlp.yaml
│       ├── control_mixgp.yaml
│       └── control_ultra_diverse_mixgp.yaml
├── checkpoints/                       # Model checkpoints
└── metrics/                          # Training metrics (JSON)
```

## Expected Insights

This experiment will reveal:

1. **Can synthetic data be helpful?**
   - If yes: Under what conditions? (diversity level, prior type)
   - If no: Why not? (collapse patterns, quality issues)

2. **Is diversity protective?**
   - Does ultra diversity show better results than standard?
   - Which aspects of diversity matter most? (architecture, kernel type, etc.)

3. **Prior comparison**:
   - Do MLPs and GPs behave differently?
   - Does the loss function (MSE vs GaussianNLL) affect results?

4. **Model behavior when underfit**:
   - Does the model recognize it's underfit?
   - Does synthetic data quality improve or degrade over time?

## Related Experiments

- **Original iterative collapse** (`experiments/iterative_collapse/`):
  - Tested collapse from converged state
  - Found: MLP stable, GP catastrophic, Diverse GP moderate degradation

- **This experiment**:
  - Tests learning from underfit state
  - Adds: Kernel diversity, activation diversity, control experiments
  - Goal: Determine if diversity enables useful synthetic data

## Citation & Acknowledgments

This experiment builds on the model collapse framework established in:
- Prior Fitting Networks (PFNs) literature
- Iterative training on model-generated data
- Model collapse and mode collapse research

## Questions or Issues?

If you encounter problems:
1. Check GPU memory (100D sequences require significant memory)
2. Verify config files are valid YAML
3. Ensure `config_utils.py` is accessible to training script
4. Check that modified `fast_gp_mix.py` imports GPyTorch kernels correctly
