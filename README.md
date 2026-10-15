# SkipNorm: Energy-Conserving Neural Networks & Polar Disentangled Representations

**SkipNorm** is a parameter-free architectural primitive that acts simultaneously as a **dimension-agnostic skip connection** and an **energy-conserving normalization layer**:

$$\hat{y} = \text{SkipNorm}(x, f(x)) = \frac{f(x)}{\|f(x)\|_p + \epsilon} \|x\|_p$$

Unlike additive residual connections ($x + f(x)$), which require $d_{\text{in}} = d_{\text{out}}$ and suffer from $\mathcal{O}(\sqrt{L})$ depthwise norm inflation, and unlike standard normalizations (LayerNorm, RMSNorm), which destroy input signal energy $\|x\|_p$ and collapse manifold degrees of freedom, **SkipNorm** conserves the exact $L_p$ norm across arbitrary dimension reductions, bottlenecks, and nonlinearities.

---

## Repository Structure

```text
skipnorm/
├── skipnorm/
│   ├── __init__.py
│   └── layers.py                             # SkipNorm, AdaptiveSkipNorm, SkipNormResBlock, SkipNormRNN
├── experiments/
│   ├── exp1_jacobian_isometry.py             # Theorem 1 & 2 verification, Jacobian SVD, depth stability
│   ├── exp2_autoencoder_disentanglement.py   # MNIST sweep, 2D polar disentanglement, 400x OOD scale test
│   ├── exp3_vanilla_rnn_signals.py           # 4D Unidirectional Vanilla RNN on EEG, PianoRoll, Speech
│   └── exp4_limitations_and_extensions.py    # Denoising noisy-norm floor, AdaptiveSkipNorm, 24L Curse of Depth
├── tests/
│   └── test_skipnorm.py                      # Property tests for exact norm & Jacobian invariants
└── results/                                  # Generated figures (fig1..fig5.png) and JSON metrics
```

## Quick Start

```bash
# Run mathematical property tests
PYTHONPATH=. uv run python tests/test_skipnorm.py

# Run all 4 experimental suites
PYTHONPATH=. uv run python experiments/exp1_jacobian_isometry.py
PYTHONPATH=. uv run python experiments/exp2_autoencoder_disentanglement.py
PYTHONPATH=. uv run python experiments/exp3_vanilla_rnn_signals.py
PYTHONPATH=. uv run python experiments/exp4_limitations_and_extensions.py
```
