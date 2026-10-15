# Norm-Transferring Skip Connections and Radial Foliation of Conserved Quantities in Dynamical Autoencoders

Official repository for the TMLR submission **"Norm-Transferring Skip Connections and Radial Foliation of Conserved Quantities in Dynamical Autoencoders"**.

## Overview

We study **SkipNorm**, a norm-transferring skip connection
$$\mathrm{SkipNorm}_p(x; f) := \frac{f(x)}{\|f(x)\|_p}\,\|x\|_p,$$
alongside its tangent-projected retraction **Geodesic SkipNorm**:
$$\mathrm{GeodSkipNorm}_\alpha(x; f) := \frac{x + \alpha f_\perp(x)}{\|x + \alpha f_\perp(x)\|_2}\,\|x\|_2, \qquad f_\perp(x) = f(x) - \frac{\langle f(x), x \rangle}{\|x\|_2^2} x,$$
and its multi-invariant toroidal generalization **$K$-Block Toroidal Geodesic SkipNorm** on $\bigoplus_{k=1}^K \mathbb{R}^{d_k}$.

## Repository Structure

```text
skipnorm/
├── skipnorm/
│   ├── __init__.py
│   ├── jax_layers.py                                  # JAX/Equinox layers, stabilized normalizer spectra, bijective RealNVP
│   └── metrics.py                                     # Whole-orbit radial CV, guarded OVR, isotonic R^2, winding numbers
├── data/
│   └── splits/                                        # Whole-orbit disjoint train/val/test archives & SHA-256 manifest
├── experiments/
│   ├── stage1_protocol_and_counterexamples.py         # Split generation, Jacobian counterexamples, stabilized SVD & controls
│   ├── stage2_foliation_and_obstructions.py           # 5-seed Lotka-Volterra, Double-Well Jordan-Brouwer & 4D Torus benchmarks
│   ├── stage3_dnls_benchmark.py                       # 5-seed 1D DNLS soliton rollout benchmark (no solver projection)
│   └── stage4_generate_publication_figures.py         # Publication vector PDFs (paper/) & 600 DPI PNGs + manifests (results/figures/)
├── tests/
│   └── test_jax_skipnorm.py                           # Float64 JAX unit & property tests for Propositions 1-3 & Theorems 1-3
├── results/
│   ├── stage1/stage1_results.json
│   ├── stage2/stage2_results.json
│   ├── stage3/stage3_dnls_results.json
│   └── figures/                                       # 600 DPI PNG previews and JSON provenance manifests
├── paper/
│   ├── main.tex                                       # TMLR LaTeX manuscript
│   ├── main.pdf                                       # Compiled TMLR PDF
│   ├── references.bib                                 # Bibliography
│   ├── tmlr.sty / tmlr.bst                            # Official TMLR style files
│   ├── fig1_geometry_and_spectra.pdf
│   ├── fig2_dnls_unitary_rollout.pdf
│   ├── fig3_scalar_foliation_lotka_volterra.pdf
│   ├── fig4_topological_obstructions_and_controls.pdf
│   └── fig5_toroidal_kblock_foliation.pdf
└── archive/                                           # Archived files from earlier drafts and exploratory prototypes
```

## Reproducing All Results and Figures

```bash
# 1. Run all mathematical property & unit tests (JAX float64)
PYTHONPATH=. uv run python tests/test_jax_skipnorm.py

# 2. Stage 1: Generate whole-orbit disjoint splits & verify exact Jacobian spectra / analytical controls
PYTHONPATH=. uv run python experiments/stage1_protocol_and_counterexamples.py

# 3. Stage 2: Run 5-seed Lotka-Volterra, Double-Well Jordan-Brouwer, and 4D Coupled Torus benchmarks
PYTHONPATH=. uv run python experiments/stage2_foliation_and_obstructions.py

# 4. Stage 3: Run 5-seed 1D DNLS unitary wavepacket rollout benchmark (no post-step projection)
PYTHONPATH=. uv run python experiments/stage3_dnls_benchmark.py

# 5. Stage 4: Generate all 5 publication figures
PYTHONPATH=. uv run python experiments/stage4_generate_publication_figures.py

# 6. Compile the TMLR manuscript
cd paper && tectonic -X compile main.tex
```
