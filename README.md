# Conditional Radial Foliation in Norm-Preserving Dynamical Autoencoders

Code, locked datasets, and manuscript sources for the TMLR submission of this title.
The paper studies when a norm-preserving latent dynamics model identifies physical
invariant level sets, and when topology prevents that identification. The spherical
projection--retraction operator is established prior work; the manuscript attributes
it to manifold optimization and SpheretNorm and focuses on conditional identification
results and their limitations.

The norm-transfer map is

$$\mathrm{SkipNorm}_p(x;f)=\frac{f(x)}{\|f(x)\|_p}\|x\|_p,$$

where the denominator is nonzero. Its Euclidean tangent-projected version is

$$\mathrm{GSN}_{\alpha}(x;f)=\frac{x+\alpha f_\perp(x)}{\|x+\alpha f_\perp(x)\|_2}\|x\|_2,
\qquad f_\perp(x)=f(x)-\frac{\langle f(x),x\rangle}{\|x\|_2^2}x.$$

Blockwise updates preserve a vector of block radii. Exact norm preservation alone
does not imply a fully isometric Jacobian, injective encoding, or discovery from
finite training data. The manuscript states the theorem hypotheses and numerical
safeguards.

## Files and provenance

- `skipnorm/jax_layers.py`: model layers and encoders.
- `skipnorm/metrics.py`: radius variation, guarded OVR, held-out scalar calibration,
  and invariant readouts.
- `data/splits/`: fixed trajectory arrays and `splits_manifest.json`, with SHA-256
  hashes, shapes, and generator metadata.
- `experiments/stage1_protocol_and_counterexamples.py`: analytical controls and
  dataset generation; `--checks-only` retains the locked data.
- `experiments/stage2_foliation_and_obstructions.py`: five-seed latent dynamics
  experiments for Lotka--Volterra, double wells, and coupled oscillators.
- `experiments/stage3_dnls_benchmark.py`: five-seed DNLS forecasts and inference
  timing; the reference solver does not project states after integration steps.
- `experiments/validate_submission.py`: data hashes, initial-state overlap checks,
  and DNLS reference refinement checks.
- `experiments/stage4_generate_publication_figures.py`: self-contained Matplotlib
  figure export; no external plotting plugins are needed.
- `experiments/export_paper_tables.py`: deterministic, standard-library-only table
  and numerical-macro export, with input hashes and consistency checks.
- `results/stage1/`, `results/stage2/`, `results/stage3/`: current aggregate JSON.
  Stage 2 run directories retain configurations, checkpoints, encoded trajectories,
  predictions, and evaluation diagnostics. Stage 3 retains checkpoints, forecasts,
  timing samples, and configuration hashes. Source snapshots and migration records
  in these directories preserve the code versions supporting the current results;
  include them in the reproducibility bundle.
- `results/submission_validation.json`: the data/reference audit.
- `results/figures/`: raster previews and figure provenance records.
- `paper/`: manuscript, bibliography, TMLR style, vector figures, and `generated/`
  tables and numerical macros.
- `archive/`, `results/*/history/`, and `*_pre_revision.json`: earlier drafts or
  exploratory results retained for provenance, **not current evidence** for the
  manuscript. Partial runs are also excluded from publication export.

## Reproduce the checked results

Run these commands from the repository root. Install uv and synchronize the pinned
environment first. The lockfile specifies package requirements; numerical
experiments use JAX float64.

```bash
uv sync --locked

# Implementation checks; these do not substitute for theorem proofs.
PYTHONPATH=. uv run --frozen python -m unittest discover -s tests -p 'test_*.py'

# Retain the published arrays and refresh analytical controls.
PYTHONPATH=. uv run --frozen python experiments/stage1_protocol_and_counterexamples.py --checks-only

# Check data identity and the unprojected DNLS reference solver.
PYTHONPATH=. uv run --frozen python experiments/validate_submission.py

# Full five-seed latent experiments; matching saved runs resume by default.
PYTHONPATH=. uv run --frozen python experiments/stage2_foliation_and_obstructions.py

# Full five-seed DNLS training and evaluation.
PYTHONPATH=. uv run --frozen python experiments/stage3_dnls_benchmark.py

# Repeat warm inference timing after other CPU-intensive work has finished.
PYTHONPATH=. uv run --frozen python experiments/stage3_dnls_benchmark.py --timing-only

# Generate figures and tables from the complete, revised aggregate results.
PYTHONPATH=. uv run --frozen python experiments/stage4_generate_publication_figures.py
python3 experiments/export_paper_tables.py
python3 experiments/export_paper_tables.py --check
```

Stage 1 without `--checks-only` regenerates the datasets and manifest. Use it only
when deliberately creating a new protocol; new data require new validation and
downstream results. The manifest and validation report check distinct initial
states, not disjoint complete invariant level sets. The symmetric double-well
train/test splits share some energy values across opposite wells; the validation
report also checks energy together with the well label.

For partial Stage 2 runs, use `--suite lv`, `--suite double_well`, or `--suite torus`
(multiple suites are accepted), with optional `--seeds` and `--models`.
`--no-resume` retrains selected configurations. `--steps-multiplier` is for smoke
checks, not publication results. Restricted or shortened runs write
`stage2_partial_results.json`; only a complete five-seed run with the full schedule
writes publication-ready `stage2_results.json`. See `--help` for the current CLI.
Checkpoint fingerprints include configuration, source, and split hashes.

Inference timing reports the median of synchronized, warmed calls per seed,
followed by mean and **sample** SD across seeds. Each DNLS timed call forecasts the
whole 12-trajectory test batch for 249 steps; compilation is excluded. Hardware
and library versions are recorded in the JSON. Timings are machine dependent and
should be collected without concurrent training jobs.

## Read the quantities

Uncertainty is sample SD (`ddof=1`) across seeds 0--4 on one fixed data split. It
does not measure variability across newly sampled datasets. The exporter checks
seed records against stored means and sample SDs, preserves negative held-out
$R^2$, and displays undefined values as `--`.

Scalar calibration fits **invariant/energy to mean latent radius** on training
trajectories and scores that fit on test trajectories. This is not inverse
invariant recovery. The torus readout instead predicts two invariants from one
total radius or two block radii using a training-fitted degree-two ridge model.
The legacy `rollout_mse_20` field in Stage 2 describes 12 coarse prediction steps:
physical horizon 1.8 for Lotka--Volterra and 2.88 for the coupled torus. Table
captions use these actual horizons.

The projected-invariant baseline is COMET-inspired; its implemented loss is
recorded in its configuration and differs from the original COMET objective.
Its scalar has an arbitrary offset, so its CV is omitted from the radius-CV
comparison. The double-well reflection-quotient control reconstructs and predicts
only up to reflection. It is omitted for the tilted potential, which lacks that
symmetry. Winding numbers of sampled, polygon-closed trajectories are diagnostics,
not certificates of complete invariant levels.

## Build the paper and prepare an anonymous submission

Install Tectonic separately, then compile after regenerating figures and tables:

```bash
cd paper
tectonic -X compile main.tex
```

Generated tables contain only `tabular` environments; `main.tex` supplies captions
and layout. Each generated file records hashes of its input JSON, split manifest,
and exporter. `summary_values.tex` defines `\Result{dataset}{model}{field}` for
individual numbers and raises a LaTeX error for unknown keys. Its generated
definitions are the key index. `--check` fails if an input has changed or an
output is missing/stale.

Use the anonymous TMLR submission configuration in `main.tex`. Before sharing a
bundle, inspect the rendered PDF, source metadata, repository history, and
supplemental files for identifying information. Share the current manuscript,
code, locked data, and supporting run records through an anonymous location; omit
historical drafts. These scripts do not upload or submit the paper.
