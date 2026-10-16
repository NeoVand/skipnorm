# Diagnostic follow-up: longer training forecasts

Added after inspecting the first completed seed of the eight-step pilot. Those
models predict eight frames well but lose long-horizon phase; all are worse over
197 frames than the posthoc equilibrium-position control. This follow-up asks
whether weak temporal supervision, rather than the chosen transition alone,
explains the failure.

Change only the training/validation forecast horizon from 8 to 24 steps (0.64s to
1.92s). Retain the same dataset, 40-observation training clips, image-only loss,
visual architecture, latent dimension, 4,000-update shared visual warm-start,
8,000 joint updates, learning-rate schedule, batch size, selection procedure,
and all eight transition kinds. Use seed 0 for this diagnostic. Runtime increases;
this is not an equal-compute comparison between horizons. Within each horizon,
models use equal update and data budgets, with counts/timing reported.

The eight-step pilot remains the declared three-seed comparison. This one-seed
follow-up is exploratory and cannot establish an advantage or replace its results.
No selection or architecture change is based on OOD results. Neither experiment
is a held-out confirmation for publication.

Output: `results/stage8_visual_world_model/pilot_h24_seed0/`.
