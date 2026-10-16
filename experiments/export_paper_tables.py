"""Export checked experiment summaries to deterministic, provenance-stamped LaTeX.

This script uses only the Python standard library and never trains a model or
modifies result/data files. It rejects incomplete or old-protocol Stage 2 output.
Run with --check to verify that the checked-in tables match their current inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUTS = {
    "stage1": "results/stage1/stage1_results.json",
    "stage2": "results/stage2/stage2_results.json",
    "stage3": "results/stage3/stage3_dnls_results.json",
    "validation": "results/submission_validation.json",
    "manifest": "data/splits/splits_manifest.json",
}
SEEDS = [0, 1, 2, 3, 4]
DNLS_MODELS = {
    "Geodesic SkipNorm": ("gsn", "Geodesic SkipNorm"),
    "Projected Residual SkipNorm": ("projected", "Projected residual"),
    "State-Dependent SO(32)": ("cayley", "State-dependent Cayley"),
    "Linear Orthogonal SO(32)": ("linear", "Linear orthogonal"),
    "Pre-RMSNorm Residual": ("pre_rms", "Pre-RMSNorm residual"),
    "Post-RMSNorm Residual": ("post_rms", "Post-RMSNorm residual"),
    "Unconstrained ResNet": ("unconstrained", "Unconstrained ResNet"),
}
LV_MODELS = {
    "Geodesic SkipNorm (Bijective)": ("gsn_bijective", "GSN: bijective, radial"),
    "Geodesic SkipNorm (MLP)": ("gsn_mlp", "GSN: MLP, radial"),
    "Projected Residual SkipNorm": ("projected", "Projected residual"),
    "State-Dependent SO(2)": ("rotation", r"State-dependent $\mathrm{SO}(2)$"),
    "Projected-Invariant Baseline (COMET-inspired)": ("invariant", "Projected invariant"),
    "Unconstrained AE": ("unconstrained", "Unconstrained AE"),
}
DW_MODELS = {
    "Geodesic SkipNorm (Origin-Pinned Bijective)": ("pinned", "Origin-pinned bijective"),
    "Geodesic SkipNorm (Reflection Quotient Control)": ("quotient", "Reflection quotient"),
    "Geodesic SkipNorm (Free-Shift Bijective)": ("free", "Free-shift bijective"),
}
TORUS_MODELS = {
    "2-Block Toroidal Geodesic SkipNorm (K=2)": ("block_radial", "Two blocks (RF)"),
    "2-Block Geodesic SkipNorm (General Vector Field)": ("block_general", "Two blocks (VF)"),
    "Single-Sphere Geodesic SkipNorm (K=1)": ("sphere", "Single sphere"),
    "Unconstrained AE (d_z=4)": ("unconstrained", "Unconstrained AE"),
}
REGIMES = {
    "balanced_50_50": ("dw_balanced", "Symmetric, balanced training"),
    "imbalanced_80_20": ("dw_imbalanced", "Symmetric, imbalanced training"),
    "tilted_gamma_008": ("dw_tilted", r"Tilted, $\gamma=0.08$"),
}
DATASETS = {
    "lotka_volterra": "Lotka--Volterra",
    "double_well_balanced": "Double well, balanced",
    "double_well_imbalanced": "Double well, imbalanced",
    "double_well_tilted": "Double well, tilted",
    "coupled_torus_4d": "Coupled torus",
    "dnls_1d": "DNLS",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same_number(actual, expected, location: str) -> None:
    if expected is None:
        require(actual is None, f"{location}: expected null")
    else:
        require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-20),
                f"{location}: stored {actual!r}, recomputed {expected!r}")


def validate_summary(summary: dict, location: str) -> None:
    records = summary["seeds"]
    require(sorted(r["seed"] for r in records) == SEEDS, f"{location}: require exactly seeds 0--4")
    for name, stored in summary.items():
        if not name.endswith("_mean"):
            continue
        metric = name[:-5]
        require(all(metric in r for r in records), f"{location}/{metric}: missing seed metric")
        values = [r[metric] for r in records if r[metric] is not None]
        require(all(finite(v) for v in values), f"{location}/{metric}: non-finite metric")
        same_number(stored, statistics.mean(values) if values else None, f"{location}/{name}")
        require(metric + "_std" in summary, f"{location}/{metric}: missing sample SD")
        same_number(summary[metric + "_std"], statistics.stdev(values) if len(values) > 1 else None,
                    f"{location}/{metric}_std (ddof=1)")
        if metric + "_valid_seeds" in summary:
            require(summary[metric + "_valid_seeds"] == len(values), f"{location}/{metric}: valid count")


def number(value, digits: int = 3) -> str:
    """TeX number with untruncated sign; null represents a guarded/undefined result."""
    if value is None:
        return r"\text{--}"
    require(finite(value), f"Cannot format non-finite value {value!r}")
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e4:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    rendered = f"{value:.{digits}g}"
    if "e" in rendered:
        mantissa, exponent = rendered.split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return rendered


def estimate(summary: dict, metric: str, scale: float = 1.0) -> str:
    mean, sd = summary[metric + "_mean"], summary[metric + "_std"]
    if mean is None:
        return "--"
    if sd is not None and mean != 0 and "r2" not in metric and "rho" not in metric:
        magnitude = abs(mean * scale)
        if magnitude < 1e-2 or magnitude >= 1e4:
            exponent = math.floor(math.log10(magnitude))
            factor = 10.0 ** exponent
            center, spread = number(mean * scale / factor), number(sd * scale / factor, 2)
            return rf"$({center} \pm {spread})\times 10^{{{exponent}}}$"
    mean_text = f"{mean * scale:.6f}" if "r2" in metric or "rho" in metric else number(mean * scale)
    if sd is None:
        return f"${mean_text}$"
    return rf"${mean_text} \pm {number(sd * scale, 2)}$"


def table(headers: list[str], rows: list[list[str] | str]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def model_rows(section: dict, expected: dict, location: str):
    require(set(section) == set(expected), f"{location}: unexpected/missing models: {set(section) ^ set(expected)}")
    for label, (key, short) in expected.items():
        summary = section[label]
        validate_summary(summary, f"{location}/{label}")
        yield key, short, summary


def build(root: Path) -> dict[str, str]:
    raw = {key: (root / name).read_bytes() for key, name in INPUTS.items()}
    payload = {key: json.loads(raw[key],
                              parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-finite JSON: {x}")))
               for key, name in INPUTS.items()}
    provenance = [f"% Source SHA-256: {name} {hashlib.sha256(raw[key]).hexdigest()}" for key, name in INPUTS.items()]
    provenance.append(f"% Exporter SHA-256: {sha256(Path(__file__))}")
    preamble = "\n".join(["% Generated by experiments/export_paper_tables.py; do not edit.", *provenance]) + "\n"
    s1, s2, s3, validation, manifest = (payload[k] for k in INPUTS)
    require(s1["splits_manifest"] == manifest, "Stage 1 embeds a different split manifest")
    meta = s2.get("metadata", {})
    require(meta.get("complete_suite") is True, "Stage 2 is incomplete or uses the old protocol; wait for the complete revised run")
    require(meta.get("protocol_version") == "stage2-v2-heldout-calibration", "Unrecognized Stage 2 protocol")
    require(meta.get("seed_sd_ddof") == 1 and meta.get("seeds") == SEEDS, "Stage 2 seed/SD metadata")
    require(meta.get("steps_multiplier") == 1, "Stage 2 must use the full training schedule")
    require(s3["config"]["seeds"] == SEEDS, "DNLS requires five seeds")
    require(s3["config"]["split_sha256"] == manifest["dnls_1d"]["sha256"], "DNLS split provenance mismatch")
    require(s3["evaluation"]["prediction_steps"] == 249, "DNLS timing must cover 249 steps")
    require(s3["evaluation"]["test_trajectories"] == 12, "DNLS timing must cover the 12-trajectory batch")
    require(s3["evaluation"].get("timing_only_pass") is True,
            "Run the DNLS --timing-only pass after training before publication export")
    require(validation["dnls_reference"]["post_step_projection"] is False, "Reference was projected")
    require(validation["source_sha256"] == sha256(root / "experiments/validate_submission.py"),
            "Submission validation is stale relative to its source")
    values: dict[tuple[str, str, str], float | int | None] = {}

    def remember(dataset, model, summary):
        for field, value in summary.items():
            if value is None or finite(value):
                values[(dataset, model, field)] = value

    dnls_rows = []
    for key, short, summary in model_rows(s3["summary"], DNLS_MODELS, "dnls"):
        counts = [r["n_parameters"] for r in summary["seeds"]]
        require(all(n == summary["n_parameters"] for n in counts), f"dnls/{key}: parameter count mismatch")
        for record in summary["seeds"]:
            samples = record["timing_samples_ms"]
            require(len(samples) >= 2 and all(finite(t) and t > 0 for t in samples), f"dnls/{key}: timing samples")
            same_number(record["rollout_ms"], statistics.median(samples), f"dnls/{key}: timing median")
        worst = max(r["max_rel_norm_drift"] for r in summary["seeds"])
        remember("dnls", key, summary)
        values[("dnls", key, "worst_power_drift")] = worst
        dnls_rows.append([short, f"{summary['n_parameters']:,}", estimate(summary, "mse_20"),
                          estimate(summary, "mse_50"), estimate(summary, "mse_200"),
                          f"${number(worst)}$", estimate(summary, "rollout_ms")])

    lv_rows = []
    for key, short, summary in model_rows(s2["lotka_volterra"]["summary"], LV_MODELS, "lv"):
        remember("lv", key, summary)
        cv = "--" if key == "invariant" else estimate(summary, "mean_cv")
        lv_rows.append([short, cv, estimate(summary, "ovr"), estimate(summary, "isotonic_r2"),
                        estimate(summary, "spearman_abs_rho"), estimate(summary, "rollout_mse_20")])
    require(s2["lotka_volterra"]["evaluation"]["physical_horizon"] == 1.8, "LV rollout horizon changed")

    dw_rows = []
    require(set(s2["double_well"]["regimes"]) == set(REGIMES), "Double-well regimes changed")
    for regime, (dataset, label) in REGIMES.items():
        if dw_rows:
            dw_rows.append(r"\midrule")
        dw_rows.append(r"\multicolumn{7}{l}{\emph{" + label + r"}} \\")
        models = {label: item for label, item in DW_MODELS.items()
                  if regime != "tilted_gamma_008" or item[0] != "quotient"}
        for key, short, summary in model_rows(s2["double_well"]["regimes"][regime], models, dataset):
            remember(dataset, key, summary)
            dw_rows.append([short, estimate(summary, "mean_cv"), estimate(summary, "ovr"),
                            estimate(summary, "global_isotonic_r2"), estimate(summary, "within_well_isotonic_r2"),
                            estimate(summary, "both_wells_enclose_origin_rate", 100),
                            estimate(summary, "radius_well_separability_acc", 100)])

    torus_rows = []
    for key, short, summary in model_rows(s2["coupled_torus_4d"]["summary"], TORUS_MODELS, "torus"):
        remember("torus", key, summary)
        torus_rows.append([short, estimate(summary, "total_radius_cv"), estimate(summary, "block_radius_cv"),
                           estimate(summary, "block_ovr"), estimate(summary, "readout_r2_from_1_scalar"),
                           estimate(summary, "readout_r2_from_2_blocks"), estimate(summary, "rollout_mse_20")])
    require(s2["coupled_torus_4d"]["evaluation"]["physical_horizon"] == 2.88, "Torus rollout horizon changed")

    protocol_rows = []
    for dataset, label in DATASETS.items():
        entry = manifest[dataset]
        require(sha256(root / entry["file"]) == entry["sha256"], f"{dataset}: locked data hash mismatch")
        checked = validation["datasets"][dataset]
        require(checked["sha256"] == entry["sha256"], f"{dataset}: validation uses different data")
        require(all(c["exact_initial_state_duplicates"] == 0 for c in checked["split_checks"].values()),
                f"{dataset}: duplicate initial states across splits")
        train, val, test = (checked["shapes"][k] for k in ("train", "validation", "test"))
        require(train == entry["shapes"]["train_orbits"] and val == entry["shapes"]["val_orbits"],
                f"{dataset}: split shape mismatch")
        require(test == entry["shapes"]["test_orbits_long" if dataset == "dnls_1d" else "test_orbits"],
                f"{dataset}: test shape mismatch")
        dt = entry["meta"]["dt"]
        protocol_rows.append([label, str(train[0]), str(val[0]), str(test[0]),
                              f"{train[1]}/{test[1]}", f"${number(dt)}$", f"${number((test[1]-1)*dt)}$"])
        for field, value in {"n_train": train[0], "n_val": val[0], "n_test": test[0],
                             "states_train": train[1], "states_test": test[1], "dt": dt,
                             "test_horizon": (test[1]-1)*dt}.items():
            values[("protocol", dataset, field)] = value
    for field, value in validation["dnls_reference"].items():
        if finite(value):
            values[("reference", "dnls", field)] = value

    macros = [r"\makeatletter", r"\providecommand{\Result}[3]{%",
              r"  \ifcsname paperresult@#1@#2@#3\endcsname",
              r"    \ensuremath{\csname paperresult@#1@#2@#3\endcsname}%",
              r"  \else",
              r"    \PackageError{paperresults}{Unknown result #1/#2/#3}{Regenerate tables and use a documented result key.}%",
              r"  \fi}"]
    for (dataset, model, field), value in sorted(values.items()):
        formatted = number(value, 7 if "r2" in field or "rho" in field else 4)
        macros.append(rf"\expandafter\def\csname paperresult@{dataset}@{model}@{field}\endcsname{{{formatted}}}")
    macros += [r"\makeatother", ""]
    tables = {
        "table_dnls.tex": "% Drift is the worst relative POWER drift across seeds, test trajectories, and 249 steps.\n"
                          "% Timing is mean +/- sample SD of per-seed median warm calls, batch=12, steps=249.\n" +
                          table(["Model", "Params.", r"MSE$_{20}$", r"MSE$_{50}$", r"MSE$_{200}$", "Worst drift", "Time (ms)"], dnls_rows),
        "table_lv.tex": "% Calibration predicts mean radius from the invariant, fitted on training trajectories.\n"
                        "% Projected-invariant CV is omitted: its scalar offset makes it incomparable to encoder radius CV.\n"
                        "% Rollout: 12 coarse steps, stride=3, physical horizon=1.8; legacy JSON field ends in _20.\n" +
                        table(["Model", "CV", "OVR", r"Calib. $R^2$", r"$|\rho|$", "Rollout MSE"], lv_rows),
        "table_double_well.tex": "% Calibration predicts mean radius from energy; each fit uses training trajectories only.\n"
                                 "% Both is the across-seed rate of both well families having a sampled polygon enclosing zero.\n"
                                 "% Well probe is held-out accuracy of a training-fitted radius threshold/orientation.\n"
                                 "% Reflection quotient is deliberately absent for the tilted potential.\n" +
                                 table(["Encoder", "CV", "OVR", r"Global $R^2$", r"Within $R^2$", r"Both (\%)", r"Probe (\%)"], dw_rows),
        "table_torus.tex": "% Readouts predict the two invariants from one total radius or two planar block radii.\n"
                           "% Degree-two ridge uses training standardization/fit; negative held-out R2 is retained.\n"
                           "% Rollout: 12 coarse steps, stride=3, physical horizon=2.88; legacy JSON field ends in _20.\n" +
                           table(["Model", "Total CV", "Block CV", "Block OVR", r"$R^2_1$", r"$R^2_2$", "Rollout MSE"], torus_rows),
        "table_splits.tex": "% Counts are trajectories, states are train/test counts, and test horizon=(states_test-1)*dt.\n"
                              "% Distinct initial states do not establish disjoint full invariant level sets.\n" +
                              table(["Dataset", "Train", "Val.", "Test", "States (tr./te.)", r"$\Delta t$", r"$T_{\rm test}$"], protocol_rows),
        "summary_values.tex": "% Usage: \\Result{dataset}{model}{field}; each macro is one scalar, not mean +/- SD.\n" + "\n".join(macros),
    }
    return {name: preamble + body for name, body in tables.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root; defaults to this script's parent repository")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to ROOT/paper/generated")
    parser.add_argument("--check", action="store_true", help="Verify existing outputs without writing")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / "paper/generated"
    try:
        generated = build(root)
        if args.check:
            stale = [name for name, text in generated.items()
                     if not (output / name).exists() or (output / name).read_text() != text]
            require(not stale, "Missing/stale generated files: " + ", ".join(stale))
            print(f"Verified {len(generated)} deterministic LaTeX outputs.")
        else:
            output.mkdir(parents=True, exist_ok=True)
            for name, text in generated.items():
                temporary = output / (name + ".tmp")
                temporary.write_text(text)
                temporary.replace(output / name)
            print(f"Wrote {len(generated)} LaTeX outputs to {output}.")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(1, f"Table export refused: {exc}\n")


if __name__ == "__main__":
    main()
