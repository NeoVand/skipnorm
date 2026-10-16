"""Aggregate a locked visual-world-model run without making research claims.

Usage:
  python experiments/stage8_visual_report.py --run results/.../pilot_v1 \
      --output /absolute/path/to/report

Incomplete runs fail by default. --allow-partial is a development aid and marks
figures and summary as partial. No test metric selects a model or a displayed trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.visual_physics_data import image_positions


LABELS = {
    "residual": "Residual", "prenorm": "Pre-norm residual",
    "scale": "Pre-norm + scale", "projected": "Projected SkipNorm",
    "geodesic": "Geodesic SkipNorm", "rotation": "Invertible rotation",
    "linear_orthogonal": "Linear orthogonal", "hamiltonian": "Hamiltonian",
}
COLORS = {
    "residual": "#d55e00", "prenorm": "#0072b2", "scale": "#56b4e9",
    "projected": "#009e73", "geodesic": "#006347", "rotation": "#9b4f96",
    "linear_orthogonal": "#c99600", "hamiltonian": "#555555",
}
BASELINES = {"residual", "prenorm", "scale", "linear_orthogonal", "hamiltonian"}
SKIP_STYLES = {"geodesic": "-", "rotation": "-", "projected": "-"}


def save_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    # Invalid values must be visible, never silently excluded from a mean.
    return {
        "mean": float(np.mean(values)) if finite.all() else None,
        "sd": float(np.std(values, ddof=1)) if finite.all() and len(values) > 1 else None,
        "n": len(values), "finite_n": int(finite.sum()),
    }


def load_run(run, allow_partial):
    manifest = json.loads((run / "manifest.json").read_text())
    expected = [(kind, seed) for kind in manifest["planned_kinds"] for seed in manifest["planned_seeds"]]
    records, missing = {}, []
    for kind, seed in expected:
        stem = run / f"{kind}_seed{seed}"
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".npz").exists():
            missing.append({"kind": kind, "seed": seed})
            continue
        record = json.loads(stem.with_suffix(".json").read_text())
        if record["kind"] != kind or record["seed"] != seed:
            raise ValueError(f"Result identity mismatch: {stem}")
        config = {key: value for key, value in record["config"].items() if key != "kind"}
        if config != manifest["config"]:
            raise ValueError(f"Result configuration mismatch: {stem}")
        # A follow-up may train at a horizon absent from the original metric list.
        # Saved traces are RMS over the same trajectories and coordinates, so
        # RMS of their first H values recovers the cumulative H-step metric.
        horizon = config["horizon"]
        with np.load(stem.with_suffix(".npz")) as data:
            for split in ("test","ood"):
                metric = f"position_nrmse_{horizon}"
                if metric not in record[split]:
                    trace = data[split + "_position_nrmse_trace"]
                    if horizon > len(trace):
                        raise ValueError("Training horizon exceeds evaluation horizon.")
                    record[split][metric] = float(np.sqrt(np.mean(trace[:horizon]**2)))
        records[(kind, seed)] = record
    if missing and not allow_partial:
        raise RuntimeError(f"Incomplete run: {len(records)}/{len(expected)} complete. Missing {missing}")
    if not records:
        raise RuntimeError("No completed model results.")
    return manifest, records, missing


def aggregate(records):
    per_seed, groups, details = [], {}, []
    for (kind, seed), record in records.items():
        details.append({"kind": kind, "seed": seed, **record["parameters"], **record["training"]})
        for split in ("test", "ood"):
            for metric, value in record[split].items():
                if not isinstance(value, (int, float)):
                    continue
                per_seed.append({"kind": kind, "seed": seed, "split": split, "metric": metric, "value": value})
                groups.setdefault((kind, split, metric), []).append(value)
    means = [{"kind": key[0], "split": key[1], "metric": key[2], **mean_sd(values)}
             for key, values in groups.items()]
    return per_seed, means, details


def _configure_plot():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 130, "savefig.dpi": 220,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def _save_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def equilibrium_reference(run, manifest, records, data_path):
    """Post-hoc zero-position reference on exactly the selected trajectories."""
    digest = hashlib.sha256()
    with data_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["config"]["data_sha256"]:
        raise ValueError("Reference data does not match the locked run's dataset hash.")
    with np.load(data_path) as data:
        scale = np.std(data["train_states"][..., :4], axis=(0, 1))
    kind, seed = next(iter(records))
    result = {"label": "Equilibrium (post hoc)",
        "status": "Post-hoc diagnostic reference, not a prespecified model or control.",
        "prediction": "q=0 at every forecast step; simulator states used only to score it.",
        "normalization_scale": scale.tolist(), "splits": {}}
    with np.load(run / f"{kind}_seed{seed}.npz") as data:
        for split in ("test", "ood"):
            truth = data[split + "_truth_q"].astype(np.float64)
            if truth.shape[0] != manifest["config"]["eval_count"]:
                raise ValueError("Saved truth does not contain the expected evaluation trajectory count.")
            error = truth / scale
            metrics = {"position_nrmse": float(np.sqrt(np.mean(error**2))),
                "position_rmse": float(np.sqrt(np.mean(truth**2))),
                "position_nrmse_trace": np.sqrt(np.mean(error**2, axis=(0,2))).tolist(),
                "n_trajectories": truth.shape[0], "forecast_steps": truth.shape[1]}
            for horizon in sorted({8,20,50,100,197,manifest["config"]["horizon"]}):
                if horizon <= truth.shape[1]:
                    metrics[f"position_nrmse_{horizon}"] = float(np.sqrt(np.mean(error[:,:horizon]**2)))
            result["splits"][split] = metrics
    return result


def training_mean_image_reference(data_path, manifest, equilibrium):
    """Predict the average training image forever; use no physical prior."""
    history = manifest["config"]["history"]
    n = manifest["config"]["eval_count"]
    scale = np.asarray(equilibrium["normalization_scale"])
    with np.load(data_path) as data:
        mean_image = np.mean(data["train_images"], axis=(0,1))
        q = image_positions(mean_image)
        result = {"label": "Mean training image (post hoc)",
            "status": "Post-hoc diagnostic reference, not a prespecified model or control.",
            "prediction": "Pixelwise mean over all training frames, repeated forever; no state supervision or physical prior.",
            "normalization_scale": scale.tolist(), "decoded_constant_position": q.tolist(), "splits": {}}
        for split in ("test", "ood"):
            target = data[split + "_images"][:n,history:]
            truth = data[split + "_states"][:n,history:,:4].astype(np.float64)
            error = (q - truth) / scale
            squared_pixel_error = (mean_image - target)**2
            foreground = np.clip(target,0.,1.)
            background = 1. - foreground
            fg_loss = np.sum(foreground * squared_pixel_error,axis=(-2,-1)) / np.maximum(foreground.sum(axis=(-2,-1)),1e-6)
            bg_loss = np.sum(background * squared_pixel_error,axis=(-2,-1)) / np.maximum(background.sum(axis=(-2,-1)),1e-6)
            metrics = {"position_nrmse": float(np.sqrt(np.mean(error**2))),
                "position_rmse": float(np.sqrt(np.mean((q-truth)**2))),
                "pixel_mse": float(squared_pixel_error.mean()),
                "balanced_pixel_loss": float(np.mean(.5 * fg_loss + .5 * bg_loss)),
                "position_nrmse_trace": np.sqrt(np.mean(error**2,axis=(0,2))).tolist(),
                "n_trajectories": truth.shape[0], "forecast_steps": truth.shape[1]}
            for horizon in sorted({8,20,50,100,197,manifest["config"]["horizon"]}):
                if horizon <= truth.shape[1]:
                    metrics[f"position_nrmse_{horizon}"] = float(np.sqrt(np.mean(error[:,:horizon]**2)))
                    metrics[f"balanced_pixel_loss_{horizon}"] = float(np.mean(.5 * fg_loss[:,:horizon] + .5 * bg_loss[:,:horizon]))
            result["splits"][split] = metrics
    return result


def plot_forecasts(run, output, manifest, records, partial, static_reference):
    _configure_plot()
    kinds = [k for k in manifest["planned_kinds"] if any(key[0] == k for key in records)]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.5), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    dt = manifest["config"]["dt"]
    for ax, split, title in zip(axes[:2], ("test", "ood"), ("A  Familiar amplitudes", "B  Higher amplitudes")):
        for kind in kinds:
            traces = []
            for k, seed in records:
                if k == kind:
                    with np.load(run / f"{kind}_seed{seed}.npz") as data:
                        traces.append(data[split + "_position_nrmse_trace"])
            traces = np.array(traces)
            mean = traces.mean(axis=0)
            t = dt * np.arange(1, len(mean) + 1)
            ax.plot(t, mean, color=COLORS[kind], linestyle=SKIP_STYLES.get(kind, "--"),
                    linewidth=2.0 if kind in SKIP_STYLES else 1.5)
            if len(traces) > 1:
                sd = traces.std(axis=0, ddof=1)
                ax.fill_between(t, np.maximum(mean - sd, 1e-5), mean + sd, color=COLORS[kind], alpha=.10, linewidth=0)
        reference = static_reference["splits"][split]["position_nrmse_trace"]
        ax.plot(dt * np.arange(1,len(reference)+1), reference, color="#111111", linestyle=":", linewidth=2.1)
        ax.axvline(dt * manifest["config"]["horizon"], color="#777777", linestyle=":", linewidth=1)
        ax.text(dt * manifest["config"]["horizon"] + .12, .98, "Training horizon", transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="left", fontsize=8, color="#666666")
        ax.set(title=title, xlabel="Time since final observation (s)", ylabel="Position NRMSE per forecast step")
        ax.set_yscale("log")
        ax.grid(alpha=.17)
    ax = axes[2]
    x = np.arange(len(kinds) + 1)
    trained_horizon = manifest["config"]["horizon"]
    forecast_horizon = static_reference["splits"]["test"]["forecast_steps"]
    for offset, horizon, opacity in ((-.19, trained_horizon, .38), (.19, forecast_horizon, 1.)):
        metric = f"position_nrmse_{horizon}" if horizon == trained_horizon else "position_nrmse"
        for index, kind in enumerate(kinds):
            values = [r["test"][metric] for (k, _), r in records.items() if k == kind]
            stats = mean_sd(values)
            ax.bar(index + offset, stats["mean"], width=.35, color=COLORS[kind], alpha=opacity,
                   yerr=stats["sd"], capsize=2, error_kw={"linewidth": 1})
        ax.bar(len(kinds) + offset, static_reference["splits"]["test"][metric],
               width=.35, color="#111111", alpha=opacity)
    ax.set_xticks(x, [LABELS[k] for k in kinds] + [static_reference["label"]], rotation=48, ha="right", fontsize=9)
    ax.set(title="C  Cumulative error, familiar amplitudes", ylabel="Position NRMSE over forecast interval")
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#666666", alpha=.38, label=f"First {trained_horizon} steps"),
                       plt.Rectangle((0, 0), 1, 1, color="#666666", label=f"All {forecast_horizon} steps")], frameon=False, fontsize=9)
    handles = [Line2D([0], [0], color=COLORS[k], label=LABELS[k], linestyle=SKIP_STYLES.get(k, "--")) for k in kinds]
    handles.append(Line2D([0],[0],color="#111111",linestyle=":",label=static_reference["label"]))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.055))
    count = len(records)
    planned = len(manifest["planned_seeds"])*len(manifest["planned_kinds"])
    seed_count = len(manifest["planned_seeds"])
    tag = f"PARTIAL {count}/{planned}" if partial else f"{seed_count} initialization seed" + ("s" if seed_count != 1 else "")
    statistic = "mean ± sample SD" if seed_count > 1 else "single seed; no uncertainty estimate"
    fig.suptitle(f"Visual physics pilot | {tag} | {statistic}", fontsize=14, y=1.02)
    fig.subplots_adjust(bottom=.27, wspace=.30)
    _save_figure(fig, output, "forecast_comparison")


def plot_diagnostics(output, manifest, records, partial):
    _configure_plot()
    kinds = [k for k in manifest["planned_kinds"] if any(key[0] == k for key in records)]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.7))
    for kind in kinds:
        own = [r for (k, _), r in records.items() if k == kind]
        for record in own:
            updates = [entry["update"] for entry in record["curve"]]
            loss = [entry["validation_loss"] for entry in record["curve"]]
            axes[0].plot(updates, loss, color=COLORS[kind], alpha=.55, linewidth=1.1)
    axes[0].set(title="A  Short-horizon validation image loss", xlabel="Joint training update", ylabel="Balanced pixel + reconstruction loss")
    axes[0].set_yscale("log")
    for axis, metric, title, ylabel in (
        (axes[1], "localized_object_fraction", "B  Localized objects during forecast", "Fraction of predicted objects"),
        (axes[2], "localized_1px_valid_horizon_mean", "C  Localized, ≤1-pixel RMS horizon", "Steps before first failure"),
    ):
        for index, kind in enumerate(kinds):
            for split, offset, opacity in (("test", -.19, 1.), ("ood", .19, .38)):
                stats = mean_sd([r[split][metric] for (k, _), r in records.items() if k == kind])
                axis.bar(index + offset, stats["mean"], width=.35, yerr=stats["sd"], color=COLORS[kind], alpha=opacity,
                         capsize=2, error_kw={"linewidth": 1})
        axis.set_xticks(np.arange(len(kinds)), [LABELS[k] for k in kinds], rotation=48, ha="right", fontsize=9)
        axis.set(title=title, ylabel=ylabel)
        axis.set_axisbelow(True)
        axis.legend(handles=[plt.Rectangle((0,0),1,1,color="#666666", label="Familiar amplitudes"),
                             plt.Rectangle((0,0),1,1,color="#666666", alpha=.38,label="Higher amplitudes")], frameon=False,fontsize=9)
    for ax in axes:
        ax.grid(alpha=.16)
    fig.suptitle("Visual and training diagnostics" + (" — PARTIAL" if partial else ""), fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output, "visual_diagnostics")


def choose_baseline(records):
    scores = {kind: np.mean([r["training"]["best_validation_loss"] for (k, _), r in records.items() if k == kind])
              for kind in BASELINES if (kind, 0) in records}
    return min(scores, key=scores.get) if scores else None


def _font(size):
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def make_examples(run, output, manifest, records, partial):
    comparator = choose_baseline(records)
    required = ["geodesic", "rotation", comparator]
    selection = {
        "seed": 0, "trajectory_index": 0,
        "selection": "Fixed seed 0, trajectory 0; illustrative, not asserted typical.",
        "comparator": comparator,
        "comparator_selection": "Minimum mean best short-horizon ID validation loss among residual, prenorm, scale, linear_orthogonal, hamiltonian; no test selection.",
    }
    if any(kind is None or (kind, 0) not in records for kind in required):
        selection["status"] = "skipped: required examples are incomplete"
        return selection
    arrays = {}
    for kind in required:
        with np.load(run / f"{kind}_seed0.npz") as data:
            arrays[kind] = {key: data[key] for key in data.files if "example" in key or key.endswith(("predicted_q", "truth_q"))}
    dt = manifest["config"]["dt"]
    selected_split = "test"
    images = [arrays["geodesic"][selected_split + "_example_truth_images"][0]] + [arrays[k][selected_split + "_example_prediction_images"][0] for k in required]
    names = ["Ground truth"] + [LABELS[k] for k in required]
    width, image_height, margin, header, footer = 288, 144, 20, 86, 44
    canvas_width = 4 * width + 5 * margin
    frames = []
    frame_indices = list(range(0, len(images[0]), 2))
    if frame_indices[-1] != len(images[0]) - 1:
        frame_indices.append(len(images[0]) - 1)
    for t in frame_indices:
        canvas = Image.new("RGB", (canvas_width, header + image_height + footer), "#f6f7f8")
        draw = ImageDraw.Draw(canvas)
        draw.text((margin, 9), f"Forecast without observations · t = {(t+1)*dt:.2f} s", font=_font(21), fill="#222222")
        draw.text((margin, 36), "Familiar amplitudes · seed 0 / trajectory 0 · fixed illustration, not a typicality claim" + (" · PARTIAL" if partial else ""), font=_font(15), fill="#555555")
        for column, (name, sequence) in enumerate(zip(names, images)):
            x = margin + column * (width + margin)
            draw.text((x, 63), name, font=_font(16), fill="#222222")
            pixels = np.uint8(np.clip(sequence[t], 0, 1) * 255)
            tile = Image.fromarray(pixels).convert("RGB").resize((width, image_height), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, header))
        draw.text((margin, header + image_height + 13), "Baseline chosen by short-horizon validation loss. All panels use identical brightness and magnification.", font=_font(15), fill="#555555")
        frames.append(canvas)
    duration_ms = int(round(2 * dt * 1000))
    frames[0].save(output / "fixed_example.gif", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)
    frames[min(10,len(frames)-1)].save(output / "fixed_example_preview.png")
    _configure_plot()
    fig, axes = plt.subplots(4, 2, figsize=(12.2, 9), sharex=True, sharey=True)
    for col, split in enumerate(("test", "ood")):
        truth = arrays["geodesic"][split + "_truth_q"][0]
        times = dt * np.arange(1,len(truth)+1)
        for row in range(4):
            ax = axes[row,col]
            ax.plot(times, truth[:,row], color="#111111", label="Ground truth", linewidth=1.9)
            for kind in required:
                ax.plot(times, arrays[kind][split + "_predicted_q"][0,:,row], color=COLORS[kind], label=LABELS[kind], linewidth=1.4)
            ax.set_ylabel(f"Mass {row+1}: position")
            ax.grid(alpha=.16)
            if row == 0:
                ax.set_title("Familiar amplitudes" if split == "test" else "Higher amplitudes")
            if row == 3:
                ax.set_xlabel("Time since final observation (s)")
    axes[0,0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Fixed illustration: seed 0, trajectory 0 (not asserted typical)" + (" — PARTIAL" if partial else ""))
    fig.tight_layout()
    _save_figure(fig, output, "fixed_example_tracks")
    selection.update({"status": "generated", "gif_split": "test", "frame_stride": 2, "duration_ms": duration_ms,
                      "brightness": "Fixed [0,1], all panels", "resize": "6x nearest-neighbor"})
    return selection


def plot_horizon_comparison(other_run, output, manifest, records, mean_image):
    """Descriptive comparison; expose unequal replication and paired seed zero."""
    other_manifest, other_records, _ = load_run(other_run, False)
    for key in ("data_sha256", "history", "latent_dim", "eval_count"):
        if manifest["config"][key] != other_manifest["config"][key]:
            raise ValueError(f"Horizon comparison has incompatible {key}.")
    kinds = [kind for kind in manifest["planned_kinds"] if any(k == kind for k,_ in other_records)]
    h0, h1 = manifest["config"]["horizon"], other_manifest["config"]["horizon"]
    n0, n1 = len(manifest["planned_seeds"]), len(other_manifest["planned_seeds"])
    _configure_plot()
    fig, axes = plt.subplots(1,2,figsize=(13.2,5.6))
    comparisons = []
    for axis, metric, title, ylabel in (
        (axes[0],"position_nrmse_50","A  Position error over first 50 forecast steps","Position NRMSE (lower is better)"),
        (axes[1],"localized_1px_valid_horizon_mean","B  Localized, ≤1-pixel RMS horizon","Steps before first failure (higher is better)"),
    ):
        for index,kind in enumerate(kinds):
            for source, horizon, offset, opacity in ((records,h0,-.19,.38),(other_records,h1,.19,1.)):
                values = [record["test"][metric] for (k,_),record in source.items() if k == kind]
                stats = mean_sd(values)
                axis.bar(index+offset,stats["mean"],width=.35,color=COLORS[kind],alpha=opacity,
                         yerr=stats["sd"],capsize=2,error_kw={"linewidth":1})
                comparisons.append({"kind":kind,"trained_horizon":horizon,"metric":metric,**stats})
            if (kind,0) in records:
                axis.plot(index-.19,records[(kind,0)]["test"][metric],"o",color="black",markersize=3)
        axis.set_xticks(np.arange(len(kinds)),[LABELS[k] for k in kinds],rotation=42,ha="right",fontsize=9)
        axis.set(title=title,ylabel=ylabel)
        axis.grid(axis="y",alpha=.16)
        axis.set_axisbelow(True)
    axes[0].axhline(mean_image["splits"]["test"]["position_nrmse_50"],color="#111111",linestyle=":",linewidth=1.3)
    axes[0].text(.99,mean_image["splits"]["test"]["position_nrmse_50"],"Mean training image (post hoc)",
                 transform=axes[0].get_yaxis_transform(),ha="right",va="bottom",fontsize=8)
    fig.legend(handles=[plt.Rectangle((0,0),1,1,color="#666666",alpha=.38,label=f"{h0}-step training: {n0} seeds, mean ± sample SD"),
                        plt.Rectangle((0,0),1,1,color="#666666",label=f"{h1}-step training: {n1} seed"+("s" if n1 != 1 else "")),
                        Line2D([0],[0],marker="o",color="black",linestyle="none",markersize=4,label=f"{h0}-step training, seed 0")],
               loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.01),fontsize=9)
    fig.suptitle("Exploratory training-horizon follow-up · familiar amplitudes",fontsize=14)
    fig.text(.5,-.015,"Same test trajectories; unequal seed counts. This is a diagnostic comparison, not independent confirmation.",ha="center",fontsize=9,color="#555555")
    fig.subplots_adjust(bottom=.30,wspace=.28,top=.86)
    _save_figure(fig,output,"horizon_followup_comparison")
    result = {"main_horizon":h0,"followup_horizon":h1,"main_seeds":manifest["planned_seeds"],
        "followup_seeds":other_manifest["planned_seeds"],"followup_run":str(other_run),
        "scope":"Same test set; descriptive mean/sample SD and paired seed-zero marker. Follow-up is post-hoc and not independent confirmation.",
        "metrics":comparisons}
    (output / "horizon_followup_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/visual_physics/pilot_v1.npz")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--compare-run", type=Path, help="Optional completed horizon follow-up to compare descriptively.")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    manifest, records, missing = load_run(run, args.allow_partial)
    output.mkdir(parents=True, exist_ok=True)
    per_seed, means, details = aggregate(records)
    save_csv(output / "metrics_per_seed.csv", per_seed, ["kind", "seed", "split", "metric", "value"])
    save_csv(output / "metrics_mean_sd.csv", means, ["kind", "split", "metric", "mean", "sd", "n", "finite_n"])
    save_csv(output / "model_details.csv", details, list(details[0]))
    shutil.copy2(run / "manifest.json", output / "run_manifest.json")
    followup = manifest["config"]["horizon"] != 8
    if (run.parent / "PROTOCOL.md").exists():
        name = "ORIGINAL_PILOT_PROTOCOL.md" if followup else "PROTOCOL.md"
        shutil.copy2(run.parent / "PROTOCOL.md", output / name)
    if followup and (run.parent / "FOLLOWUP_H24.md").exists():
        shutil.copy2(run.parent / "FOLLOWUP_H24.md", output / "FOLLOWUP_H24.md")
    controls = json.loads((run / "controls.json").read_text())
    shutil.copy2(run / "controls.json", output / "controls.json")
    equilibrium = equilibrium_reference(run, manifest, records, args.data)
    (output / "posthoc_equilibrium_reference.json").write_text(json.dumps(equilibrium, indent=2) + "\n")
    mean_image = training_mean_image_reference(args.data, manifest, equilibrium)
    (output / "posthoc_mean_training_image.json").write_text(json.dumps(mean_image, indent=2) + "\n")
    reference_rows = []
    for name, reference in (("mean_training_image", mean_image),("equilibrium",equilibrium)):
        for split, metrics in reference["splits"].items():
            for metric,value in metrics.items():
                if isinstance(value,(int,float)):
                    reference_rows.append({"posthoc_control": name,"split":split,"metric":metric,"value":value})
    save_csv(output / "posthoc_controls.csv", reference_rows, ["posthoc_control","split","metric","value"])
    plot_forecasts(run, output, manifest, records, bool(missing), mean_image)
    plot_diagnostics(output, manifest, records, bool(missing))
    examples = make_examples(run, output, manifest, records, bool(missing))
    comparison = None
    if args.compare_run:
        if missing:
            raise ValueError("Horizon comparison requires a complete main run.")
        comparison = plot_horizon_comparison(args.compare_run.resolve(),output,manifest,records,mean_image)
    summary = {
        "run_directory": str(run), "complete": not missing,
        "completed_models": len(records), "missing": missing,
        "aggregation": "Unweighted mean and sample SD across initialization seeds; descriptive, not significance claims. Invalid values are not dropped.",
        "normalization": "Each position divided by its fixed training-set standard deviation; OOD uses the same scale.",
        "cumulative_metrics": "position_nrmse_K aggregates forecast steps 1 through K, not error at step K.",
        "scope": "No future observations after three input frames; no literal occluder or appearance variation. Fixed four-column rendering with known evaluation strips. All model predictions are learned from pixels only.",
        "trained_horizon": manifest["config"]["horizon"],
        "exploratory_followup": followup,
        "mean_sd": means, "controls": controls, "posthoc_equilibrium_reference": equilibrium, "examples": examples,
        "posthoc_mean_training_image": mean_image,
        "horizon_followup_comparison": comparison,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "summary_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": not missing, "completed": len(records), "output": str(output), "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
