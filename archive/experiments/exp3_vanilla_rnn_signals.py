"""Experiment 3: Unidirectional Vanilla RNN Signal Reconstruction Benchmark
(EEG Bursts, Polyphonic Piano Roll, and Speech Formant Spectrograms) from a 4D Latent Bottleneck.

Reproduces and dissects the phenomenon from the SkipNorm video where a simple unidirectional
Vanilla RNN with SkipNorm achieves 100%-300% improvements over standard RNNs and LayerNorm RNNs.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skipnorm import SkipNorm, skip_norm

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_signal_datasets(n_samples: int = 600, seq_len: int = 80, n_channels: int = 64, seed: int = 42):
    """Generate realistic multi-channel non-stationary sequences for EEG, Piano Roll, and Speech."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, seq_len, dtype=np.float32)

    # 1. Multi-channel EEG: spatial lead-field mixing of oscillatory alpha/beta/theta bursts
    n_sources = 4
    spatial_maps = rng.standard_normal((n_sources, n_channels)).astype(np.float32)
    spatial_maps /= np.linalg.norm(spatial_maps, axis=-1, keepdims=True)
    eeg_data = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    for i in range(n_samples):
        sources = np.zeros((seq_len, n_sources), dtype=np.float32)
        for s in range(n_sources):
            freq = rng.uniform(3.0, 18.0)
            phase = rng.uniform(0, 2 * np.pi)
            center = rng.uniform(0.15, 0.85)
            width = rng.uniform(0.05, 0.20)
            amp = np.exp(rng.uniform(-1.5, 1.2))  # wide dynamic range of burst amplitudes
            envelope = amp * np.exp(-0.5 * ((t - center) / width) ** 2)
            sources[:, s] = envelope * np.sin(2 * np.pi * freq * t + phase)
        eeg_data[i] = sources @ spatial_maps + 0.02 * rng.standard_normal((seq_len, n_channels)).astype(np.float32)

    # 2. Polyphonic Piano Roll: chord templates with ADSR velocity envelopes and silent rests
    n_chords = 8
    chord_templates = np.zeros((n_chords, n_channels), dtype=np.float32)
    for c in range(n_chords):
        root = rng.integers(5, 35)
        for interval in [0, 4, 7, 12, 16]:
            if root + interval < n_channels:
                chord_templates[c, root + interval] = rng.uniform(0.6, 1.0)
                if root + interval + 12 < n_channels:
                    chord_templates[c, root + interval + 12] = 0.4  # harmonic overtone

    piano_data = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    for i in range(n_samples):
        pos = 0
        while pos < seq_len:
            dur = int(rng.integers(8, 22))
            end = min(seq_len, pos + dur)
            L = end - pos
            if rng.random() > 0.2:  # 80% note, 20% rest/silence
                chord_idx = rng.integers(0, n_chords)
                vel = np.exp(rng.uniform(-1.2, 1.0))
                local_t = np.linspace(0, 1, L, dtype=np.float32)
                adsr = vel * (1.0 - np.exp(-15 * local_t)) * np.exp(-3.0 * local_t)
                piano_data[i, pos:end] = adsr[:, None] * chord_templates[chord_idx][None, :]
            pos = end

    # 3. Speech Formant Mel-Spectrogram: moving formant bands with syllable energy envelope
    freq_grid = np.linspace(0, 1, n_channels, dtype=np.float32)[None, :]
    speech_data = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    for i in range(n_samples):
        # 2 moving formants parameterized smoothly in time
        f1 = 0.15 + 0.12 * np.sin(2 * np.pi * rng.uniform(1.0, 3.0) * t + rng.uniform(0, 6.28))[:, None]
        f2 = 0.45 + 0.20 * np.cos(2 * np.pi * rng.uniform(1.0, 3.0) * t + rng.uniform(0, 6.28))[:, None]
        f3 = 0.78 + 0.10 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t + rng.uniform(0, 6.28))[:, None]
        spec_shape = (
            np.exp(-0.5 * ((freq_grid - f1) / 0.045) ** 2)
            + 0.8 * np.exp(-0.5 * ((freq_grid - f2) / 0.055) ** 2)
            + 0.4 * np.exp(-0.5 * ((freq_grid - f3) / 0.06) ** 2)
        )
        # Syllable amplitude modulation (voiced bursts + unvoiced/quiet dips)
        syl_freq = rng.uniform(2.5, 4.5)
        syl_env = np.maximum(0.02, np.sin(2 * np.pi * syl_freq * t + rng.uniform(0, 6.28))) ** 2
        global_gain = np.exp(rng.uniform(-1.0, 1.0))
        speech_data[i] = (global_gain * syl_env[:, None]) * spec_shape

    return {
        "EEG": torch.tensor(eeg_data),
        "PianoRoll": torch.tensor(piano_data),
        "Speech": torch.tensor(speech_data),
    }


class RNNSequenceAutoencoder(nn.Module):
    """Unidirectional Vanilla Elman RNN Sequence Autoencoder:
    64 -> 32 -> 4 (latent bottleneck sequence) -> 32 -> 64.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32, latent_dim: int = 4, mode: str = "vanilla", nonneg_out: bool = False):
        super().__init__()
        self.mode = mode
        self.nonneg_out = nonneg_out

        self.rnn_enc1 = nn.RNN(in_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.rnn_enc2 = nn.RNN(hidden_dim, latent_dim, batch_first=True, nonlinearity="tanh")
        self.lin_z = nn.Linear(latent_dim, latent_dim)

        self.rnn_dec1 = nn.RNN(latent_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.rnn_dec2 = nn.RNN(hidden_dim, in_dim, batch_first=True, nonlinearity="tanh")
        self.lin_out = nn.Linear(in_dim, in_dim)

        if mode == "orthogonal":
            for rnn in [self.rnn_enc1, self.rnn_enc2, self.rnn_dec1, self.rnn_dec2]:
                nn.init.orthogonal_(rnn.weight_hh_l0)
                nn.init.orthogonal_(rnn.weight_ih_l0)

        if mode == "layernorm":
            self.ln_e1 = nn.LayerNorm(hidden_dim)
            self.ln_e2 = nn.LayerNorm(latent_dim)
            self.ln_d1 = nn.LayerNorm(hidden_dim)
        elif mode == "rmsnorm":
            self.ln_e1 = nn.RMSNorm(hidden_dim)
            self.ln_e2 = nn.RMSNorm(latent_dim)
            self.ln_d1 = nn.RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode
        if self.mode == "skipnorm":
            # Normalize input per frame so RNN tanh operates in its linear/unsaturated regime,
            # then SkipNorm restores the exact frame energy ||x_t||_2 at every stage!
            x_unit = x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-8)
            h1, _ = self.rnn_enc1(x_unit)
            h1 = skip_norm(x, h1, dim=-1)

            h1_unit = h1 / (torch.linalg.vector_norm(h1, dim=-1, keepdim=True) + 1e-8)
            h2, _ = self.rnn_enc2(h1_unit)
            z_raw = self.lin_z(h2)
            z = skip_norm(h1, z_raw, dim=-1)

            # Decode strictly from 4D latent sequence z!
            z_unit = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-8)
            d1, _ = self.rnn_dec1(z_unit)
            d1 = skip_norm(z, d1, dim=-1)

            d1_unit = d1 / (torch.linalg.vector_norm(d1, dim=-1, keepdim=True) + 1e-8)
            d2, _ = self.rnn_dec2(d1_unit)
            out_raw = self.lin_out(d2)
            if self.nonneg_out:
                out_raw = F.relu(out_raw)
            x_hat = skip_norm(d1, out_raw, dim=-1)
            return x_hat, z

        else:
            h1, _ = self.rnn_enc1(x)
            if self.mode in ("layernorm", "rmsnorm"):
                h1 = self.ln_e1(h1)
            h2, _ = self.rnn_enc2(h1)
            z = self.lin_z(h2)
            if self.mode in ("layernorm", "rmsnorm"):
                z = self.ln_e2(z)

            d1, _ = self.rnn_dec1(z)
            if self.mode in ("layernorm", "rmsnorm"):
                d1 = self.ln_d1(d1)
            d2, _ = self.rnn_dec2(d1)
            out_raw = self.lin_out(d2)
            if self.nonneg_out:
                out_raw = F.relu(out_raw)
            return out_raw, z


def run_rnn_experiments():
    device = torch.device("cpu")  # Small sequences run faster on M4 CPU without kernel launch overhead
    datasets_dict = generate_signal_datasets(n_samples=600, seq_len=80, n_channels=64)
    modes = ["vanilla", "orthogonal", "layernorm", "rmsnorm", "skipnorm"]

    all_results = {}
    sample_reconstructions = {}

    for domain, data in datasets_dict.items():
        print(f"\n=== Domain: {domain} (4D Unidirectional Vanilla RNN Bottleneck) ===")
        x_train = data[:480].to(device)
        x_test = data[480:].to(device)
        nonneg = domain in ("PianoRoll", "Speech")

        all_results[domain] = {}
        sample_reconstructions[domain] = {"ground_truth": x_test[0].cpu().numpy()}

        for mode in modes:
            torch.manual_seed(42)
            model = RNNSequenceAutoencoder(in_dim=64, hidden_dim=32, latent_dim=4, mode=mode, nonneg_out=nonneg).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)

            loss_curve = []
            for epoch in range(25):
                model.train()
                perm = torch.randperm(len(x_train))
                for i in range(0, len(x_train), 64):
                    batch = x_train[perm[i : i + 64]]
                    opt.zero_grad()
                    x_hat, _ = model(batch)
                    loss = F.mse_loss(x_hat, batch)
                    loss.backward()
                    opt.step()

                model.eval()
                with torch.no_grad():
                    xh_te, _ = model(x_test)
                    loss_curve.append(float(F.mse_loss(xh_te, x_test).item()))

            model.eval()
            with torch.no_grad():
                xh_te, z_te = model(x_test)
                final_mse = float(F.mse_loss(xh_te, x_test).item())
                r2 = float(1.0 - ((xh_te - x_test) ** 2).sum() / ((x_test - x_test.mean()) ** 2).sum())
                sample_reconstructions[domain][mode] = xh_te[0].cpu().numpy()

            all_results[domain][mode] = {
                "final_mse": final_mse,
                "r2_score": r2,
                "loss_curve": loss_curve,
            }
            print(f"  {mode:12s} | Test MSE: {final_mse:.6f} | R^2: {r2*100:6.2f}%")

        van_mse = all_results[domain]["vanilla"]["final_mse"]
        sn_mse = all_results[domain]["skipnorm"]["final_mse"]
        improvement_pct = ((van_mse - sn_mse) / sn_mse) * 100.0
        print(f"  --> SkipNorm Improvement over Vanilla RNN: +{improvement_pct:.1f}% (MSE ratio {van_mse/sn_mse:.2f}x lower!)")

    # Plot Figure 4: RNN Reconstruction Curves & Qualitative Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), dpi=160)
    domains = ["EEG", "PianoRoll", "Speech"]
    style_map = [
        ("skipnorm", "SkipNorm Vanilla RNN", "#e65100", "-", 2.5),
        ("layernorm", "LayerNorm RNN", "#2ca02c", "--", 1.8),
        ("rmsnorm", "RMSNorm RNN", "#9467bd", "-.", 1.8),
        ("orthogonal", "Orthogonal RNN", "#8c564b", ":", 1.8),
        ("vanilla", "Vanilla RNN", "#1f77b4", "--", 2.0),
    ]

    for col, dom in enumerate(domains):
        ax = axes[0, col]
        for m, label, color, ls, lw in style_map:
            ax.plot(range(1, 26), all_results[dom][m]["loss_curve"], label=label, color=color, linestyle=ls, linewidth=lw)
        van_mse = all_results[dom]["vanilla"]["final_mse"]
        sn_mse = all_results[dom]["skipnorm"]["final_mse"]
        ratio = van_mse / sn_mse
        ax.set_title(f"{dom}: 4D RNN Bottleneck\n(SkipNorm MSE is {ratio:.2f}x lower!)", fontweight="bold")
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Test Reconstruction MSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.legend(fontsize=8)

    # Bottom row: visual comparison on EEG waveform and Speech spectrogram
    # 1. EEG multi-channel trace overlay (top 6 channels)
    ax = axes[1, 0]
    gt_eeg = sample_reconstructions["EEG"]["ground_truth"]
    van_eeg = sample_reconstructions["EEG"]["vanilla"]
    sn_eeg = sample_reconstructions["EEG"]["skipnorm"]
    for ch in range(5):
        offset = ch * 2.5
        ax.plot(gt_eeg[:, ch] + offset, color="black", linewidth=1.3, alpha=0.85, label="Ground Truth" if ch == 0 else None)
        ax.plot(van_eeg[:, ch] + offset, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8, label="Vanilla RNN" if ch == 0 else None)
        ax.plot(sn_eeg[:, ch] + offset, color="#e65100", linestyle="-", linewidth=1.3, alpha=0.9, label="SkipNorm RNN" if ch == 0 else None)
    ax.set_title("EEG Reconstruction (5 Channels, 4D Latent)")
    ax.set_xlabel("Time Step t")
    ax.set_ylabel("Channel Offset Amplitude")
    ax.legend(fontsize=8, loc="upper right")

    # 2. Piano Roll frame norm tracking
    ax = axes[1, 1]
    gt_p = sample_reconstructions["PianoRoll"]["ground_truth"]
    van_p = sample_reconstructions["PianoRoll"]["vanilla"]
    ln_p = sample_reconstructions["PianoRoll"]["layernorm"]
    sn_p = sample_reconstructions["PianoRoll"]["skipnorm"]
    ax.plot(np.linalg.norm(gt_p, axis=-1), color="black", linewidth=2.2, label="True Chord Energy ||x_t||_2")
    ax.plot(np.linalg.norm(sn_p, axis=-1), color="#e65100", linestyle=":", linewidth=2.2, label="SkipNorm ||x_hat_t||_2 (Exact)")
    ax.plot(np.linalg.norm(van_p, axis=-1), color="#1f77b4", linestyle="--", linewidth=1.6, label="Vanilla RNN ||x_hat_t||_2")
    ax.plot(np.linalg.norm(ln_p, axis=-1), color="#2ca02c", linestyle="-.", linewidth=1.6, label="LayerNorm RNN ||x_hat_t||_2")
    ax.set_title("Piano Roll Dynamic Envelope Tracking")
    ax.set_xlabel("Time Step t")
    ax.set_ylabel("Frame Energy ||x_t||_2")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.25)

    # 3. Speech Spectrogram Residual Error Map
    ax = axes[1, 2]
    gt_s = sample_reconstructions["Speech"]["ground_truth"]
    van_s = sample_reconstructions["Speech"]["vanilla"]
    sn_s = sample_reconstructions["Speech"]["skipnorm"]
    combined = np.concatenate([gt_s.T, van_s.T, sn_s.T], axis=0)
    im = ax.imshow(combined, aspect="auto", origin="lower", cmap="magma")
    ax.axhline(64, color="white", linestyle="--", linewidth=1)
    ax.axhline(128, color="white", linestyle="--", linewidth=1)
    ax.set_yticks([32, 96, 160])
    ax.set_yticklabels(["Ground Truth", "Vanilla RNN", "SkipNorm RNN"], fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Time Step t")
    ax.set_title("Speech Spectrogram (Bottom->Top: GT / Vanilla / SkipNorm)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig4_path = RESULTS_DIR / "fig4_rnn_signals_benchmark.png"
    plt.savefig(fig4_path)
    plt.close(fig)
    print(f"Saved {fig4_path}")

    with open(RESULTS_DIR / "exp3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    run_rnn_experiments()
