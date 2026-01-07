# diagnose_training.py
# אבחון מה המודל באמת למד

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train_realistic_waveforms import ChirpWaveformGenerator, RealisticGWDataset
import yaml

from model.flow import build_flow
from model.conditioning import ContextEncoder

# טעינת config ומודל
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cpu")
model_path = "model_realistic.pt"

ckpt = torch.load(model_path, map_location=device)
input_dim = ckpt["encoder"]["net.0.weight"].shape[1]
context_dim = config["model"]["context_dim"]

encoder = ContextEncoder(input_dim, context_dim).to(device)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

flow = build_flow(
    param_dim=2,
    context_dim=context_dim,
    hidden_dim=config["model"]["hidden_dim"],
    n_transforms=config["model"]["n_transforms"]
).to(device)
flow.load_state_dict(ckpt["flow"])
flow.eval()

# =========================
# Test 1: האם המודל מבחין בין מסות שונות?
# =========================
print("="*60)
print("TEST 1: Can the model distinguish different masses?")
print("="*60)

generator = ChirpWaveformGenerator(duration=0.5)

test_cases = [
    (10, 10, "Low mass (10+10 M☉)"),
    (20, 15, "Medium mass (20+15 M☉)"),
    (36, 29, "GW150914-like (36+29 M☉)"),
    (50, 45, "High mass (50+45 M☉)"),
]

results = []

for m1, m2, label in test_cases:
    # Generate clean waveform (NO noise)
    waveform = generator.generate_chirp(m1, m2, distance=410, inclination=0)
    
    # Preprocess
    fft = np.fft.rfft(waveform)[:1025]
    fft_abs = np.abs(fft)
    fft_normalized = fft_abs / (np.max(fft_abs) + 1e-8)
    X = torch.tensor(fft_normalized, dtype=torch.float32).to(device)
    
    # Inference
    with torch.no_grad():
        context = encoder(X.unsqueeze(0))
        samples = flow.sample(500, context=context).squeeze(0).cpu().numpy()
    
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    
    results.append({
        'true': [m1, m2],
        'pred_mean': mean,
        'pred_std': std,
        'label': label,
        'samples': samples
    })
    
    print(f"\n{label}:")
    print(f"  True:      m1={m1:.1f}, m2={m2:.1f}")
    print(f"  Predicted: m1={mean[0]:.1f}±{std[0]:.1f}, m2={mean[1]:.1f}±{std[1]:.1f}")
    error_m1 = abs(mean[0] - m1)
    error_m2 = abs(mean[1] - m2)
    print(f"  Error:     Δm1={error_m1:.1f}, Δm2={error_m2:.1f}")

# =========================
# Test 2: Prior vs Posterior
# =========================
print("\n" + "="*60)
print("TEST 2: Prior vs Posterior comparison")
print("="*60)

# Sample from prior (no conditioning)
with torch.no_grad():
    # Create dummy context (all zeros)
    dummy_context = torch.zeros(1, context_dim).to(device)
    prior_samples = flow.sample(1000, context=dummy_context).squeeze(0).cpu().numpy()

prior_mean = prior_samples.mean(axis=0)
prior_std = prior_samples.std(axis=0)

print(f"\nPrior (no conditioning):")
print(f"  Mean: m1={prior_mean[0]:.1f}, m2={prior_mean[1]:.1f}")
print(f"  Std:  m1={prior_std[0]:.1f}, m2={prior_std[1]:.1f}")

print(f"\nPosterior (on GW150914-like signal):")
gw_result = results[2]  # GW150914-like case
print(f"  Mean: m1={gw_result['pred_mean'][0]:.1f}, m2={gw_result['pred_mean'][1]:.1f}")
print(f"  Std:  m1={gw_result['pred_std'][0]:.1f}, m2={gw_result['pred_std'][1]:.1f}")

kl_estimate = np.sum((gw_result['pred_mean'] - prior_mean)**2 / (prior_std**2 + 1e-6))
print(f"\nKL divergence estimate: {kl_estimate:.2f}")
if kl_estimate < 1.0:
    print("  ⚠️  WARNING: Posterior is too similar to prior!")
    print("     The model is not learning from the data.")

# =========================
# Test 3: Context vectors
# =========================
print("\n" + "="*60)
print("TEST 3: Are context vectors different?")
print("="*60)

contexts = []
for m1, m2, label in test_cases:
    waveform = generator.generate_chirp(m1, m2, distance=410, inclination=0)
    fft = np.fft.rfft(waveform)[:1025]
    fft_abs = np.abs(fft)
    fft_normalized = fft_abs / (np.max(fft_abs) + 1e-8)
    X = torch.tensor(fft_normalized, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        context = encoder(X.unsqueeze(0))
        contexts.append(context.cpu().numpy().flatten())

contexts = np.array(contexts)

print(f"\nContext vector statistics:")
print(f"  Shape: {contexts.shape}")
print(f"  Mean: {contexts.mean():.4f}")
print(f"  Std:  {contexts.std():.4f}")

# Pairwise distances
print(f"\nPairwise distances between contexts:")
for i in range(len(test_cases)):
    for j in range(i+1, len(test_cases)):
        dist = np.linalg.norm(contexts[i] - contexts[j])
        print(f"  {test_cases[i][2]} <-> {test_cases[j][2]}: {dist:.4f}")

if contexts.std() < 0.1:
    print("\n  ⚠️  WARNING: Contexts are nearly identical!")
    print("     The encoder is not extracting useful features.")

# =========================
# Visualization
# =========================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Predicted vs True for each case
for idx, result in enumerate(results[:4]):
    ax = axes[idx // 2, idx % 2]
    ax.scatter(result['samples'][:, 0], result['samples'][:, 1],
              alpha=0.3, s=5, c='blue')
    ax.scatter(result['true'][0], result['true'][1],
              s=200, c='red', marker='*', edgecolors='black', linewidths=2,
              label='True', zorder=10)
    ax.scatter(result['pred_mean'][0], result['pred_mean'][1],
              s=150, c='orange', marker='x', linewidths=3,
              label='Predicted mean', zorder=10)
    ax.set_xlabel('m1 (M☉)')
    ax.set_ylabel('m2 (M☉)')
    ax.set_title(result['label'])
    ax.legend()
    ax.grid(alpha=0.3)

# Row 2: Prior vs Posterior overlay
ax = axes[1, 2]
ax.scatter(prior_samples[:, 0], prior_samples[:, 1],
          alpha=0.2, s=5, c='gray', label='Prior')
ax.scatter(results[2]['samples'][:, 0], results[2]['samples'][:, 1],
          alpha=0.3, s=5, c='blue', label='Posterior (GW150914)')
ax.set_xlabel('m1 (M☉)')
ax.set_ylabel('m2 (M☉)')
ax.set_title('Prior vs Posterior')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_diagnosis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved diagnosis plot to model_diagnosis.png")

# =========================
# Summary
# =========================
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)

errors = [abs(r['pred_mean'][0] - r['true'][0]) + abs(r['pred_mean'][1] - r['true'][1]) 
          for r in results]
avg_error = np.mean(errors)

print(f"\nAverage mass prediction error: {avg_error:.1f} M☉")

if avg_error > 10:
    print("❌ Model is NOT learning the mapping from waveforms to masses")
    print("\nPossible issues:")
    print("  1. Prior collapse - model ignores conditioning")
    print("  2. Preprocessing destroys chirp information")
    print("  3. Training data quality (SNR too low, unrealistic waveforms)")
    print("  4. Architectural issues (encoder too simple, flow capacity)")
elif contexts.std() < 0.1:
    print("⚠️  Encoder is not extracting useful features")
    print("    Context vectors are nearly identical for different inputs")
else:
    print("✅ Model shows some learning, but needs improvement")

plt.show()