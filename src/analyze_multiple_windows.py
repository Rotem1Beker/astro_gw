# analyze_multiple_windows.py
# ניתוח של כמה חלונות שונים - signal vs multiple noise regions

import torch
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import preprocess
from model.flow import build_flow
from model.conditioning import ContextEncoder

# קונפיגורציה
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cpu")
local_file = "data/H-H1_LOSC_4_V1-1126259446-32.hdf5"

EVENT_GPS = 1126259462.4
SAMPLE_RATE = 4096
WINDOW_SIZE = 0.5

# טעינת נתונים
with h5py.File(local_file, "r") as f:
    strain = np.array(f["strain"]["Strain"])
    gps_start = f["meta"]["GPSstart"][()]

event_time_rel = EVENT_GPS - gps_start
event_idx = int(event_time_rel * SAMPLE_RATE)
window_samples = int(WINDOW_SIZE * SAMPLE_RATE)

# טעינת מודל
model_path = "model_realistic.pt" if Path("model_realistic.pt").exists() else "model.pt"
print(f"Loading model: {model_path}\n")
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

# פונקציית inference
def run_inference(data_window, n_samples=1000):
    X = preprocess(data_window)
    if len(X) < 1025:
        X = np.pad(X, (0, 1025 - len(X)), mode='constant')
    else:
        X = X[:1025]
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        context = encoder(X.unsqueeze(0))
        samples = flow.sample(n_samples, context=context).squeeze(0)
    return samples.cpu().numpy()

# חלון האירוע
event_start = max(0, event_idx - window_samples // 2)
event_end = min(len(strain), event_idx + window_samples // 2)
signal_window = strain[event_start:event_end]

# כמה חלונות רעש
noise_windows = []
noise_times = [2.0, 5.0, 8.0, 12.0, 20.0, 25.0]  # שניות אחרי תחילת הקובץ

for t in noise_times:
    start = int(t * SAMPLE_RATE)
    end = start + window_samples
    if end < len(strain) and abs(t - event_time_rel) > 2.0:  # רחוק מהאירוע
        noise_windows.append((t, strain[start:end]))

print(f"🎯 Signal window at t={event_time_rel:.1f}s")
print(f"🔇 Analyzing {len(noise_windows)} noise windows\n")

# Inference
print("Running inference...")
signal_samples = run_inference(signal_window)
signal_mean = signal_samples.mean(axis=0)
signal_std = signal_samples.std(axis=0)

results = []
for i, (t, window) in enumerate(noise_windows):
    samples = run_inference(window)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    results.append({
        'time': t,
        'samples': samples,
        'mean': mean,
        'std': std
    })
    print(f"  Noise {i+1} @ t={t:.1f}s: mean={mean}, std={std}")

print(f"\n  Signal @ t={event_time_rel:.1f}s: mean={signal_mean}, std={signal_std}")

# ויזואליזציה
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Signal posterior
ax = axes[0]
ax.scatter(signal_samples[:, 0], signal_samples[:, 1], 
          alpha=0.5, s=10, c='red', label='Signal')
ax.scatter(signal_mean[0], signal_mean[1], 
          s=150, c='darkred', marker='x', linewidths=4)
ax.set_xlabel('m1 (M☉)')
ax.set_ylabel('m2 (M☉)')
ax.set_title(f'SIGNAL @ t={event_time_rel:.1f}s', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Noise posteriors
colors = ['blue', 'green', 'orange', 'purple', 'brown']
for i, (res, color) in enumerate(zip(results[:5], colors)):
    ax = axes[i+1]
    ax.scatter(res['samples'][:, 0], res['samples'][:, 1],
              alpha=0.5, s=10, c=color, label=f"Noise @ t={res['time']:.1f}s")
    ax.scatter(res['mean'][0], res['mean'][1],
              s=150, c='black', marker='x', linewidths=4)
    ax.set_xlabel('m1 (M☉)')
    ax.set_ylabel('m2 (M☉)')
    ax.set_title(f"Noise @ t={res['time']:.1f}s")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('multiple_windows_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved to multiple_windows_analysis.png")

# סטטיסטיקות
print("\n" + "="*60)
print("📊 SUMMARY STATISTICS")
print("="*60)
print(f"Signal posterior volume: {np.linalg.det(np.cov(signal_samples.T)):.2e}")
for i, res in enumerate(results):
    vol = np.linalg.det(np.cov(res['samples'].T))
    dist = np.linalg.norm(signal_mean - res['mean'])
    print(f"Noise {i+1} @ t={res['time']:.1f}s:")
    print(f"  Volume: {vol:.2e}, Distance from signal: {dist:.2f}")

plt.show()