# analyze_event_window.py
# השוואה בין חלון האירוע GW150914 לבין רעש רקע

import torch
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from preprocess import preprocess
from model.flow import build_flow
from model.conditioning import ContextEncoder

# =========================
# קונפיגורציה
# =========================
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cpu")
local_file = "data/H-H1_LOSC_4_V1-1126259446-32.hdf5"

# =========================
# פרמטרי האירוע GW150914
# =========================
# זמן האירוע המדויק (GPS time)
EVENT_GPS = 1126259462.4  # זמן מרכז האירוע
WINDOW_SIZE = 0.5  # ±0.25s חלון סביב האירוע (2048 samples @ 4096Hz)
SAMPLE_RATE = 4096  # Hz

# =========================
# טעינת נתונים
# =========================
with h5py.File(local_file, "r") as f:
    strain = np.array(f["strain"]["Strain"])
    gps_start = f["meta"]["GPSstart"][()]
    duration = f["meta"]["Duration"][()]

# המרה לזמן יחסי
event_time_rel = EVENT_GPS - gps_start  # זמן האירוע ביחס לתחילת הקובץ
event_idx = int(event_time_rel * SAMPLE_RATE)
window_samples = int(WINDOW_SIZE * SAMPLE_RATE)

# חלון סביב האירוע
event_start = max(0, event_idx - window_samples // 2)
event_end = min(len(strain), event_idx + window_samples // 2)
signal_window = strain[event_start:event_end]

# חלון רעש (מתחילת הקובץ, רחוק מהאירוע)
noise_start = int(2.0 * SAMPLE_RATE)  # מתחיל אחרי 2 שניות
noise_end = noise_start + window_samples
noise_window = strain[noise_start:noise_end]

print(f"Event window: {len(signal_window)} samples (~{len(signal_window)/SAMPLE_RATE:.3f}s)")
print(f"Noise window: {len(noise_window)} samples (~{len(noise_window)/SAMPLE_RATE:.3f}s)")
print(f"Event GPS time: {EVENT_GPS}, relative time: {event_time_rel:.3f}s, index: {event_idx}")

# בדיקת גודל FFT
test_fft = preprocess(signal_window)
print(f"FFT output size: {len(test_fft)} (will be padded/truncated to 1025)")

# =========================
# טעינת מודל
# =========================
ckpt = torch.load("model.pt", map_location=device)
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
# Inference על שני החלונות
# =========================
def run_inference(data_window, n_samples=1000):
    """מריץ inference על חלון נתונים"""
    X = preprocess(data_window)
    
    # 🔑 Padding/truncation ל-1025 תדרים (להתאמה למודל)
    if len(X) < 1025:
        # Pad with zeros
        X = np.pad(X, (0, 1025 - len(X)), mode='constant')
    else:
        # Truncate
        X = X[:1025]
    
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        context = encoder(X.unsqueeze(0))
        samples = flow.sample(n_samples, context=context).squeeze(0)
    
    return samples.cpu().numpy()

print("\n🔍 Running inference on signal window...")
signal_samples = run_inference(signal_window)
signal_mean = signal_samples.mean(axis=0)
signal_std = signal_samples.std(axis=0)

print("🔍 Running inference on noise window...")
noise_samples = run_inference(noise_window)
noise_mean = noise_samples.mean(axis=0)
noise_std = noise_samples.std(axis=0)

# =========================
# חישוב Bayes Factor (פשוט)
# =========================
# Log-likelihood הערכה פשוטה: כמה קומפקטית ההתפלגות
signal_det = np.linalg.det(np.cov(signal_samples.T) + 1e-6 * np.eye(2))
noise_det = np.linalg.det(np.cov(noise_samples.T) + 1e-6 * np.eye(2))
log_bayes_factor = -0.5 * (np.log(signal_det) - np.log(noise_det))

print(f"\n📊 Results:")
print(f"Signal posterior: mean={signal_mean}, std={signal_std}")
print(f"Noise posterior: mean={noise_mean}, std={noise_std}")
print(f"Log Bayes Factor (signal vs noise): {log_bayes_factor:.2f}")
print(f"  → {'SIGNAL DETECTED' if log_bayes_factor > 5 else 'Inconclusive/Noise'}")

# =========================
# ויזואליזציה
# =========================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Row 1: Time series
ax1 = fig.add_subplot(gs[0, 0])
time_signal = np.arange(len(signal_window)) / SAMPLE_RATE
ax1.plot(time_signal, signal_window, 'r-', linewidth=0.5)
ax1.axvline(x=len(signal_window)/(2*SAMPLE_RATE), color='black', linestyle='--', alpha=0.5, label='Event time')
ax1.set_title('Signal Window (±0.2s around GW150914)', fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Strain')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
time_noise = np.arange(len(noise_window)) / SAMPLE_RATE
ax2.plot(time_noise, noise_window, 'b-', linewidth=0.5)
ax2.set_title('Noise Window (background)', fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Strain')
ax2.grid(alpha=0.3)

# Row 2: Spectrograms
ax3 = fig.add_subplot(gs[1, 0])
f, t, Sxx = signal.spectrogram(signal_window, SAMPLE_RATE, nperseg=256)
ax3.pcolormesh(t, f, np.log10(Sxx + 1e-20), shading='gouraud', cmap='inferno')
ax3.set_ylabel('Frequency (Hz)')
ax3.set_xlabel('Time (s)')
ax3.set_title('Signal Spectrogram')
ax3.set_ylim([20, 500])

ax4 = fig.add_subplot(gs[1, 1])
f, t, Sxx = signal.spectrogram(noise_window, SAMPLE_RATE, nperseg=256)
ax4.pcolormesh(t, f, np.log10(Sxx + 1e-20), shading='gouraud', cmap='inferno')
ax4.set_ylabel('Frequency (Hz)')
ax4.set_xlabel('Time (s)')
ax4.set_title('Noise Spectrogram')
ax4.set_ylim([20, 500])

# Row 3: Posterior distributions (2D)
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(signal_samples[:, 0], signal_samples[:, 1], alpha=0.3, s=5, c='red', label='Signal')
ax5.scatter(signal_mean[0], signal_mean[1], s=100, c='darkred', marker='x', linewidths=3)
ax5.set_xlabel('Parameter 1')
ax5.set_ylabel('Parameter 2')
ax5.set_title('Signal Posterior Distribution')
ax5.legend()
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(noise_samples[:, 0], noise_samples[:, 1], alpha=0.3, s=5, c='blue', label='Noise')
ax6.scatter(noise_mean[0], noise_mean[1], s=100, c='darkblue', marker='x', linewidths=3)
ax6.set_xlabel('Parameter 1')
ax6.set_ylabel('Parameter 2')
ax6.set_title('Noise Posterior Distribution')
ax6.legend()
ax6.grid(alpha=0.3)

# Row 4: Comparison
ax7 = fig.add_subplot(gs[3, :])
ax7.scatter(signal_samples[:, 0], signal_samples[:, 1], alpha=0.3, s=10, c='red', label='Signal posterior')
ax7.scatter(noise_samples[:, 0], noise_samples[:, 1], alpha=0.3, s=10, c='blue', label='Noise posterior')
ax7.scatter(signal_mean[0], signal_mean[1], s=150, c='darkred', marker='x', linewidths=4, label='Signal mean')
ax7.scatter(noise_mean[0], noise_mean[1], s=150, c='darkblue', marker='+', linewidths=4, label='Noise mean')
ax7.set_xlabel('Parameter 1')
ax7.set_ylabel('Parameter 2')
ax7.set_title(f'Overlay: Signal vs Noise | Log BF = {log_bayes_factor:.2f}', fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

plt.savefig('event_vs_noise_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved plot to event_vs_noise_analysis.png")
plt.show()