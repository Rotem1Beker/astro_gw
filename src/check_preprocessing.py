# check_preprocessing.py
# בדיקה אם ה-preprocessing הורס את המידע

import numpy as np
import matplotlib.pyplot as plt
from train_realistic_waveforms import ChirpWaveformGenerator

generator = ChirpWaveformGenerator(duration=0.5)

# יצירת waveforms שונים
cases = [
    (10, 10, "Low mass"),
    (36, 29, "GW150914"),
    (50, 45, "High mass")
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, (m1, m2, label) in enumerate(cases):
    waveform = generator.generate_chirp(m1, m2, distance=410, inclination=0)
    
    # Time domain
    axes[i, 0].plot(generator.t, waveform, linewidth=0.5)
    axes[i, 0].set_title(f'{label}: Time domain')
    axes[i, 0].set_xlabel('Time (s)')
    axes[i, 0].set_ylabel('Strain')
    axes[i, 0].grid(alpha=0.3)
    
    # FFT (raw)
    fft = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(len(waveform), 1/generator.sample_rate)
    fft_abs = np.abs(fft)
    
    axes[i, 1].plot(freqs[:1025], fft_abs[:1025], linewidth=0.5)
    axes[i, 1].set_title(f'{label}: FFT (raw)')
    axes[i, 1].set_xlabel('Frequency (Hz)')
    axes[i, 1].set_ylabel('Amplitude')
    axes[i, 1].set_yscale('log')
    axes[i, 1].grid(alpha=0.3)
    
    # FFT (normalized - what model sees)
    # 🔧 Use the FIXED preprocessing (no standardization!)
    fft_log = np.log10(fft_abs[:1025] + 1e-25)
    fft_clipped = np.clip(fft_log, -25, -10)
    fft_normalized = (fft_clipped + 25) / 15.0
    
    axes[i, 2].plot(freqs[:1025], fft_normalized, linewidth=0.5)
    axes[i, 2].set_title(f'{label}: FFT (normalized)')
    axes[i, 2].set_xlabel('Frequency (Hz)')
    axes[i, 2].set_ylabel('Normalized amplitude')
    axes[i, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_check.png', dpi=150)
print("✅ Saved to preprocessing_check.png")

# Statistics
print("\n" + "="*60)
print("FFT STATISTICS (what encoder sees)")
print("="*60)

for m1, m2, label in cases:
    waveform = generator.generate_chirp(m1, m2, distance=410, inclination=0)
    fft = np.fft.rfft(waveform)
    fft_abs = np.abs(fft)[:1025]
    # 🔧 Use FIXED preprocessing (no standardization!)
    fft_log = np.log10(fft_abs + 1e-25)
    fft_clipped = np.clip(fft_log, -25, -10)
    fft_normalized = (fft_clipped + 25) / 15.0
    
    print(f"\n{label} ({m1}+{m2} M☉):")
    print(f"  Mean:   {fft_normalized.mean():.6f}")
    print(f"  Std:    {fft_normalized.std():.6f}")
    print(f"  Max:    {fft_normalized.max():.6f}")
    print(f"  Min:    {fft_normalized.min():.6f}")
    print(f"  L2 norm: {np.linalg.norm(fft_normalized):.6f}")

# Pairwise differences
print("\n" + "="*60)
print("PAIRWISE DIFFERENCES")
print("="*60)

waveforms_processed = []
for m1, m2, label in cases:
    waveform = generator.generate_chirp(m1, m2, distance=410, inclination=0)
    fft = np.fft.rfft(waveform)
    fft_abs = np.abs(fft)[:1025]
    # 🔧 Use FIXED preprocessing (no standardization!)
    fft_log = np.log10(fft_abs + 1e-25)
    fft_clipped = np.clip(fft_log, -25, -10)
    fft_normalized = (fft_clipped + 25) / 15.0
    waveforms_processed.append((label, fft_normalized))

for i in range(len(waveforms_processed)):
    for j in range(i+1, len(waveforms_processed)):
        label1, w1 = waveforms_processed[i]
        label2, w2 = waveforms_processed[j]
        
        l2_dist = np.linalg.norm(w1 - w2)
        cosine_sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        
        print(f"\n{label1} <-> {label2}:")
        print(f"  L2 distance:     {l2_dist:.6f}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

# If distances are very small, preprocessing is the problem
min_dist = min([np.linalg.norm(waveforms_processed[i][1] - waveforms_processed[j][1]) 
                for i in range(len(waveforms_processed)) 
                for j in range(i+1, len(waveforms_processed))])

if min_dist < 0.1:
    print("❌ Preprocessing DESTROYS information!")
    print("   All waveforms look nearly identical after normalization.")
    print("\n   Solutions:")
    print("   1. Don't normalize by max (use fixed scale)")
    print("   2. Use log-scale instead of linear")
    print("   3. Keep raw FFT amplitudes")
    print("   4. Use whitening instead of normalization")
else:
    print("✅ Preprocessing preserves some information")
    print(f"   Min distance: {min_dist:.6f}")

plt.show()