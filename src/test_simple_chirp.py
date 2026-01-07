# test_simple_chirp.py
# בדיקה עם chirp פשוט מאוד (ללא פיזיקה מסובכת)

import numpy as np
import matplotlib.pyplot as plt

# פרמטרים
sample_rate = 4096
duration = 0.5
t = np.linspace(0, duration, int(sample_rate * duration))

# Chirp פשוט: f(t) = f0 + (f1-f0)*t/T
def simple_chirp(m1, m2):
    """
    Chirp פשוט שתלוי במסות
    מסות גבוהות → תדרים גבוהים
    """
    # מסה כוללת משפיעה על טווח התדרים
    total_mass = m1 + m2
    
    f_start = 20 + (total_mass - 20) * 2  # תדר התחלתי
    f_end = 100 + (total_mass - 20) * 5    # תדר סופי
    
    # Chirp ליניארי
    f = f_start + (f_end - f_start) * (t / duration)
    
    # אמפליטודה תלויה במסה
    amplitude = 1e-20 * (m1 * m2)**(1/2)
    
    # Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(f) / sample_rate
    
    # Generate signal
    h = amplitude * np.sin(phase)
    
    # Add taper
    taper = np.sin(np.linspace(0, np.pi, len(t)))**2
    h *= taper
    
    return h

# Test cases
cases = [
    (10, 10, "Low mass"),
    (36, 29, "GW150914"),
    (50, 45, "High mass")
]

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for i, (m1, m2, label) in enumerate(cases):
    # Generate
    h = simple_chirp(m1, m2)
    
    # Time domain
    axes[i, 0].plot(t, h, linewidth=0.5)
    axes[i, 0].set_title(f'{label}: Time domain (max={h.max():.2e})')
    axes[i, 0].set_xlabel('Time (s)')
    axes[i, 0].set_ylabel('Strain')
    axes[i, 0].grid(alpha=0.3)
    
    # FFT
    fft = np.fft.rfft(h)
    freqs = np.fft.rfftfreq(len(h), 1/sample_rate)
    fft_abs = np.abs(fft)[:1025]
    
    axes[i, 1].plot(freqs[:1025], fft_abs, linewidth=0.5)
    axes[i, 1].set_title(f'{label}: FFT (max={fft_abs.max():.2e})')
    axes[i, 1].set_xlabel('Frequency (Hz)')
    axes[i, 1].set_ylabel('Amplitude')
    axes[i, 1].set_yscale('log')
    axes[i, 1].grid(alpha=0.3)
    
    print(f"\n{label}:")
    print(f"  Time domain max: {h.max():.2e}")
    print(f"  FFT max: {fft_abs.max():.2e}")
    print(f"  FFT mean: {fft_abs.mean():.2e}")

plt.tight_layout()
plt.savefig('simple_chirp_test.png', dpi=150)
print("\n✅ Saved to simple_chirp_test.png")
plt.show()