# train_realistic_waveforms.py
# אימון עם waveforms ריאליסטיים (chirp signals) במקום צעצוע סינוסואידי

import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.flow import build_flow
from model.conditioning import ContextEncoder

# =========================
# Realistic Waveform Generator
# =========================
class ChirpWaveformGenerator:
    """
    מייצר waveforms ריאליסטיים של gravitational waves
    כולל chirp (תדר משתנה) וריקבון אמפליטודה
    """
    
    def __init__(self, sample_rate=4096, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.t = np.linspace(0, duration, self.n_samples)
    
    def generate_chirp(self, m1, m2, distance=410, inclination=0):
        """
        מייצר chirp signal ריאליסטי
        
        Args:
            m1, m2: מסות (solar masses)
            distance: מרחק (Mpc)
            inclination: זוית הטיה (radians)
        
        Returns:
            waveform: h(t) - הגל הכבידתי
        """
        # פרמטרים פיזיקליים
        G = 6.674e-11  # m^3 kg^-1 s^-2
        c = 3e8        # m/s
        M_sun = 1.989e30  # kg
        
        # המרה ליחידות SI
        M1 = m1 * M_sun
        M2 = m2 * M_sun
        M_total = M1 + M2
        M_chirp = (M1 * M2)**(3/5) / M_total**(1/5)  # chirp mass
        
        # תדר ואמפליטודה כפונקציה של הזמן
        # זוהי אפרוקסימציה פשוטה לנוסחאות post-Newtonian
        
        # זמן לפני המיזוג
        t_to_merger = self.duration - self.t
        t_to_merger = np.maximum(t_to_merger, 1e-3)  # מניעת חלוקה באפס
        
        # תדר כפונקציה של זמן (עולה לקראת המיזוג)
        f = (1 / (8 * np.pi)) * (5 / (256 * t_to_merger))**(3/8) * \
            (G * M_chirp / c**3)**(-5/8)
        f = np.clip(f, 20, 500)  # גבולות תדר ריאליסטיים
        
        # אמפליטודה (יורדת עם מרחק, עולה לקראת המיזוג)
        h0 = 1e-21 * (M_chirp / M_sun)**(5/6) / distance
        amplitude = h0 * (f / 100)**(2/3)
        
        # הפאזה (אינטגרל של התדר)
        phase = 2 * np.pi * np.cumsum(f) / self.sample_rate
        
        # Polarizations
        h_plus = amplitude * (1 + np.cos(inclination)**2) * np.cos(phase)
        h_cross = amplitude * 2 * np.cos(inclination) * np.sin(phase)
        
        # Combined strain
        h = h_plus + h_cross
        
        # Taper (חלון) בתחילה ובסוף
        window = np.hanning(self.n_samples)
        h *= window
        
        return h
    
    def add_realistic_noise(self, signal, snr=10):
        """
        מוסיף רעש ריאליסטי (לא לבן, אלא עם PSD דומה ל-LIGO)
        """
        # צבע הרעש: 1/f^α noise + white noise
        freqs = np.fft.rfftfreq(self.n_samples, 1/self.sample_rate)
        
        # LIGO-like PSD (מופשט)
        psd = np.ones_like(freqs)
        psd[freqs > 10] = (freqs[freqs > 10] / 100)**(-2)  # יורד בתדרים נמוכים
        psd[freqs < 10] = 100  # עולה מאוד בתדרים נמוכים מאוד
        
        # רעש בדומיין תדר
        noise_fft = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
        noise_fft *= np.sqrt(psd)
        
        # חזרה לדומיין זמן
        noise = np.fft.irfft(noise_fft, n=self.n_samples)
        
        # נרמול ל-SNR רצוי
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        noise *= np.sqrt(signal_power / (snr**2 * noise_power))
        
        return signal + noise

# =========================
# Dataset
# =========================
class RealisticGWDataset(Dataset):
    """Dataset עם waveforms ריאליסטיים"""
    
    def __init__(self, n_samples=10000, snr=10):
        self.n_samples = n_samples
        self.generator = ChirpWaveformGenerator()
        self.snr = snr
        
        # טווחי פרמטרים
        self.m1_range = (5, 50)   # מסת החור השחור הראשון
        self.m2_range = (5, 50)   # מסת החור השחור השני
        self.dist_range = (100, 1000)  # מרחק ב-Mpc
        self.inc_range = (0, np.pi)    # זוית
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # דגימת פרמטרים אקראיים
        m1 = np.random.uniform(*self.m1_range)
        m2 = np.random.uniform(*self.m2_range)
        
        # חשוב: m1 >= m2 (קונבנציה)
        if m1 < m2:
            m1, m2 = m2, m1
        
        distance = np.random.uniform(*self.dist_range)
        inclination = np.random.uniform(*self.inc_range)
        
        # יצירת waveform
        waveform = self.generator.generate_chirp(m1, m2, distance, inclination)
        noisy_waveform = self.generator.add_realistic_noise(waveform, self.snr)
        
        # FFT preprocessing (כמו באימון המקורי)
        fft = np.fft.rfft(noisy_waveform)[:1025]  # לוקחים רק חלק
        fft_abs = np.abs(fft)
        fft_normalized = fft_abs / (np.max(fft_abs) + 1e-8)
        
        # פרמטרים להעריך (נעבוד רק על המסות כרגע)
        params = np.array([m1, m2], dtype=np.float32)
        
        return torch.tensor(fft_normalized, dtype=torch.float32), \
               torch.tensor(params, dtype=torch.float32)

# =========================
# Training Loop
# =========================
def train_realistic(config_path="config.yaml", epochs=50, batch_size=32):
    """
    אימון מחדש עם waveforms ריאליסטיים
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    
    # Dataset
    train_dataset = RealisticGWDataset(n_samples=10000, snr=10)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    encoder = ContextEncoder(
        input_dim=1025,
        context_dim=config["model"]["context_dim"]
    ).to(device)
    
    flow = build_flow(
        param_dim=2,  # m1, m2
        context_dim=config["model"]["context_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        n_transforms=config["model"]["n_transforms"]
    ).to(device)
    
    # Learning rate - נסה מה-config או ברירת מחדל
    lr = config.get("training", {}).get("lr", 1e-3)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(flow.parameters()),
        lr=lr
    )
    
    print(f"Using learning rate: {lr}")
    
    # Training
    losses = []
    print(f"\n📚 Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for X_batch, params_batch in pbar:
            X_batch = X_batch.to(device)
            params_batch = params_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            context = encoder(X_batch)
            log_prob = flow.log_prob(params_batch, context=context)
            loss = -log_prob.mean()
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")
    
    # Save
    torch.save({
        "encoder": encoder.state_dict(),
        "flow": flow.state_dict(),
        "config": config
    }, "model_realistic.pt")
    
    print("\n✅ Training complete! Saved to model_realistic.pt")
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Training Loss (Realistic Waveforms)")
    plt.grid(alpha=0.3)
    plt.savefig("training_loss_realistic.png", dpi=150)
    print("📊 Saved loss plot to training_loss_realistic.png")
    
    return encoder, flow, losses


# =========================
# Test waveform generation
# =========================
def test_waveform_generation():
    """בדיקת generation של waveforms"""
    gen = ChirpWaveformGenerator(duration=0.5)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Case 1: Low mass
    h1 = gen.generate_chirp(m1=10, m2=10, distance=410)
    axes[0].plot(gen.t, h1)
    axes[0].set_title("Low mass binary (10+10 M☉)")
    axes[0].set_ylabel("Strain")
    axes[0].grid(alpha=0.3)
    
    # Case 2: High mass (like GW150914)
    h2 = gen.generate_chirp(m1=36, m2=29, distance=410)
    axes[1].plot(gen.t, h2)
    axes[1].set_title("GW150914-like binary (36+29 M☉)")
    axes[1].set_ylabel("Strain")
    axes[1].grid(alpha=0.3)
    
    # Case 3: With noise
    h3 = gen.add_realistic_noise(h2, snr=10)
    axes[2].plot(gen.t, h3)
    axes[2].set_title("With realistic noise (SNR=10)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Strain")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("realistic_waveforms_test.png", dpi=150)
    print("✅ Saved test waveforms to realistic_waveforms_test.png")
    plt.show()


if __name__ == "__main__":
    print("🔬 Testing realistic waveform generation...")
    test_waveform_generation()
    
    print("\n" + "="*60)
    response = input("Start full training? (y/n): ")
    if response.lower() == 'y':
        train_realistic(epochs=50, batch_size=32)