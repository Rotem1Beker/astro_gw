# infer_online.py
# Inference בזמן אמת על קובץ LIGO מקומי (מותאם לאימון)

import torch
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess
from model.flow import build_flow
from model.conditioning import ContextEncoder

# =========================
# 1. טעינת קונפיגורציה
# =========================
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cpu")

# =========================
# 2. טעינת קובץ LIGO מקומי
# =========================
local_file = "data/H-H1_LOSC_4_V1-1126259446-32.hdf5"

with h5py.File(local_file, "r") as f:
    strain = np.array(f["strain"]["Strain"])
    gps_start = f["meta"]["GPSstart"][()]
    duration = f["meta"]["Duration"][()]

print(f"Loaded LIGO data: {strain.shape}, GPS start: {gps_start}, duration: {duration}s")

# =========================
# 3. Preprocessing
# =========================
X = preprocess(strain)        # FFT / whitening וכו'
X = X[:1025]                  # 🔑 התאמה לאימון
X = torch.tensor(X, dtype=torch.float32).to(device)

print("FFT input shape to encoder:", X.shape)

# =========================
# 4. טעינת מודל מאומן
# =========================
ckpt = torch.load("model.pt", map_location=device)

input_dim = ckpt["encoder"]["net.0.weight"].shape[1]
context_dim = config["model"]["context_dim"]

assert X.shape[0] == input_dim, (
    f"Input dim mismatch: got {X.shape[0]}, expected {input_dim}"
)

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
# 5. Inference
# =========================
with torch.no_grad():
    context = encoder(X.unsqueeze(0))   # (1, context_dim)
    samples = flow.sample(1000, context=context)  # (1, 1000, 2)

# הסרת ממד הבאץ'
samples = samples.squeeze(0)  # (1000, 2)

print("Samples shape:", samples.shape)

# =========================
# 6. ניתוח תוצאות
# =========================
mean_est = samples.mean(dim=0).cpu().numpy()  # (2,)
std_est = samples.std(dim=0).cpu().numpy()    # (2,)

print("\n--- Inference Results ---")
print("Estimated parameter means:", mean_est)
print("Estimated parameter std:", std_est)

print("\n--- Interpretation ---")
print(
    "המודל קיבל קטע מאות LIGO אמיתי, עבר preprocessing זהה לאימון,\n"
    "והפיק התפלגות פוסטריורית על פרמטרי המקור.\n"
    "סטיית תקן קטנה → ביטחון גבוה, סטייה גדולה → אי־ודאות."
)

# =========================
# 7. גרפים
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# אות LIGO
axes[0, 0].plot(strain, color="black", linewidth=0.5)
axes[0, 0].set_title("Measured LIGO strain")
axes[0, 0].set_xlabel("Time sample")
axes[0, 0].set_ylabel("Strain")
axes[0, 0].grid(alpha=0.3)

# ספקטרום כוח
axes[0, 1].plot(np.abs(X.cpu().numpy()), color="blue", linewidth=0.5)
axes[0, 1].set_title("Preprocessed spectrum (input to model)")
axes[0, 1].set_xlabel("Frequency bin")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].set_yscale("log")
axes[0, 1].grid(alpha=0.3)

# פרמטרים מוערכים
axes[1, 0].errorbar(
    [0, 1],
    mean_est,
    yerr=std_est,
    fmt="o",
    capsize=8,
    markersize=8,
    color="darkred"
)
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(["Parameter 1", "Parameter 2"])
axes[1, 0].set_ylabel("Estimated value")
axes[1, 0].set_title("Estimated GW source parameters")
axes[1, 0].grid(alpha=0.3)

# התפלגות פוסטריורית
axes[1, 1].scatter(
    samples[:, 0].cpu().numpy(),
    samples[:, 1].cpu().numpy(),
    alpha=0.3,
    s=5,
    color="purple"
)
axes[1, 1].set_xlabel("Parameter 1")
axes[1, 1].set_ylabel("Parameter 2")
axes[1, 1].set_title("Posterior distribution (1000 samples)")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("inference_results.png", dpi=150)
print("\nSaved plot to inference_results.png")
plt.show()