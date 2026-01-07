# infer_online.py – inference בזמן אמת על קובץ LIGO מקומי
import torch
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt

from preprocess import preprocess  # ודאי שיש לך פונקציה זו
from model.flow import build_flow
from model.conditioning import ContextEncoder

# --- טען את ההגדרות ---
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# --- נתיב לקובץ LIGO מקומי ---
local_file = "data/H-H1_LOSC_4_V1-1126259446-32.hdf5"

# --- טען את האות מ-LIGO ---
with h5py.File(local_file, "r") as f:
    strain = np.array(f["strain"]["Strain"])
    gps_start = f["meta"]["GPSstart"][()]
    duration = f["meta"]["Duration"][()]

print(f"Loaded LIGO data: {strain.shape}, GPS start: {gps_start}, duration: {duration}s")

# --- preprocessing (למשל whitening, bandpass) ---
X = preprocess(strain)
X = torch.tensor(X, dtype=torch.float32)

# --- טען את המודל והשחזר encoder/flow לפי checkpoint ---
ckpt = torch.load("model.pt")
input_dim = ckpt["encoder"]["net.0.weight"].shape[1]
context_dim = config["model"]["context_dim"]

encoder = ContextEncoder(input_dim, context_dim)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

flow = build_flow(
    param_dim=2,
    context_dim=context_dim,
    hidden_dim=config["model"]["hidden_dim"],
    n_transforms=config["model"]["n_transforms"]
)
flow.load_state_dict(ckpt["flow"])
flow.eval()

# --- צור context מהאות המדידה ---
with torch.no_grad():
    context = encoder(X.unsqueeze(0))  # הוסף batch dimension
    samples = flow.sample(1000, context=context)

# --- ניתוח התוצאות ---
estimated_mean = samples.mean(0).detach().numpy()

print("\n--- Inference Results ---")
print("GW signal (מדידה חדשה):")
print(strain[:500])  # רק תצוגה קצרה של גל
print("\nEstimated mean of posterior samples:")
print(estimated_mean)

# --- אם ידועים true parameters, אפשר להשוות ---
# כאן אין לנו 'true_theta' מ-LIGO אמיתי, אז נסתפק בהערכת הפיזור
std_estimation = samples.std(0).detach().numpy()
print("\nStandard deviation of posterior samples:")
print(std_estimation)

# --- גרפים ---
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(strain, color='black')
plt.title("Measured LIGO GW signal")
plt.xlabel("Time sample")
plt.ylabel("Strain")

plt.subplot(2, 1, 2)
plt.scatter(np.arange(len(estimated_mean)), estimated_mean, color='blue', alpha=0.6, label="Estimated parameters")
plt.errorbar(np.arange(len(estimated_mean)), estimated_mean, yerr=std_estimation, fmt='none', ecolor='red', alpha=0.5, label="Std deviation")
plt.title("Estimated parameters from flow model")
plt.xlabel("Parameter index")
plt.ylabel("Estimated value")
plt.legend()

plt.tight_layout()
plt.show()
