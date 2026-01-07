# diagnose_loss.py
# אבחון למה ה-loss לא יורד

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from train_realistic_waveforms import RealisticGWDataset
from model.flow import build_flow
from model.conditioning import ContextEncoder

# טעינת config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*60)
print("LOSS DIAGNOSIS")
print("="*60)

# =========================
# Test 1: Dataset quality
# =========================
print("\n📦 TEST 1: Dataset Quality")
print("-"*60)

dataset = RealisticGWDataset(n_samples=100, snr=10)

X_samples = []
params_samples = []

for i in range(10):
    X, params = dataset[i]
    X_samples.append(X.numpy())
    params_samples.append(params.numpy())

X_samples = np.array(X_samples)
params_samples = np.array(params_samples)

print(f"Input X statistics:")
print(f"  Shape: {X_samples.shape}")
print(f"  Mean: {X_samples.mean():.4f}")
print(f"  Std: {X_samples.std():.4f}")
print(f"  Min: {X_samples.min():.4f}")
print(f"  Max: {X_samples.max():.4f}")

print(f"\nParameter statistics:")
print(f"  m1 range: [{params_samples[:, 0].min():.1f}, {params_samples[:, 0].max():.1f}]")
print(f"  m2 range: [{params_samples[:, 1].min():.1f}, {params_samples[:, 1].max():.1f}]")
print(f"  m1 mean: {params_samples[:, 0].mean():.1f} ± {params_samples[:, 0].std():.1f}")
print(f"  m2 mean: {params_samples[:, 1].mean():.1f} ± {params_samples[:, 1].std():.1f}")

# Check if inputs are too similar
pairwise_dists = []
for i in range(len(X_samples)):
    for j in range(i+1, len(X_samples)):
        dist = np.linalg.norm(X_samples[i] - X_samples[j])
        pairwise_dists.append(dist)

print(f"\nInput diversity:")
print(f"  Mean pairwise distance: {np.mean(pairwise_dists):.4f}")
print(f"  Min distance: {np.min(pairwise_dists):.4f}")

if np.mean(pairwise_dists) < 1.0:
    print("  ⚠️  WARNING: Inputs are very similar!")

# =========================
# Test 2: Model initialization
# =========================
print("\n\n🏗️  TEST 2: Model Initialization")
print("-"*60)

encoder = ContextEncoder(
    input_dim=1025,
    context_dim=config["model"]["context_dim"]
).to(device)

flow = build_flow(
    param_dim=2,
    context_dim=config["model"]["context_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    n_transforms=config["model"]["n_transforms"]
).to(device)

# Test forward pass
X_batch = torch.tensor(X_samples[:5], dtype=torch.float32).to(device)
params_batch = torch.tensor(params_samples[:5], dtype=torch.float32).to(device)

with torch.no_grad():
    context = encoder(X_batch)
    log_prob = flow.log_prob(params_batch, context=context)

print(f"Initial context statistics:")
print(f"  Mean: {context.mean().item():.6f}")
print(f"  Std: {context.std().item():.6f}")
print(f"  Max: {context.abs().max().item():.6f}")

print(f"\nInitial log probabilities:")
print(f"  Mean: {log_prob.mean().item():.4f}")
print(f"  Std: {log_prob.std().item():.4f}")
print(f"  Range: [{log_prob.min().item():.4f}, {log_prob.max().item():.4f}]")

if context.std().item() < 0.01:
    print("  ⚠️  WARNING: Context vectors are nearly zero!")
    print("     Encoder might have vanishing gradients")

if log_prob.mean().item() < -20:
    print("  ⚠️  WARNING: Initial log probs are extremely negative!")
    print("     Flow initialization might be poor")

# =========================
# Test 3: Prior vs data distribution
# =========================
print("\n\n📊 TEST 3: Prior vs Data Distribution")
print("-"*60)

# Sample from prior (no context)
with torch.no_grad():
    dummy_context = torch.zeros(100, config["model"]["context_dim"]).to(device)
    prior_samples = flow.sample(1, context=dummy_context).squeeze(1).cpu().numpy()

print(f"Prior distribution:")
print(f"  m1: {prior_samples[:, 0].mean():.1f} ± {prior_samples[:, 0].std():.1f}")
print(f"  m2: {prior_samples[:, 1].mean():.1f} ± {prior_samples[:, 1].std():.1f}")

print(f"\nData distribution:")
print(f"  m1: {params_samples[:, 0].mean():.1f} ± {params_samples[:, 0].std():.1f}")
print(f"  m2: {params_samples[:, 1].mean():.1f} ± {params_samples[:, 1].std():.1f}")

prior_mean = prior_samples.mean(axis=0)
data_mean = params_samples.mean(axis=0)
mismatch = np.linalg.norm(prior_mean - data_mean)

print(f"\nDistribution mismatch: {mismatch:.2f}")

if mismatch > 20:
    print("  ⚠️  WARNING: Prior is very far from data!")
    print("     Consider better initialization or wider prior")

# =========================
# Test 4: Gradient flow
# =========================
print("\n\n⚡ TEST 4: Gradient Flow")
print("-"*60)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(flow.parameters()),
    lr=1e-3
)

# Single training step
X_batch = torch.tensor(X_samples[:5], dtype=torch.float32).to(device)
params_batch = torch.tensor(params_samples[:5], dtype=torch.float32).to(device)

optimizer.zero_grad()
context = encoder(X_batch)
log_prob = flow.log_prob(params_batch, context=context)
loss = -log_prob.mean()
loss.backward()

# Check gradients
encoder_grads = []
flow_grads = []

for name, param in encoder.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        encoder_grads.append(grad_norm)

for name, param in flow.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        flow_grads.append(grad_norm)

print(f"Encoder gradients:")
print(f"  Mean: {np.mean(encoder_grads):.6f}")
print(f"  Max: {np.max(encoder_grads):.6f}")
print(f"  Min: {np.min(encoder_grads):.6f}")

print(f"\nFlow gradients:")
print(f"  Mean: {np.mean(flow_grads):.6f}")
print(f"  Max: {np.max(flow_grads):.6f}")
print(f"  Min: {np.min(flow_grads):.6f}")

if np.max(encoder_grads) < 1e-6:
    print("  ⚠️  WARNING: Encoder gradients are vanishing!")

if np.max(flow_grads) < 1e-6:
    print("  ⚠️  WARNING: Flow gradients are vanishing!")

if np.max(encoder_grads) > 100 or np.max(flow_grads) > 100:
    print("  ⚠️  WARNING: Gradients are exploding!")
    print("     Consider gradient clipping")

# =========================
# Test 5: Overfitting check
# =========================
print("\n\n🎯 TEST 5: Can Model Overfit Single Sample?")
print("-"*60)

# Try to overfit a single example
encoder_small = ContextEncoder(1025, config["model"]["context_dim"]).to(device)
flow_small = build_flow(2, config["model"]["context_dim"], 
                        config["model"]["hidden_dim"], 
                        config["model"]["n_transforms"]).to(device)

optimizer_small = torch.optim.Adam(
    list(encoder_small.parameters()) + list(flow_small.parameters()),
    lr=1e-3
)

X_single = X_batch[0:1]
params_single = params_batch[0:1]

losses = []
for step in range(200):
    optimizer_small.zero_grad()
    context = encoder_small(X_single)
    log_prob = flow_small.log_prob(params_single, context=context)
    loss = -log_prob.mean()
    loss.backward()
    optimizer_small.step()
    losses.append(loss.item())

print(f"Overfitting single sample:")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Improvement: {losses[0] - losses[-1]:.4f}")

if losses[-1] > losses[0] - 1.0:
    print("  ❌ FAIL: Cannot overfit even a single sample!")
    print("     Model architecture or optimization is broken")
else:
    print("  ✅ PASS: Model can learn (at least overfit)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(losses)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Overfitting Single Sample')
axes[0].grid(alpha=0.3)

# Distribution comparison
axes[1].scatter(params_samples[:, 0], params_samples[:, 1], 
               alpha=0.5, label='Data', s=50)
axes[1].scatter(prior_samples[:, 0], prior_samples[:, 1], 
               alpha=0.3, label='Prior', s=30)
axes[1].set_xlabel('m1 (M☉)')
axes[1].set_ylabel('m2 (M☉)')
axes[1].set_title('Prior vs Data Distribution')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_diagnosis.png', dpi=150)
print("\n✅ Saved diagnosis to loss_diagnosis.png")

# =========================
# Summary
# =========================
print("\n" + "="*60)
print("SUMMARY & RECOMMENDATIONS")
print("="*60)

issues = []

if np.mean(pairwise_dists) < 1.0:
    issues.append("Inputs too similar - preprocessing problem")

if context.std().item() < 0.01:
    issues.append("Encoder outputs vanishing - need stronger architecture")

if np.max(encoder_grads) < 1e-6:
    issues.append("Vanishing gradients in encoder")

if losses[-1] > losses[0] - 1.0:
    issues.append("Cannot overfit single sample - fundamental problem")

if mismatch > 20:
    issues.append("Prior-data mismatch - need better initialization")

if issues:
    print("\n🔴 Issues found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 Suggested fixes:")
    if "Inputs too similar" in str(issues):
        print("  → Fix preprocessing (we just did this!)")
    if "Encoder outputs vanishing" in str(issues):
        print("  → Use stronger encoder (CNN, attention)")
    if "Vanishing gradients" in str(issues):
        print("  → Add LayerNorm, use better initialization")
    if "Cannot overfit" in str(issues):
        print("  → Check model architecture, learning rate")
    if "Prior-data mismatch" in str(issues):
        print("  → Better flow initialization, wider prior")
else:
    print("\n✅ No major issues detected!")
    print("   Loss should decrease with more training")

plt.show()