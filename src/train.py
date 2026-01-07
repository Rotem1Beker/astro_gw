# אימון המודל
import yaml
import torch
from torch.utils.data import DataLoader
from dataset import GWDataset
from model.flow import build_flow
from model.conditioning import ContextEncoder

with open("config.yaml") as f:
    config = yaml.safe_load(f)

dataset = GWDataset(2000, config)
loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

sample_X, _ = dataset[0]
encoder = ContextEncoder(len(sample_X), config["model"]["context_dim"])
flow = build_flow(
    param_dim=2,
    context_dim=config["model"]["context_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    n_transforms=config["model"]["n_transforms"],
)

optimizer = torch.optim.Adam(
    list(flow.parameters()) + list(encoder.parameters()),
    lr=config["train"]["lr"],
)

for epoch in range(config["train"]["epochs"]):
    total_loss = 0
    for X, theta in loader:
        context = encoder(X)
        loss = -flow.log_prob(theta, context=context).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}: loss = {total_loss:.3f}")

torch.save({"flow": flow.state_dict(), "encoder": encoder.state_dict()}, "model.pt")
