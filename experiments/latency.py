# מדידת זמן
import time
import torch

start = time.time()
_ = torch.randn(1000, 2)
print("Latency:", time.time() - start)
