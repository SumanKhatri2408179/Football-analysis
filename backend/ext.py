import torch
print("CUDA available:", torch.cuda.is_available())          # ✅ True
print("GPU Name:", torch.cuda.get_device_name(0))            # ✅ NVIDIA GeForce RTX 3060 Ti
