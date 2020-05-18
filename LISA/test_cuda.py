import torch
import numpy as np

print(torch.cuda.is_available())

print(torch.cuda.current_blas_handle())

print(torch.cuda.device_count())

print(torch.cuda.get_device_capability(device=None))

print(torch.cuda.get_device_name(device=None))

print(torch.cuda.ipc_collect())

print(torch.cuda.memory_summary(device=None, abbreviated=False))

print(torch.cuda.max_memory_allocated(device=None))

print(torch.cuda.max_memory_reserved(device=None))

